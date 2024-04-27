use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use std::fmt::{Debug, Write};
use std::path::Path;
use std::sync::Arc;

use ort::{
    CUDAExecutionProvider, CoreMLExecutionProvider, DirectMLExecutionProvider,
    GraphOptimizationLevel, TensorRTExecutionProvider,
};
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tokio::sync::mpsc::UnboundedReceiver;

use crate::input_ids::InputIds;
use crate::logits::Logits;
use crate::past_key_values::{PastKeyValues, PastKeyValuesConfig};
use crate::session_input_builder::SessionInputsBuilder;
use crate::tensor::Tensor;

// TODO: this are hardcoded now, maybe they are in one of the config.json files?
const PAD_TOKEN_ID: i64 = 2048;
const NUM_ENCODER_HEADS: usize = 16;
const NUM_DECODER_HEADS: usize = 16;
const NUM_ENCODER_LAYERS: usize = 24;
const NUM_DECODER_LAYERS: usize = 24;
const ENCODER_DIM_KV: usize = 64;
const DECODER_DIM_KV: usize = 64;
const GUIDANCE_SCALE: usize = 3;
const TOP_K: usize = 50;
const MAX_LEN: usize = 300;

async fn build_session<P: AsRef<Path> + Debug>(path: P) -> ort::Result<ort::Session> {
    println!("Loading {path:?}...");
    ort::Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_optimization_level(GraphOptimizationLevel::Level2)?
        .with_inter_threads(8)?
        .with_execution_providers([
            // Prefer TensorRT over CUDA.
            TensorRTExecutionProvider::default().build(),
            CUDAExecutionProvider::default().build(),
            // Use DirectML on Windows if NVIDIA EPs are not available
            DirectMLExecutionProvider::default().build(),
            // Or use ANE on Apple platforms
            CoreMLExecutionProvider::default().build(),
        ])?
        .commit_from_file(path)
}

pub struct MusicGen {
    tokenizer: Tokenizer,
    text_encoder: ort::Session,
    decoder_model_merged: Arc<ort::Session>,
    delay_pattern_mask: ort::Session,
    audio_encodec_decode: ort::Session,
}

impl MusicGen {
    pub async fn load() -> ort::Result<Self> {
        println!("Loading tokenizer...");
        let mut tokenizer = Tokenizer::from_file("onnx/tokenizer.json")?;
        tokenizer.with_padding(None).with_truncation(None)?;

        let result = tokio::join!(
            build_session("onnx/text_encoder.onnx"),
            build_session("onnx/decoder_model_merged.onnx"),
            build_session("onnx/build_delay_pattern_mask.onnx"),
            build_session("onnx/encodec_decode.onnx")
        );
        Ok(Self {
            tokenizer,
            text_encoder: result.0?,
            decoder_model_merged: Arc::new(result.1?),
            delay_pattern_mask: result.2?,
            audio_encodec_decode: result.3?,
        })
    }

    fn make_bar() -> ProgressBar {
        let pb = ProgressBar::new(MAX_LEN as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
            )
            .unwrap()
            .with_key("eta", |state: &ProgressState, w: &mut dyn Write| {
                write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap()
            })
            .progress_chars("#>-"),
        );
        pb
    }

    pub fn generate(&self, text: &str) -> ort::Result<UnboundedReceiver<[i64; 4]>> {
        println!("Tokenizing...");
        let tokens = self
            .tokenizer
            .encode(text, true)
            .expect("Error tokenizing text")
            .get_ids()
            .iter()
            .map(|e| *e as i64)
            .collect::<Vec<_>>();

        let tokens_len = tokens.len();
        let input_ids = Tensor::from_shape_vec((1, tokens_len), tokens);
        let attention_mask = Tensor::<i64>::ones((1, tokens_len));

        println!("Tokenization finished, encoding text...");
        let mut output = self.text_encoder.run(
            SessionInputsBuilder::start()
                .add(input_ids.into_inner())
                .add(attention_mask.view())
                .end(),
        )?;

        let last_hidden_state: Tensor<f32> = output
            .remove("last_hidden_state")
            .expect("last_hidden_state not found in output")
            .try_into()?;

        // Apparently, there's a setting in huggingface's transformers that says that
        // if `guidance_scale` > 1 then you should concatenate 0 along the first axis.
        let encoder_hidden_states = last_hidden_state.dupe_zeros_along_first_dim();
        let encoder_attention_mask = attention_mask.dupe_zeros_along_first_dim();

        let mut input_ids = InputIds::<4>::new();
        input_ids.push([PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID]);
        let mut past_key_values = PastKeyValues::empty(PastKeyValuesConfig {
            num_encoder_heads: NUM_ENCODER_HEADS,
            num_decoder_heads: NUM_DECODER_HEADS,
            encoder_dim_kv: ENCODER_DIM_KV,
            decoder_dim_kv: DECODER_DIM_KV,
            num_decoder_layers: NUM_DECODER_LAYERS,
        });

        let (tx, rx) = mpsc::unbounded_channel::<[i64; 4]>();
        let decoder_model_merged = self.decoder_model_merged.clone();

        println!("Text encoded, generating...");
        tokio::spawn(async move {
            let bar = Self::make_bar();
            for i in 0..MAX_LEN {
                bar.set_position(i as u64);
                let prepared_input_ids = input_ids
                    .apply_delay_pattern_mask(PAD_TOKEN_ID)
                    .last()
                    .unsqueeze(1)
                    .dupe_along_first_dim();

                let has_past_key_values = !past_key_values.is_empty();

                let outputs = decoder_model_merged
                    .run(
                        SessionInputsBuilder::start()
                            .add(encoder_attention_mask.view())
                            .add(prepared_input_ids.into_inner())
                            .add(encoder_hidden_states.view())
                            .add_many(past_key_values.view_all())
                            .add(Tensor::bool(has_past_key_values).into_inner())
                            .end(),
                    )
                    .expect("Error running inference on decoder_model_merged");

                // logits: float32[batch_size,decoder_sequence_length,2048]
                let logits = Logits::from_3d_dyn_value(&outputs[0])
                    .expect("Error extracting logits from onnx runtime value");
                let logits = logits.apply_free_guidance(GUIDANCE_SCALE);

                input_ids.push(logits.sample(TOP_K).iter().map(|e| e.0));

                if let Some(last_de_delayed) = input_ids.last_de_delayed() {
                    let sent = tx.send(last_de_delayed);
                    if sent.is_err() {
                        break;
                    }
                }

                // `outputs` is:
                // [
                //   logits,
                //   past_key_values.0.decoder.key,
                //   past_key_values.0.decoder.value,
                //   past_key_values.0.encoder.key,
                //   past_key_values.0.encoder.value,
                //   ...,
                //   past_key_values.n.decoder.key,
                //   past_key_values.n.decoder.value,
                //   past_key_values.n.encoder.key,
                //   past_key_values.n.encoder.value
                // ]
                for i in 0..(outputs.len() - 1) / 4 {
                    let idx = (i * 4) + 1;
                    if has_past_key_values {
                        // Optimization introduced by optimum to reuse past key values. So, we just replace the constant
                        // outputs with the previous past key values.
                        // https://github.com/huggingface/optimum/blob/0bf2c05fb7e1182b52d21b703cfc95fd9e4ea3dc/optimum/onnxruntime/base.py#L677-L704
                        // TODO: too many unwraps
                        past_key_values.set(i, 0, Tensor::try_from(&outputs[idx]).unwrap());
                        past_key_values.set(i, 1, Tensor::try_from(&outputs[idx + 1]).unwrap());
                        // past_key_values.set(i, 2, Tensor::try_from(&outputs[idx + 2])?); optimization consist on omitting the replacement of the "encoder" key-values
                        // past_key_values.set(i, 3, Tensor::try_from(&outputs[idx + 3])?);
                    } else {
                        past_key_values.set(i, 0, Tensor::try_from(&outputs[idx]).unwrap());
                        past_key_values.set(i, 1, Tensor::try_from(&outputs[idx + 1]).unwrap());
                        past_key_values.set(i, 2, Tensor::try_from(&outputs[idx + 2]).unwrap());
                        past_key_values.set(i, 3, Tensor::try_from(&outputs[idx + 3]).unwrap());
                    }
                }
            }
            bar.finish_with_message("done")
        });

        Ok(rx)
    }

    pub fn encode_audio(&self, input_ids: [Vec<i64>; 4]) -> ort::Result<Vec<f32>> {
        let seq_len = input_ids[0].len();
        let mut data = Vec::with_capacity(seq_len * 4);
        for batch in input_ids {
            for n in batch {
                data.push(n)
            }
        }
        let tensor = Tensor::from_shape_vec((1, 1, 4, seq_len), data);

        let mut outputs = self
            .audio_encodec_decode
            .run(ort::inputs![tensor.into_inner()]?)?;

        let result: Tensor<f32> = outputs
            .remove("audio_values")
            .expect("audio_values not found in output")
            .try_into()?;

        let result = result.squeeze(0).squeeze(0).into_inner().into_raw_vec();
        Ok(result)
    }
}
