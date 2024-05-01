use std::fmt::{Debug, Write};
use std::path::Path;
use std::sync::Arc;

use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use ndarray::Array;
use num_traits::Zero;
use ort::{
    CUDAExecutionProvider, CoreMLExecutionProvider, DirectMLExecutionProvider,
    TensorRTExecutionProvider,
};
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tokio::sync::mpsc::UnboundedReceiver;

use crate::delay_pattern_mask_ids::DelayedPatternMaskIds;
use crate::music_gen_inputs::MusicGenInputs;
use crate::music_gen_outputs::MusicGenOutputs;
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
const MAX_LEN: usize = 1000;

async fn build_session<P: AsRef<Path> + Debug>(path: P) -> ort::Result<ort::Session> {
    println!("Loading {path:?}...");
    ort::Session::builder()?
        .with_intra_threads(8)?
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

    pub fn generate(&self, text: &str) -> ort::Result<UnboundedReceiver<ort::Result<[i64; 4]>>> {
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
        let mut output = self
            .text_encoder
            .run(ort::inputs![input_ids.into_inner(), attention_mask.view()]?)?;

        let last_hidden_state: Tensor<f32> = output
            .remove("last_hidden_state")
            .expect("last_hidden_state not found in output")
            .try_into()?;

        // Apparently, there's a setting in huggingface's transformers that says that
        // if `guidance_scale` > 1 then you should concatenate 0 along the first axis.
        let encoder_hidden_states = last_hidden_state.dupe_zeros_along_first_dim();
        let encoder_attention_mask = attention_mask.dupe_zeros_along_first_dim();

        let mut delay_pattern_mask_ids = DelayedPatternMaskIds::<4>::new();

        let (tx, rx) = mpsc::unbounded_channel::<ort::Result<[i64; 4]>>();

        let decoder_model_merged = self.decoder_model_merged.clone();
        let tx2 = tx.clone();

        let future = async move {
            let mut inputs = MusicGenInputs::new();
            inputs.encoder_attention_mask(encoder_attention_mask.into_inner())?;
            inputs.input_ids(ort::Tensor::from_array(([8, 1], vec![PAD_TOKEN_ID; 8]))?)?;
            inputs.encoder_hidden_states(encoder_hidden_states.into_inner())?;

            let decoder_dims = &[1, NUM_DECODER_HEADS, 0, DECODER_DIM_KV];
            let encoder_dims = &[1, NUM_ENCODER_HEADS, 0, ENCODER_DIM_KV];
            for i in 0..NUM_DECODER_LAYERS {
                inputs.past_key_value_decoder_key(i, zeros_tensor::<f32>(decoder_dims))?;
                inputs.past_key_value_decoder_value(i, zeros_tensor::<f32>(decoder_dims))?;
                inputs.past_key_value_encoder_key(i, zeros_tensor::<f32>(encoder_dims))?;
                inputs.past_key_value_encoder_value(i, zeros_tensor::<f32>(encoder_dims))?;
            }
            inputs.use_cache_branch(false);
            let bar = Self::make_bar();
            for i in 0..MAX_LEN {
                if i > 0 {
                    bar.set_position(i as u64);
                }

                let outputs = decoder_model_merged.run(inputs.ort())?;
                let mut outputs = MusicGenOutputs::new(outputs);

                delay_pattern_mask_ids.push(
                    outputs
                        .take_logits()?
                        .apply_free_guidance(GUIDANCE_SCALE)
                        .sample(TOP_K)
                        .iter()
                        .map(|e| e.0),
                );

                let [a, b, c, d] = delay_pattern_mask_ids.last_delayed_masked(PAD_TOKEN_ID);
                inputs.input_ids(ort::Tensor::from_array((
                    [8, 1],
                    vec![a, b, c, d, a, b, c, d],
                ))?)?;

                if let Some(last_de_delayed) = delay_pattern_mask_ids.last_de_delayed() {
                    let sent = tx.send(Ok(last_de_delayed));
                    if sent.is_err() {
                        break;
                    }
                }

                for j in 0..NUM_DECODER_LAYERS {
                    let v = outputs.take_present_decoder_key(j);
                    inputs.past_key_value_decoder_key(j, v)?;
                    let v = outputs.take_present_decoder_value(j);
                    inputs.past_key_value_decoder_value(j, v)?;
                    if !inputs.use_cache_branch {
                        // Optimization introduced by optimum to reuse past key values. So, we just replace the constant
                        // outputs with the previous past key values.
                        // https://github.com/huggingface/optimum/blob/0bf2c05fb7e1182b52d21b703cfc95fd9e4ea3dc/optimum/onnxruntime/base.py#L677-L704
                        let v = outputs.take_present_encoder_key(j);
                        inputs.past_key_value_encoder_key(j, v)?;
                        let v = outputs.take_present_encoder_value(j);
                        inputs.past_key_value_encoder_value(j, v)?;
                    }
                }

                inputs.use_cache_branch(true);
            }
            bar.finish_with_message("done");
            Ok::<(), ort::Error>(())
        };

        tokio::spawn(async move {
            if let Err(err) = future.await {
                let _ = tx2.send(Err(err));
            };
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

fn zeros_tensor<T: ort::IntoTensorElementType + Debug + Clone + Zero + 'static>(
    shape: &[usize],
) -> ort::Tensor<T> {
    ort::Value::from_array(Array::<T, _>::zeros(shape)).expect("Could not build zeros tensor")
}
