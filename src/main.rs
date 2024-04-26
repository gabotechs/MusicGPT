use clap::Parser;
use ndarray::Array;
use tokenizers::Tokenizer;

use crate::input_ids::InputIds;
use tensor::Tensor;

use crate::logits::Logits;
use crate::past_key_values::{PastKeyValues, PastKeyValuesConfig};
use crate::session_input_builder::SessionInputsBuilder;

mod input_ids;
mod logits;
mod past_key_values;
mod session_input_builder;
mod tensor;

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

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "make a pop song")]
    prompt: String,
}

struct MusicGen {
    tokenizer: Tokenizer,
    text_encoder: ort::Session,
    decoder_model_merged: ort::Session,
    delay_pattern_mask: ort::Session,
    audio_encodec_decode: ort::Session,
}

impl MusicGen {
    fn load() -> ort::Result<Self> {
        let mut tokenizer = Tokenizer::from_file("onnx/tokenizer.json")?;

        tokenizer.with_padding(None).with_truncation(None)?;

        let text_encoder = ort::Session::builder()?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_inter_threads(4)?
            .commit_from_file("onnx/text_encoder.onnx")?;

        let decoder = ort::Session::builder()?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_inter_threads(4)?
            .commit_from_file("onnx/decoder_model_merged.onnx")?;

        let delay_pattern_mask = ort::Session::builder()?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_inter_threads(4)?
            .commit_from_file("onnx/build_delay_pattern_mask.onnx")?;

        let audio_encodec_decode = ort::Session::builder()?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_inter_threads(4)?
            .commit_from_file("onnx/encodec_decode.onnx")?;

        Ok(Self {
            tokenizer,
            text_encoder,
            decoder_model_merged: decoder,
            delay_pattern_mask,
            audio_encodec_decode,
        })
    }

    /// Tokenizes a text using on `self.tokenizer`.
    ///
    /// # Arguments
    ///
    /// * `text`: the text to be tokenized
    ///
    /// returns: A tuple with:
    /// - The tokens extracted from the text
    /// - An attention mask
    fn tokenize(&self, text: &str) -> ort::Result<(Tensor<i64>, Tensor<i64>)> {
        let input_ids = self
            .tokenizer
            .encode(text, true)?
            .get_ids()
            .iter()
            .map(|e| *e as i64)
            .collect::<Vec<i64>>();

        let seq_len = input_ids.len();
        let input_ids =
            Array::from_shape_vec((1, seq_len), input_ids).expect("Could not reshape inputs");
        let attention_mask: Array<i64, _> = Array::ones((1, seq_len));
        Ok((
            Tensor::from_array(input_ids),
            Tensor::from_array(attention_mask),
        ))
    }

    fn encode_tokens(
        &self,
        input_ids: &Tensor<i64>,
        attention_mask: &Tensor<i64>,
    ) -> ort::Result<Tensor<f32>> {
        let mut output: ort::SessionOutputs = self
            .text_encoder
            .run(ort::inputs![input_ids.view(), attention_mask.view()]?)?;

        Tensor::try_from(output.remove("last_hidden_state").unwrap())
    }

    fn generation_step(
        &self,
        last_hidden_state: &Tensor<f32>,
        attention_mask: &Tensor<i64>,
    ) -> ort::Result<ort::DynValue> {
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

        loop {
            let prepared_input_ids = input_ids
                .apply_delay_pattern_mask(PAD_TOKEN_ID)
                .last()
                .unsqueeze(1)
                .dupe_along_first_dim();

            println!("input_ids: {prepared_input_ids:?}\n");
            println!("encoder_hidden_states: {encoder_hidden_states:?}\n");
            println!("encoder_attention_mask: {encoder_attention_mask:?}\n");

            let has_past_key_values = !past_key_values.is_empty();

            let outputs = self.decoder_model_merged.run(
                SessionInputsBuilder::start()
                    .add(encoder_attention_mask.view())
                    .add(prepared_input_ids.view())
                    .add(encoder_hidden_states.view())
                    .add_many(past_key_values.view_all())
                    .add(Tensor::bool(has_past_key_values).view())
                    .end(),
            )?;

            // logits: float32[batch_size,decoder_sequence_length,2048]
            let logits = Logits::from_3d_dyn_value(&outputs[0])?;
            let logits = logits.apply_free_guidance(GUIDANCE_SCALE);
            println!("logits: {logits:?}\n");

            input_ids.push(logits.sample(TOP_K).iter().map(|e| e.0));

            println!("generated input_ids: {input_ids:?}\n");

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
                    past_key_values.set(i, 0, Tensor::try_from(&outputs[idx])?);
                    past_key_values.set(i, 1, Tensor::try_from(&outputs[idx + 1])?);
                    // past_key_values.set(i, 2, Tensor::try_from(&outputs[idx + 2])?); optimization consist on omitting the replacement of the "encoder" key-values
                    // past_key_values.set(i, 3, Tensor::try_from(&outputs[idx + 3])?);
                } else {
                    past_key_values.set(i, 0, Tensor::try_from(&outputs[idx])?);
                    past_key_values.set(i, 1, Tensor::try_from(&outputs[idx + 1])?);
                    past_key_values.set(i, 2, Tensor::try_from(&outputs[idx + 2])?);
                    past_key_values.set(i, 3, Tensor::try_from(&outputs[idx + 3])?);
                }
            }
        }
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let music_gen = MusicGen::load()?;

    let (input_ids, attention_mask) = music_gen.tokenize(&args.prompt)?;

    let last_hidden_state = music_gen.encode_tokens(&input_ids, &attention_mask)?;

    let generated = music_gen.generation_step(&last_hidden_state, &attention_mask)?;
    println!("generated: {generated:?}\n");

    Ok(())
}
