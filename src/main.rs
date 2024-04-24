use std::ops::Mul;

use clap::Parser;
use ndarray::{Array, Axis};
use ort::SessionInputs;
use tokenizers::Tokenizer;

use crate::logits::Logits;
use crate::session_input_builder::SessionInputsBuilder;
use tensor::Tensor;

mod logits;
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
        let mut tokenizer = Tokenizer::from_file("musicgen_onnx/tokenizer.json")?;

        tokenizer.with_padding(None).with_truncation(None)?;

        let text_encoder = ort::Session::builder()?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_inter_threads(4)?
            .commit_from_file("musicgen_onnx/text_encoder.onnx")?;

        let decoder = ort::Session::builder()?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_inter_threads(4)?
            .commit_from_file("musicgen_onnx/decoder_model_merged.onnx")?;

        let delay_pattern_mask = ort::Session::builder()?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_inter_threads(4)?
            .commit_from_file("musicgen_onnx/build_delay_pattern_mask.onnx")?;

        let audio_encodec_decode = ort::Session::builder()?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_inter_threads(4)?
            .commit_from_file("musicgen_onnx/encodec_decode.onnx")?;

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
        let input_ids = Array::from(input_ids)
            .into_shape((1, seq_len))
            .expect("Could not reshape inputs");
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

    fn empty_past_key_values() -> Vec<Tensor<f32>> {
        let mut values = vec![];
        let encoder_dims = (1, NUM_ENCODER_HEADS, 0, ENCODER_DIM_KV);
        let decoder_dims = (1, NUM_DECODER_HEADS, 0, DECODER_DIM_KV);
        for _ in 0..NUM_DECODER_LAYERS {
            // "past_key_values.0.decoder.key"
            // "past_key_values.0.decoder.value"
            // "past_key_values.0.encoder.key"
            // "past_key_values.0.encoder.value"

            // format!("past_key_values.{i}.decoder.key"),
            values.push(Tensor::from_array(Array::<f32, _>::zeros(decoder_dims)));
            // format!("past_key_values.{i}.decoder.value"),
            values.push(Tensor::from_array(Array::<f32, _>::zeros(decoder_dims)));
            // format!("past_key_values.{i}.encoder.key"),
            values.push(Tensor::from_array(Array::<f32, _>::zeros(encoder_dims)));
            // format!("past_key_values.{i}.encoder.value"),
            values.push(Tensor::from_array(Array::<f32, _>::zeros(encoder_dims)));
        }
        values
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

        let mut input_ids = Tensor::from_value((4, 1), PAD_TOKEN_ID).dupe_along_first_dim();
        let mut use_cache_branch = Tensor::bool(false);
        let mut past_key_values = Self::empty_past_key_values();

        loop {
            let outputs = self.decoder_model_merged.run(
                SessionInputsBuilder::start()
                    .add(encoder_attention_mask.view())
                    .add(input_ids.view())
                    .add(encoder_hidden_states.view())
                    .add_many(past_key_values.iter().map(|e| e.view()))
                    .add(use_cache_branch.view())
                    .end(),
            )?;

            // logits: float32[batch_size,decoder_sequence_length,2048]
            let logits = Logits::from_3d_dyn_value(&outputs[0])?;
            let logits = logits.apply_free_guidance(GUIDANCE_SCALE);
            println!("logits: {logits:?}\n");

            let mut generated_input_ids = vec![];
            for (token_id, _) in logits.sample(TOP_K) {
                generated_input_ids.push(token_id)
            }

            let input_ids_arr = Array::from_shape_vec((4, 1), generated_input_ids)
                .expect("Could not generate input_ids array");
            input_ids = Tensor::from_array(input_ids_arr).dupe_along_first_dim();
            println!("generated input_ids: {input_ids:?}\n");

            past_key_values = Vec::with_capacity(outputs.len() - 1);
            for i in 1..outputs.len() {
                past_key_values.push(Tensor::try_from(&outputs[i])?)
            }
            use_cache_branch = Tensor::bool(true);
        }

        todo!()
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let music_gen = MusicGen::load()?;

    let (input_ids, attention_mask) = music_gen.tokenize(&args.prompt)?;

    println!("input_ids: {input_ids:?}");
    println!("attention_mask: {attention_mask:?}");
    let last_hidden_state = music_gen.encode_tokens(&input_ids, &attention_mask)?;
    println!("last_hidden_state: {last_hidden_state:?}");

    let generated = music_gen.generation_step(&last_hidden_state, &attention_mask)?;
    println!("generated: {generated:?}");

    Ok(())
}
