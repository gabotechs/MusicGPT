use std::fmt::Debug;
use std::fs;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use num_traits::Zero;
use tokenizers::Tokenizer;

use crate::delay_pattern_mask_ids::DelayedPatternMaskIds;
use crate::music_gen_audio_encodec::MusicGenAudioEncodec;
use crate::music_gen_config::MusicGenConfig;
use crate::music_gen_inputs::MusicGenInputs;
use crate::music_gen_outputs::MusicGenOutputs;
use crate::music_gen_text_encoder::MusicGenTextEncoder;
use crate::tensor_ops::{dupe_zeros_along_first_dim, zeros_tensor};

pub trait MusicGenType: ort::IntoTensorElementType + Debug + Clone + Zero {}

impl MusicGenType for u8 {}
impl MusicGenType for i8 {}
impl MusicGenType for f32 {}
impl MusicGenType for half::f16 {}

// TODO: is this configurable?
const GUIDANCE_SCALE: usize = 3;
// I calculated this myself.
pub const INPUT_IDS_BATCH_PER_SECOND: usize = 50;

pub async fn build_session<P: AsRef<Path> + Debug>(path: P) -> ort::Result<ort::Session> {
    ort::Session::builder()?
        .with_intra_threads(4)?
        .commit_from_file(path)
}

pub struct MusicGen<T: MusicGenType> {
    text_encoder: MusicGenTextEncoder,
    audio_encodec: MusicGenAudioEncodec,
    decoder_model_merged: Arc<ort::Session>,
    config: MusicGenConfig,
    _phantom_data: PhantomData<T>,
}

pub struct MusicGenLoadOptions {
    pub tokenizer: PathBuf,
    pub text_encoder: PathBuf,
    pub decoder_model_merged: PathBuf,
    pub audio_encodec_decode: PathBuf,
    pub config: PathBuf,
}

impl<T: MusicGenType + 'static> MusicGen<T> {
    pub async fn load(opts: MusicGenLoadOptions) -> ort::Result<Self> {
        let mut tokenizer = Tokenizer::from_file(opts.tokenizer)?;
        tokenizer.with_padding(None).with_truncation(None)?;

        let config = fs::read_to_string(opts.config).expect("Error reading config file from disk");
        let config = serde_json::from_str(&config).expect("Could not deserialize config file");

        let result = tokio::join!(
            build_session(opts.text_encoder),
            build_session(opts.decoder_model_merged),
            build_session(opts.audio_encodec_decode)
        );

        Ok(Self {
            text_encoder: MusicGenTextEncoder {
                tokenizer,
                text_encoder: result.0?,
            },
            decoder_model_merged: Arc::new(result.1?),
            audio_encodec: MusicGenAudioEncodec {
                audio_encodec_decode: result.2?,
            },
            config,
            _phantom_data: PhantomData,
        })
    }

    fn generate_tokens(
        &self,
        text: &str,
        max_len: usize,
    ) -> ort::Result<tokio::sync::mpsc::Receiver<ort::Result<[i64; 4]>>> {
        let (last_hidden_state, encoder_attention_mask) = self.text_encoder.encode(text)?;

        // Apparently, there's a setting in huggingface's transformers that says that
        // if `guidance_scale` > 1 then you should concatenate 0 along the first axis.
        let encoder_hidden_states = dupe_zeros_along_first_dim::<T>(last_hidden_state.downcast()?)?;
        let encoder_attention_mask =
            dupe_zeros_along_first_dim::<i64>(encoder_attention_mask.downcast()?)?;

        let mut delay_pattern_mask_ids = DelayedPatternMaskIds::<4>::new();

        let decoder_model_merged = self.decoder_model_merged.clone();

        let mut inputs = MusicGenInputs::new();
        inputs.encoder_attention_mask(encoder_attention_mask)?;
        inputs.encoder_hidden_states(encoder_hidden_states)?;

        let num_hidden_layers = self.config.decoder.num_hidden_layers;
        let num_attention_heads = self.config.decoder.num_attention_heads;
        let pad_token_id = self.config.decoder.pad_token_id;
        let d_kv = self.config.text_encoder.d_kv;
        let top_k = self.config.decoder.top_k;
        let decoder_dims = [1, num_attention_heads, 0, d_kv];
        let encoder_dims = [1, num_attention_heads, 0, d_kv];

        // TODO: 100?
        let (tx, rx) = tokio::sync::mpsc::channel::<ort::Result<[i64; 4]>>(100);
        let tx2 = tx.clone();
        let future = async move {
            inputs.input_ids(ort::Tensor::from_array(([8, 1], vec![pad_token_id; 8]))?)?;

            for i in 0..num_hidden_layers {
                inputs.past_key_value_decoder_key(i, zeros_tensor::<T>(&decoder_dims))?;
                inputs.past_key_value_decoder_value(i, zeros_tensor::<T>(&decoder_dims))?;
                inputs.past_key_value_encoder_key(i, zeros_tensor::<T>(&encoder_dims))?;
                inputs.past_key_value_encoder_value(i, zeros_tensor::<T>(&encoder_dims))?;
            }
            inputs.use_cache_branch(false);
            for _ in 0..max_len {
                let outputs = decoder_model_merged.run(inputs.ort())?;
                let mut outputs = MusicGenOutputs::new(outputs);

                delay_pattern_mask_ids.push(
                    outputs
                        .take_logits()?
                        .apply_free_guidance(GUIDANCE_SCALE)
                        .sample(top_k)
                        .iter()
                        .map(|e| e.0),
                );

                let [a, b, c, d] = delay_pattern_mask_ids.last_delayed_masked(pad_token_id);
                inputs.input_ids(ort::Tensor::from_array((
                    [8, 1],
                    vec![a, b, c, d, a, b, c, d],
                ))?)?;

                if let Some(last_de_delayed) = delay_pattern_mask_ids.last_de_delayed() {
                    let sent = tx.send(Ok(last_de_delayed)).await;
                    if sent.is_err() {
                        break;
                    }
                }

                for j in 0..num_hidden_layers {
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
            Ok::<(), ort::Error>(())
        };

        tokio::spawn(async move {
            if let Err(err) = future.await {
                let _ = tx2.send(Err(err)).await;
            };
        });

        Ok(rx)
    }

    pub async fn generate<Cb: Fn(usize, usize)>(
        &self,
        text: &str,
        secs: usize,
        cb: Cb,
    ) -> ort::Result<tokio::sync::mpsc::Receiver<ort::Result<f32>>> {
        let max_len = secs * INPUT_IDS_BATCH_PER_SECOND;
        let generator = self.generate_tokens(text, max_len)?;
        self.audio_encodec.encode(generator, max_len, cb).await
    }
}

