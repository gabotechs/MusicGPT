use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::Arc;

use num_traits::Zero;

use crate::delay_pattern_mask_ids::DelayedPatternMaskIds;
use crate::music_gen_config::MusicGenConfig;
use crate::music_gen_inputs::MusicGenInputs;
use crate::music_gen_outputs::MusicGenOutputs;
use crate::tensor_ops::{dupe_zeros_along_first_dim, zeros_tensor};

pub trait MusicGenType: ort::IntoTensorElementType + Debug + Clone + Zero {}

impl MusicGenType for u8 {}
impl MusicGenType for i8 {}
impl MusicGenType for f32 {}
impl MusicGenType for half::f16 {}

// TODO: is this configurable?
const GUIDANCE_SCALE: usize = 3;

pub trait MusicGenDecoder: Send + Sync {
    fn generate_tokens(
        &self,
        last_hidden_state: ort::DynValue,
        encoder_attention_mask: ort::DynValue,
        abort_controller: Arc<AtomicBool>,
        max_len: usize,
    ) -> ort::Result<Receiver<ort::Result<[i64; 4]>>>;
}

pub struct MusicGenMergedDecoder<T: MusicGenType> {
    pub decoder_model_merged: Arc<ort::Session>,
    pub config: MusicGenConfig,
    pub _phantom_data: PhantomData<T>,
}

unsafe impl<T: MusicGenType> Send for MusicGenMergedDecoder<T> {}
unsafe impl<T: MusicGenType> Sync for MusicGenMergedDecoder<T> {}

impl<T: MusicGenType + 'static> MusicGenDecoder for MusicGenMergedDecoder<T> {
    fn generate_tokens(
        &self,
        last_hidden_state: ort::DynValue,
        encoder_attention_mask: ort::DynValue,
        abort_controller: Arc<AtomicBool>,
        max_len: usize,
    ) -> ort::Result<Receiver<ort::Result<[i64; 4]>>> {
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
        let (tx, rx) = std::sync::mpsc::channel::<ort::Result<[i64; 4]>>();
        let tx2 = tx.clone();

        std::thread::spawn(move || {
            let result = {
                inputs.input_ids(ort::Tensor::from_array(([8, 1], vec![pad_token_id; 8]))?)?;

                for i in 0..num_hidden_layers {
                    inputs.past_key_value_decoder_key(i, zeros_tensor::<T>(&decoder_dims))?;
                    inputs.past_key_value_decoder_value(i, zeros_tensor::<T>(&decoder_dims))?;
                    inputs.past_key_value_encoder_key(i, zeros_tensor::<T>(&encoder_dims))?;
                    inputs.past_key_value_encoder_value(i, zeros_tensor::<T>(&encoder_dims))?;
                }
                inputs.use_cache_branch(false);
                for _ in 0..max_len {
                    if abort_controller.load(Ordering::Relaxed) {
                        return Err(ort::Error::CustomError("Aborted".into()));
                    }
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
                        let sent = tx.send(Ok(last_de_delayed));
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
            if let Err(err) = result {
                let _ = tx2.send(Err(err));
            }
            Ok(())
        });

        Ok(rx)
    }
}

pub struct MusicGenSplitDecoder<T: MusicGenType> {
    pub decoder_model: ort::Session,
    pub decoder_with_past_model: Arc<ort::Session>,
    pub config: MusicGenConfig,
    pub _phantom_data: PhantomData<T>,
}

unsafe impl<T: MusicGenType> Send for MusicGenSplitDecoder<T> {}
unsafe impl<T: MusicGenType> Sync for MusicGenSplitDecoder<T> {}

impl<T: MusicGenType + 'static> MusicGenDecoder for MusicGenSplitDecoder<T> {
    fn generate_tokens(
        &self,
        last_hidden_state: ort::DynValue,
        encoder_attention_mask: ort::DynValue,
        abort_controller: Arc<AtomicBool>,
        max_len: usize,
    ) -> ort::Result<Receiver<ort::Result<[i64; 4]>>> {
        // Apparently, there's a setting in huggingface's transformers that says that
        // if `guidance_scale` > 1 then you should concatenate 0 along the first axis.
        let encoder_hidden_states = dupe_zeros_along_first_dim::<T>(last_hidden_state.downcast()?)?;
        let encoder_attention_mask =
            dupe_zeros_along_first_dim::<i64>(encoder_attention_mask.downcast()?)?;

        let mut delay_pattern_mask_ids = DelayedPatternMaskIds::<4>::new();

        let num_hidden_layers = self.config.decoder.num_hidden_layers;
        let pad_token_id = self.config.decoder.pad_token_id;
        let top_k = self.config.decoder.top_k;

        let mut inputs = MusicGenInputs::new();
        inputs.encoder_attention_mask(encoder_attention_mask)?;
        inputs.input_ids(ort::Tensor::from_array(([8, 1], vec![pad_token_id; 8]))?)?;
        inputs.encoder_hidden_states(encoder_hidden_states)?;

        let outputs = self.decoder_model.run(inputs.ort())?;
        let mut outputs = MusicGenOutputs::new(outputs);

        delay_pattern_mask_ids.push(
            outputs
                .take_logits()?
                .apply_free_guidance(GUIDANCE_SCALE)
                .sample(top_k)
                .iter()
                .map(|e| e.0),
        );

        for j in 0..num_hidden_layers {
            let v = outputs.take_present_decoder_key(j);
            inputs.past_key_value_decoder_key(j, v)?;
            let v = outputs.take_present_decoder_value(j);
            inputs.past_key_value_decoder_value(j, v)?;
            let v = outputs.take_present_encoder_key(j);
            inputs.past_key_value_encoder_key(j, v)?;
            let v = outputs.take_present_encoder_value(j);
            inputs.past_key_value_encoder_value(j, v)?;
        }

        inputs.remove_encoder_hidden_states();

        let decoder_with_past = self.decoder_with_past_model.clone();

        // TODO: 100?
        let (tx, rx) = std::sync::mpsc::channel::<ort::Result<[i64; 4]>>();
        let tx2 = tx.clone();
        std::thread::spawn(move || {
            let result = {
                for _ in 0..max_len {
                    if abort_controller.load(Ordering::Relaxed) {
                        return Err(ort::Error::CustomError("Aborted".into()));
                    }
                    let [a, b, c, d] = delay_pattern_mask_ids.last_delayed_masked(pad_token_id);
                    inputs.input_ids(ort::Tensor::from_array((
                        [8, 1],
                        vec![a, b, c, d, a, b, c, d],
                    ))?)?;
                    let outputs = decoder_with_past.run(inputs.ort())?;
                    let mut outputs = MusicGenOutputs::new(outputs);

                    delay_pattern_mask_ids.push(
                        outputs
                            .take_logits()?
                            .apply_free_guidance(GUIDANCE_SCALE)
                            .sample(top_k)
                            .iter()
                            .map(|e| e.0),
                    );

                    if let Some(last_de_delayed) = delay_pattern_mask_ids.last_de_delayed() {
                        let sent = tx.send(Ok(last_de_delayed));
                        if sent.is_err() {
                            break;
                        }
                    }

                    for j in 0..num_hidden_layers {
                        let v = outputs.take_present_decoder_key(j);
                        inputs.past_key_value_decoder_key(j, v)?;
                        let v = outputs.take_present_decoder_value(j);
                        inputs.past_key_value_decoder_value(j, v)?;
                        // NOTE: No need to propagate encoder values.
                        //
                        // let v = outputs.take_present_encoder_key(j);
                        // inputs.past_key_value_encoder_key(j, v)?;
                        // let v = outputs.take_present_encoder_value(j);
                        // inputs.past_key_value_encoder_value(j, v)?;
                    }
                }
                Ok::<(), ort::Error>(())
            };
            if let Err(err) = result {
                let _ = tx2.send(Err(err));
            }
            Ok(())
        });

        Ok(rx)
    }
}
