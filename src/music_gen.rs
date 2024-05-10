use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;

use tokenizers::Tokenizer;
use tokio::fs;

use crate::music_gen_audio_encodec::MusicGenAudioEncodec;
use crate::music_gen_decoder::{
    MusicGenDecoder, MusicGenMergedDecoder, MusicGenSplitDecoder, MusicGenType,
};
use crate::music_gen_text_encoder::MusicGenTextEncoder;

// I calculated this myself.
pub const INPUT_IDS_BATCH_PER_SECOND: usize = 50;

pub struct MusicGen<D: MusicGenDecoder> {
    text_encoder: MusicGenTextEncoder,
    audio_encodec: MusicGenAudioEncodec,
    decoder: D,
}

pub struct MusicGenMergedLoadOptions {
    pub tokenizer: PathBuf,
    pub text_encoder: ort::Session,
    pub decoder_model_merged: ort::Session,
    pub audio_encodec_decode: ort::Session,
    pub config: PathBuf,
}

impl<T: MusicGenType + 'static> MusicGen<MusicGenMergedDecoder<T>> {
    pub async fn load(opts: MusicGenMergedLoadOptions) -> ort::Result<Self> {
        let mut tokenizer = Tokenizer::from_file(opts.tokenizer)?;
        tokenizer.with_padding(None).with_truncation(None)?;

        let config = fs::read_to_string(opts.config)
            .await
            .expect("Error reading config file from disk");
        let config = serde_json::from_str(&config).expect("Could not deserialize config file");

        Ok(Self {
            text_encoder: MusicGenTextEncoder {
                tokenizer,
                text_encoder: opts.text_encoder,
            },
            decoder: MusicGenMergedDecoder::<T> {
                decoder_model_merged: Arc::new(opts.decoder_model_merged),
                config,
                _phantom_data: PhantomData,
            },
            audio_encodec: MusicGenAudioEncodec {
                audio_encodec_decode: opts.audio_encodec_decode,
            },
        })
    }
}

pub struct MusicGenSplitLoadOptions {
    pub tokenizer: PathBuf,
    pub text_encoder: ort::Session,
    pub decoder_model: ort::Session,
    pub decoder_with_past_model: ort::Session,
    pub audio_encodec_decode: ort::Session,
    pub config: PathBuf,
}

impl<T: MusicGenType + 'static> MusicGen<MusicGenSplitDecoder<T>> {
    pub async fn load(opts: MusicGenSplitLoadOptions) -> ort::Result<Self> {
        let mut tokenizer = Tokenizer::from_file(opts.tokenizer)?;
        tokenizer.with_padding(None).with_truncation(None)?;

        let config = fs::read_to_string(opts.config)
            .await
            .expect("Error reading config file from disk");
        let config = serde_json::from_str(&config).expect("Could not deserialize config file");

        Ok(Self {
            text_encoder: MusicGenTextEncoder {
                tokenizer,
                text_encoder: opts.text_encoder,
            },
            decoder: MusicGenSplitDecoder::<T> {
                decoder_model: opts.decoder_model,
                decoder_with_past_model: Arc::new(opts.decoder_with_past_model),
                config,
                _phantom_data: PhantomData,
            },
            audio_encodec: MusicGenAudioEncodec {
                audio_encodec_decode: opts.audio_encodec_decode,
            },
        })
    }
}

impl<D: MusicGenDecoder> MusicGen<D> {
    pub async fn generate<Cb: Fn(usize, usize)>(
        &self,
        text: &str,
        secs: usize,
        cb: Cb,
    ) -> ort::Result<tokio::sync::mpsc::Receiver<ort::Result<f32>>> {
        let max_len = secs * INPUT_IDS_BATCH_PER_SECOND;
        let (last_hidden_state, attention_mask) = self.text_encoder.encode(text)?;
        let generator = self
            .decoder
            .generate_tokens(last_hidden_state, attention_mask, max_len)?;
        self.audio_encodec.encode(generator, max_len, cb).await
    }
}
