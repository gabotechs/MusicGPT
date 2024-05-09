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

async fn build_session(path: PathBuf) -> ort::Result<ort::Session> {
    let file = fs::read(&path)
        .await
        .map_err(|_| ort::Error::FileDoesNotExist { filename: path })?;
    ort::Session::builder()?
        .with_intra_threads(4)?
        .commit_from_memory(file.as_slice())
}

pub struct MusicGen<D: MusicGenDecoder> {
    text_encoder: MusicGenTextEncoder,
    audio_encodec: MusicGenAudioEncodec,
    decoder: D,
}

pub struct MusicGenMergedLoadOptions {
    pub tokenizer: PathBuf,
    pub text_encoder: PathBuf,
    pub decoder_model_merged: PathBuf,
    pub audio_encodec_decode: PathBuf,
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
            decoder: MusicGenMergedDecoder::<T> {
                decoder_model_merged: Arc::new(result.1?),
                config,
                _phantom_data: PhantomData,
            },
            audio_encodec: MusicGenAudioEncodec {
                audio_encodec_decode: result.2?,
            },
        })
    }
}

pub struct MusicGenSplitLoadOptions {
    pub tokenizer: PathBuf,
    pub text_encoder: PathBuf,
    pub decoder_model: PathBuf,
    pub decoder_with_past_model: PathBuf,
    pub audio_encodec_decode: PathBuf,
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

        let result = tokio::join!(
            build_session(opts.text_encoder),
            build_session(opts.decoder_model),
            build_session(opts.decoder_with_past_model),
            build_session(opts.audio_encodec_decode)
        );

        Ok(Self {
            text_encoder: MusicGenTextEncoder {
                tokenizer,
                text_encoder: result.0?,
            },
            decoder: MusicGenSplitDecoder::<T> {
                decoder_model: result.1?,
                decoder_with_past_model: Arc::new(result.2?),
                config,
                _phantom_data: PhantomData,
            },
            audio_encodec: MusicGenAudioEncodec {
                audio_encodec_decode: result.3?,
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
