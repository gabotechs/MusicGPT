use half::f16;
use ort::session::Session;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;

use crate::cli::download::download_many;
use crate::cli::loading_bar::LoadingBarFactory;
use crate::cli::Model;
use crate::musicgen::{
    MusicGenAudioEncodec, MusicGenDecoder, MusicGenMergedDecoder, MusicGenSplitDecoder,
    MusicGenTextEncoder,
};

pub async fn musicgen_builder(
    model: Model,
    use_split_decoder: bool,
    force_download: bool,
) -> anyhow::Result<(
    MusicGenTextEncoder,
    Box<dyn MusicGenDecoder>,
    MusicGenAudioEncodec,
)> {
    macro_rules! hf_url {
        ($t: expr) => {
            (
                concat!(
                    "https://huggingface.co/gabotechs/music_gen/resolve/main/",
                    $t
                ),
                concat!("v1/", $t,),
            )
        };
    }
    let remote_file_spec = match (model, use_split_decoder) {
        (Model::Small, true) => vec![
            hf_url!("small/config.json"),
            hf_url!("small/tokenizer.json"),
            hf_url!("small_fp32/text_encoder.onnx"),
            hf_url!("small_fp32/decoder_model.onnx"),
            hf_url!("small_fp32/decoder_with_past_model.onnx"),
            hf_url!("small_fp32/encodec_decode.onnx"),
        ],
        (Model::SmallQuant, true) => vec![
            hf_url!("small/config.json"),
            hf_url!("small/tokenizer.json"),
            hf_url!("small_fp32/text_encoder.onnx"),
            hf_url!("small_i8/decoder_model.onnx"),
            hf_url!("small_i8/decoder_with_past_model.onnx"),
            hf_url!("small_fp32/encodec_decode.onnx"),
        ],
        (Model::SmallFp16, true) => vec![
            hf_url!("small/config.json"),
            hf_url!("small/tokenizer.json"),
            hf_url!("small_fp16/text_encoder.onnx"),
            hf_url!("small_fp16/decoder_model.onnx"),
            hf_url!("small_fp16/decoder_with_past_model.onnx"),
            hf_url!("small_fp16/encodec_decode.onnx"),
        ],
        (Model::Medium, true) => vec![
            hf_url!("medium/config.json"),
            hf_url!("medium/tokenizer.json"),
            hf_url!("medium_fp32/text_encoder.onnx"),
            hf_url!("medium_fp32/decoder_model.onnx"),
            hf_url!("medium_fp32/decoder_with_past_model.onnx"),
            hf_url!("medium_fp32/encodec_decode.onnx"),
            // Files below will just be downloaded,
            hf_url!("medium_fp32/decoder_model.onnx_data"),
            hf_url!("medium_fp32/decoder_with_past_model.onnx_data"),
        ],
        (Model::MediumQuant, true) => vec![
            hf_url!("medium/config.json"),
            hf_url!("medium/tokenizer.json"),
            hf_url!("medium_fp32/text_encoder.onnx"),
            hf_url!("medium_i8/decoder_model.onnx"),
            hf_url!("medium_i8/decoder_with_past_model.onnx"),
            hf_url!("medium_fp32/encodec_decode.onnx"),
        ],
        (Model::MediumFp16, true) => vec![
            hf_url!("medium/config.json"),
            hf_url!("medium/tokenizer.json"),
            hf_url!("medium_fp16/text_encoder.onnx"),
            hf_url!("medium_fp16/decoder_model.onnx"),
            hf_url!("medium_fp16/decoder_with_past_model.onnx"),
            hf_url!("medium_fp16/encodec_decode.onnx"),
        ],
        (Model::Large, true) => vec![
            hf_url!("large/config.json"),
            hf_url!("large/tokenizer.json"),
            hf_url!("large_fp32/text_encoder.onnx"),
            hf_url!("large_fp32/decoder_model.onnx"),
            hf_url!("large_fp32/decoder_with_past_model.onnx"),
            hf_url!("large_fp32/encodec_decode.onnx"),
            // Files below will just be downloaded,
            hf_url!("large_fp32/decoder_model.onnx_data"),
            hf_url!("large_fp32/decoder_with_past_model.onnx_data"),
        ],
        (Model::Small, false) => vec![
            hf_url!("small/config.json"),
            hf_url!("small/tokenizer.json"),
            hf_url!("small_fp32/text_encoder.onnx"),
            hf_url!("small_fp32/decoder_model_merged.onnx"),
            hf_url!("small_fp32/encodec_decode.onnx"),
        ],
        (Model::SmallQuant, false) => vec![
            hf_url!("small/config.json"),
            hf_url!("small/tokenizer.json"),
            hf_url!("small_fp32/text_encoder.onnx"),
            hf_url!("small_i8/decoder_model_merged.onnx"),
            hf_url!("small_fp32/encodec_decode.onnx"),
        ],
        (Model::SmallFp16, false) => vec![
            hf_url!("small/config.json"),
            hf_url!("small/tokenizer.json"),
            hf_url!("small_fp16/text_encoder.onnx"),
            hf_url!("small_fp16/decoder_model_merged.onnx"),
            hf_url!("small_fp16/encodec_decode.onnx"),
        ],
        (Model::Medium, false) => vec![
            hf_url!("medium/config.json"),
            hf_url!("medium/tokenizer.json"),
            hf_url!("medium_fp32/text_encoder.onnx"),
            hf_url!("medium_fp32/decoder_model_merged.onnx"),
            hf_url!("medium_fp32/encodec_decode.onnx"),
            // Files below will just be downloaded,
            hf_url!("medium_fp32/decoder_model_merged.onnx_data"),
        ],
        (Model::MediumQuant, false) => vec![
            hf_url!("medium/config.json"),
            hf_url!("medium/tokenizer.json"),
            hf_url!("medium_fp32/text_encoder.onnx"),
            hf_url!("medium_i8/decoder_model_merged.onnx"),
            hf_url!("medium_fp32/encodec_decode.onnx"),
        ],
        (Model::MediumFp16, false) => vec![
            hf_url!("medium/config.json"),
            hf_url!("medium/tokenizer.json"),
            hf_url!("medium_fp16/text_encoder.onnx"),
            hf_url!("medium_fp16/decoder_model_merged.onnx"),
            hf_url!("medium_fp16/encodec_decode.onnx"),
            // Files below will just be downloaded,
            hf_url!("medium_fp16/decoder_model_merged.onnx_data"),
        ],
        (Model::Large, false) => vec![
            hf_url!("large/config.json"),
            hf_url!("large/tokenizer.json"),
            hf_url!("large_fp32/text_encoder.onnx"),
            hf_url!("large_fp32/decoder_model_merged.onnx"),
            hf_url!("large_fp32/encodec_decode.onnx"),
            // Files below will just be downloaded,
            hf_url!("large_fp32/decoder_model_merged.onnx_data"),
        ],
    };

    let mut results = download_many(
        remote_file_spec,
        force_download,
        "Some AI models need to be downloaded, this only needs to be done once",
        "AI models downloaded correctly",
    )
    .await?;

    // First result is the decoder config.
    let config = results.pop_front().unwrap();
    // Second result is the tokenizer.
    let tokenizer = results.pop_front().unwrap();
    let mut tokenizer = Tokenizer::from_file(tokenizer).expect("Could not load tokenizer");
    tokenizer
        .with_padding(None)
        .with_truncation(None)
        .expect("Could not configure tokenizer");

    let mut sessions = build_sessions(results).await?;

    let text_encoder = MusicGenTextEncoder {
        tokenizer,
        // third result is the text encoder.
        text_encoder: sessions.pop_front().unwrap(),
    };

    let config = tokio::fs::read_to_string(config)
        .await
        .expect("Error reading config file from disk");
    let config = serde_json::from_str(&config).expect("Could not deserialize config file");
    #[allow(clippy::collapsible_else_if)]
    let decoder: Box<dyn MusicGenDecoder> = if use_split_decoder {
        macro_rules! load {
            ($ty: ty) => {
                Box::new(MusicGenSplitDecoder::<$ty> {
                    // forth and fifth result are the decoder parts if split.
                    decoder_model: sessions.pop_front().unwrap(),
                    decoder_with_past_model: Arc::new(sessions.pop_front().unwrap()),
                    config,
                    _phantom_data: Default::default(),
                })
            };
        }
        if matches!(model, Model::SmallFp16 | Model::MediumFp16) {
            load!(f16)
        } else {
            load!(f32)
        }
    } else {
        macro_rules! load {
            ($ty: ty) => {
                Box::new(MusicGenMergedDecoder::<$ty> {
                    // forth result is the decoder.
                    decoder_model_merged: Arc::new(sessions.pop_front().unwrap()),
                    config,
                    _phantom_data: Default::default(),
                })
            };
        }
        if matches!(model, Model::SmallFp16 | Model::MediumFp16) {
            load!(f16)
        } else {
            load!(f32)
        }
    };
    let audio_encodec = MusicGenAudioEncodec {
        // last result is the audio encodec.
        audio_encodec_decode: sessions.pop_front().unwrap(),
    };

    Ok((text_encoder, decoder, audio_encodec))
}

async fn build_sessions(
    files: impl IntoIterator<Item = PathBuf>,
) -> anyhow::Result<VecDeque<Session>> {
    let mut results = VecDeque::new();
    for file in files {
        if file.extension() != Some("onnx".as_ref()) {
            continue;
        }
        let bar = LoadingBarFactory::spinner(
            format!("Loading {:?}...", file.file_name().unwrap_or_default()).as_str(),
        );

        let result = Session::builder()?.commit_from_file(file)?;
        bar.finish_and_clear();
        results.push_back(result);
    }
    Ok(results)
}