use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::anyhow;
use clap::{Parser, ValueEnum};
use half::f16;
use ort::{
    CoreMLExecutionProvider, CUDAExecutionProvider, ExecutionProvider, TensorRTExecutionProvider,
};
use tokenizers::Tokenizer;
use tokio::fs;

use crate::cli_interface::cli_interface;
use crate::loading_bar_factory::LoadingBarFactor;
use crate::music_gen_audio_encodec::MusicGenAudioEncodec;
use crate::music_gen_decoder::{MusicGenDecoder, MusicGenMergedDecoder, MusicGenSplitDecoder};
use crate::music_gen_text_encoder::MusicGenTextEncoder;
use crate::storage::Storage;
use crate::ui::App;

mod audio_manager;
mod cli_interface;
mod delay_pattern_mask_ids;
mod loading_bar_factory;
mod logits;
mod music_gen_audio_encodec;
mod music_gen_config;
mod music_gen_decoder;
mod music_gen_inputs;
mod music_gen_outputs;
mod music_gen_text_encoder;
mod storage;
mod tensor_ops;
mod ui;

#[derive(Clone, Copy, ValueEnum)]
enum Model {
    Small,
    SmallFp16,
    SmallQuant,
    Medium,
    MediumQuant,
    MediumFp16,
    Large,
}

#[derive(Parser)]
#[command(name = "MusicGPT")]
#[command(version, about, long_about = None)]
struct Args {
    /// The prompt for the LLM.
    #[arg(default_value = "")]
    prompt: String,

    /// The length of the audio will be generated in seconds.
    #[arg(long, default_value = "10")]
    secs: usize,

    /// The model to use. Some models are experimental, for example quantized models
    /// have a degraded quality and fp16 models are very slow.
    /// Beware of large models, you will need really powerful hardware for those.
    #[arg(long, default_value = "small")]
    model: Model,

    /// The LLM models are exported using https://github.com/huggingface/optimum,
    /// and they export transformer-based decoders either in two files, or a single
    /// merged one.
    #[arg(long, default_value = "false")]
    use_split_decoder: bool,

    /// Output path for the resulting .wav file
    #[arg(long, default_value = "musicgpt-generated.wav")]
    output: String,

    /// Force the download of LLM models
    #[arg(long, default_value = "false")]
    force_download: bool,

    /// Use the device's GPU for inference if available. GPU support is experimental.
    #[arg(long, default_value = "false")]
    gpu: bool,

    /// Do not play the audio automatically after inference.
    #[arg(long, default_value = "false")]
    no_playback: bool,

    /// Runs the interactive UI.
    #[arg(long, default_value = "false")]
    ui: bool,
}

impl Args {
    fn validate(&self) -> anyhow::Result<()> {
        if self.secs < 1 {
            return Err(anyhow!("--secs must > 0"));
        }
        if self.secs > 30 {
            return Err(anyhow!("--secs must <= 30"));
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    args.validate()?;

    if args.gpu {
        println!("WARNING: GPU support is experimental, it might not work on most platforms");
        init_gpu()?;
    }

    let (text_encoder, decoder, audio_encodec) = build_music_gen_parts(&args).await?;
    if args.ui {
        let app = App::try_from_music_gen(text_encoder, decoder, audio_encodec)?;
        app.run()
    } else {
        cli_interface(
            text_encoder,
            decoder,
            audio_encodec,
            args.secs,
            args.prompt,
            args.output,
            args.no_playback,
        )
        .await
    }
}

fn init_gpu() -> anyhow::Result<()> {
    let mut candidates = vec![];

    if cfg!(feature = "tensorrt") {
        candidates.push(TensorRTExecutionProvider::default().build());
    }
    if cfg!(feature = "cuda") {
        candidates.push(CUDAExecutionProvider::default().build());
    }
    if cfg!(feature = "coreml") {
        candidates.push(CoreMLExecutionProvider::default().with_ane_only().build());
    }

    let dummy_builder = ort::Session::builder()?;
    let mut providers = vec![];
    for provider in candidates {
        if let Err(err) = provider.register(&dummy_builder) {
            println!("Could not load {}: {}", provider.as_str(), err);
        } else {
            println!("{} detected", provider.as_str());
            providers.push(provider)
        }
    }

    if providers.is_empty() {
        return Err(anyhow!(
            "No hardware accelerator was detected, try running the program without the --gpu flag",
        ));
    }

    ort::init().with_execution_providers(providers).commit()?;
    Ok(())
}

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

async fn build_music_gen_parts(
    args: &Args,
) -> anyhow::Result<(
    MusicGenTextEncoder,
    Box<dyn MusicGenDecoder>,
    MusicGenAudioEncodec,
)> {
    let remote_file_spec = match (args.model, args.use_split_decoder) {
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

    let mut results = download(remote_file_spec, args.force_download).await?;

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

    let config = fs::read_to_string(config)
        .await
        .expect("Error reading config file from disk");
    let config = serde_json::from_str(&config).expect("Could not deserialize config file");
    #[allow(clippy::collapsible_else_if)]
    let decoder: Box<dyn MusicGenDecoder> = if args.use_split_decoder {
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
        if matches!(args.model, Model::SmallFp16 | Model::MediumFp16) {
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
        if matches!(args.model, Model::SmallFp16 | Model::MediumFp16) {
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

async fn download(
    remote_file_spec: Vec<(&'static str, &'static str)>,
    force_download: bool,
) -> anyhow::Result<VecDeque<PathBuf>> {
    let mut has_to_download = force_download;
    for (_, local_filename) in remote_file_spec.iter() {
        has_to_download = has_to_download || !Storage::exists(local_filename).await?
    }

    if has_to_download {
        println!("Some AI models need to be downloaded");
    }
    let m = LoadingBarFactor::multi();
    let mut tasks = vec![];
    for (remote_file, local_filename) in remote_file_spec {
        let bar = m.add(LoadingBarFactor::download_bar(local_filename));
        tasks.push(tokio::spawn(Storage::remote_data_file(
            remote_file,
            local_filename,
            force_download,
            bar.into_update_callback(),
        )));
    }
    let mut results = VecDeque::new();
    for task in tasks {
        results.push_back(task.await??);
    }
    m.clear()?;
    Ok(results)
}

async fn build_sessions(
    files: impl IntoIterator<Item = PathBuf>,
) -> anyhow::Result<VecDeque<ort::Session>> {
    let mut results = VecDeque::new();
    for file in files {
        if file.extension() != Some("onnx".as_ref()) {
            continue;
        }
        let bar = LoadingBarFactor::spinner(
            format!("Loading {:?}...", file.file_name().unwrap_or_default()).as_str(),
        );

        let result = ort::Session::builder()?.commit_from_file(file)?;
        bar.finish_and_clear();
        results.push_back(result);
    }
    Ok(results)
}
