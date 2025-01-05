use std::collections::VecDeque;
use std::env::consts::ARCH;
use std::fmt::{Display, Formatter};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use crate::audio_manager::{AudioManager, AudioStream};
use crate::loading_bar_factory::LoadingBarFactor;
use crate::music_gen_audio_encodec::MusicGenAudioEncodec;
use crate::music_gen_decoder::{MusicGenDecoder, MusicGenMergedDecoder, MusicGenSplitDecoder};
use crate::music_gen_text_encoder::MusicGenTextEncoder;
use crate::storage::{AppFs, Storage};
use anyhow::anyhow;
use build_system::BuildInfo;
use clap::{Parser, ValueEnum};
use directories::ProjectDirs;
use half::f16;
use lazy_static::lazy_static;
use log::{error, info};
use ort::execution_providers::{
    CUDAExecutionProvider, CoreMLExecutionProvider, ExecutionProvider, ExecutionProviderDispatch,
    TensorRTExecutionProvider,
};
use ort::session::Session;
use regex::Regex;
use text_io::read;
use tokenizers::Tokenizer;
use tracing::warn;
use tracing_subscriber::fmt::time::UtcTime;
use tracing_subscriber::{fmt, EnvFilter};

mod audio_manager;
mod backend;
mod delay_pattern_mask_ids;
mod fetch_remove_data_file;
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

include!(concat!(env!("OUT_DIR"), "/built.rs"));

#[derive(Clone, Copy, ValueEnum)]
enum Model {
    Small,
    SmallFp16,
    SmallQuant,
    Medium,
    MediumFp16,
    MediumQuant,
    Large,
}

impl Display for Model {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Model::Small => write!(f, "MusicGen Small"),
            Model::SmallFp16 => write!(f, "MusicGen Small Fp16"),
            Model::SmallQuant => write!(f, "MusicGen Small Quantized"),
            Model::Medium => write!(f, "MusicGen Medium"),
            Model::MediumFp16 => write!(f, "MusicGen Medium Fp16"),
            Model::MediumQuant => write!(f, "MusicGen Medium Quantized"),
            Model::Large => write!(f, "MusicGen Large"),
        }
    }
}

#[derive(Parser)]
#[command(name = "MusicGPT")]
#[command(version, about, long_about = None)]
struct Args {
    /// The prompt for the LLM.
    /// If this argument is provided, MusicGPT will enter
    /// [CLI mode], where audio playback and prompting is managed through the terminal.
    /// If this argument is omitted, MusicGPT will enter
    /// [UI mode], where prompting and audio playback is managed through a web application.
    #[arg(default_value = "")]
    prompt: String,

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

    /// Force the download of LLM models.
    #[arg(long, default_value = "false")]
    force_download: bool,

    /// Use the device's GPU for inference if available. GPU support is experimental.
    #[arg(long, default_value = "false")]
    gpu: bool,

    /// [CLI mode] The seconds of audio to generate.
    #[arg(long, default_value = "10")]
    secs: usize,

    /// [CLI mode] Output path for the resulting .wav file.
    #[arg(long, default_value = "musicgpt-generated.wav")]
    output: String,

    /// [CLI mode] Do not play the audio automatically after inference.
    #[arg(long, default_value = "false")]
    no_playback: bool,

    /// [CLI mode] Disable interactive mode.
    #[arg(long, default_value = "false")]
    no_interactive: bool,

    /// [UI mode] Omits automatically opening the web app in a browser.
    #[arg(long, default_value = "false")]
    ui_no_open: bool,

    /// [UI mode] Port in which the MusicGPT web app will run.
    #[arg(long, default_value = "8642")]
    ui_port: usize,

    /// [UI mode] Exposes the MusicGPT web app in 0.0.0.0 instead of 127.0.0.1.
    #[arg(long, default_value = "false")]
    ui_expose: bool,
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

lazy_static! {
    static ref PROJECT_FS: AppFs = AppFs::new(
        ProjectDirs::from("com", "gabotechs", "musicgpt")
            .expect("Could not load project directory")
            .data_dir()
    );
}

async fn _main() -> anyhow::Result<()> {
    let args = Args::parse();
    args.validate()?;

    #[cfg(feature = "onnxruntime-from-source")]
    let mut ort_builder = ort::init_from(lookup_dyn_onnxruntime_lib().await?);
    #[cfg(not(feature = "onnxruntime-from-source"))]
    let mut ort_builder = ort::init();

    let mut device = "Cpu";
    if args.gpu {
        warn!("GPU support is experimental, it might not work on most platforms");
        let (gpu_device, provider) = init_gpu()?;
        device = gpu_device;
        ort_builder = ort_builder.with_execution_providers(&[provider]);
    }
    ort_builder.commit()?;

    if args.prompt.is_empty() {
        let (text_encoder, decoder, audio_encodec) = build_music_gen_parts(&args).await?;
        backend::run(
            PROJECT_FS.clone(),
            backend::MusicGenJobProcessor {
                name: args.model.to_string(),
                device: device.to_string(),
                text_encoder,
                decoder,
                audio_encodec,
            },
            backend::RunOptions {
                port: args.ui_port,
                auto_open: true,
                expose: args.ui_expose,
            },
        )
        .await
    } else {
        cli_interface(&args).await
    }
}

#[tokio::main]
async fn main() {
    let time_format = time::format_description::parse(
        "[year]-[month]-[day] [hour]:[minute]:[second].[subsecond digits:3]",
    )
    .expect("Failed to create timestamp format");
    let format = fmt::format()
        .with_target(false)
        .with_timer(UtcTime::new(time_format));
    let filter = EnvFilter::new("info,ort=off");

    tracing_subscriber::fmt()
        .event_format(format)
        .with_max_level(tracing::Level::INFO)
        .with_env_filter(filter)
        .init();
    if let Err(err) = _main().await {
        error!("{err}")
    }
}

const INPUT_IDS_BATCH_PER_SECOND: usize = 50;

#[allow(unused_assignments, unused_variables)]
async fn cli_interface(args: &Args) -> anyhow::Result<()> {
    let (text_encoder, decoder, audio_encodec) = build_music_gen_parts(args).await?;
    let secs_re = Regex::new("--secs[ =](\\d+)")?;
    let output_re = Regex::new(r"--output[ =]([.a-zA-Z_-]+)")?;

    let audio_player = AudioManager::default();
    // This variable holds the audio stream. The stream stops when this is dropped,
    // so we need to maintain it referenced here.
    let mut curr_stream: Option<AudioStream> = None;
    let mut prompt = args.prompt.clone();
    let mut secs = args.secs;
    let mut output = args.output.clone();

    loop {
        if prompt.is_empty() {
            if args.no_interactive {
                return Err(anyhow!(
                    "A prompt must be provided when not in interactive mode"
                ));
            }
            print!(">>> ");
            prompt = read!("{}\n");
            if prompt == "exit" {
                return Ok(());
            }
            if let Some(captures) = secs_re.captures(&prompt) {
                if let Some(capture) = captures.get(1) {
                    if let Ok(s) = usize::from_str(capture.as_str()) {
                        secs = s;
                    }
                }
            }
            if let Some(captures) = output_re.captures(&prompt) {
                if let Some(capture) = captures.get(1) {
                    if !capture.is_empty() {
                        output = capture.as_str().to_string()
                    }
                }
            }
        }
        // First, encode the text.
        let (last_hidden_state, attention_mask) = text_encoder.encode(&prompt)?;

        // Second, generate tokens.
        let max_len = secs * INPUT_IDS_BATCH_PER_SECOND;
        let token_stream = decoder.generate_tokens(last_hidden_state, attention_mask, max_len)?;
        let bar = LoadingBarFactor::bar("Generating audio");
        let mut data = VecDeque::new();
        while let Ok(tokens) = token_stream.recv() {
            data.push_back(tokens?);
            bar.update_elapsed_total(data.len(), max_len)
        }

        // Third, encode the tokens into audio.
        let samples = audio_encodec.encode(data)?;

        // Last, play the audio.
        if !args.no_playback {
            let samples_copy = samples.clone();
            let stream = audio_player.play_from_queue(samples_copy);
            if let Ok(stream) = stream {
                curr_stream = Some(stream);
            }
        }
        if !output.ends_with(".wav") {
            output += ".wav";
        }
        let bytes = audio_player.to_wav(samples)?;
        tokio::fs::write(&output, bytes).await?;

        prompt = "".into();
        if args.no_interactive {
            break;
        }
    }

    Ok(())
}

#[cfg(feature = "onnxruntime-from-source")]
async fn lookup_dyn_onnxruntime_lib() -> anyhow::Result<String> {
    // Build info dumped by the build-system crate at compile time.
    let BuildInfo {
        local_dynlib_filepaths,
        main_dynlib_filename,
        dynlib_filenames,
        version,
    } = BuildInfo::from_out_dir(env!("OUT_DIR"));

    // If running with Cargo, build.rs have set this ONNXRUNTIME_LOCAL_FILES env to the
    // path of the generated dynamic library files compiled from source.
    // If not running with cargo, this will not be set.
    for local_filepath in local_dynlib_filepaths {
        if local_filepath.ends_with(&main_dynlib_filename)
            && tokio::fs::try_exists(&local_filepath)
                .await
                .unwrap_or(false)
        {
            let local_filepath = local_filepath.to_str().expect("local filepath is not UTF8");
            return Ok(local_filepath.to_string());
        }
    }

    // If there's no local file, attempt to download it from a GitHub release.
    let remote_file_spec = dynlib_filenames
        .iter()
        .map(|v| {
            (
                format!("{PKG_REPOSITORY}/releases/download/v{PKG_VERSION}/{TARGET}-{v}"),
                format!("dynlibs/{version}/{v}"),
            )
        })
        .collect::<Vec<_>>();
    download(
        remote_file_spec,
        false,
        &format!("Dynamic libraries not found in path set by ONNXRUNTIME_LOCAL_FILES env variable. Downloading them from GitHub release {PKG_VERSION}..."),
        "Dynamic libraries downloaded",
    )
    .await?;
    Ok(format!("dynlibs/{version}/{main_dynlib_filename}"))
}

fn init_gpu() -> anyhow::Result<(&'static str, ExecutionProviderDispatch)> {
    let mut dummy_builder = Session::builder()?;

    if cfg!(feature = "tensorrt") {
        let provider = TensorRTExecutionProvider::default();
        match provider.register(&mut dummy_builder) {
            Ok(_) => {
                info!("{} detected", provider.as_str());
                return Ok(("TensorRT", provider.build()));
            }
            Err(err) => error!("Could not load {}: {}", provider.as_str(), err),
        }
    }
    if cfg!(feature = "cuda") {
        let provider = CUDAExecutionProvider::default();
        match provider.register(&mut dummy_builder) {
            Ok(_) => {
                info!("{} detected", provider.as_str());
                return Ok(("Cuda", provider.build()));
            }
            Err(err) => error!("Could not load {}: {}", provider.as_str(), err),
        }
    }
    if cfg!(feature = "coreml") {
        let provider = CoreMLExecutionProvider::default().with_ane_only();
        match provider.register(&mut dummy_builder) {
            Ok(_) => {
                info!("{} detected", provider.as_str());
                return Ok(("CoreML", provider.build()));
            }
            Err(err) => error!("Could not load {}: {}", provider.as_str(), err),
        }
    }

    Err(anyhow!(
        "No hardware accelerator was detected, try running the program without the --gpu flag",
    ))
}

async fn build_music_gen_parts(
    args: &Args,
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

    let mut results = download(
        remote_file_spec,
        args.force_download,
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

async fn download<T: Display>(
    remote_file_spec: Vec<(T, T)>,
    force_download: bool,
    on_download_msg: &str,
    on_finished_msg: &str,
) -> anyhow::Result<VecDeque<PathBuf>> {
    let mut has_to_download = force_download;
    for (_, local_filename) in remote_file_spec.iter() {
        has_to_download = has_to_download || !PROJECT_FS.exists(&local_filename.to_string()).await?
    }

    if has_to_download {
        info!("{on_download_msg}");
    }
    let m = LoadingBarFactor::multi();
    let mut tasks = vec![];
    for (remote_file, local_filename) in remote_file_spec {
        let remote_file = remote_file.to_string();
        let local_filename = local_filename.to_string();
        let bar = m.add(LoadingBarFactor::download_bar(&local_filename));
        tasks.push(tokio::spawn(async move {
            PROJECT_FS
                .fetch_remote_data_file(
                    &remote_file,
                    &local_filename,
                    force_download,
                    bar.into_update_callback(),
                )
                .await
        }));
    }
    let mut results = VecDeque::new();
    for task in tasks {
        results.push_back(task.await??);
    }
    m.clear()?;
    if has_to_download {
        info!("{on_finished_msg}");
    }
    Ok(results)
}

async fn build_sessions(
    files: impl IntoIterator<Item = PathBuf>,
) -> anyhow::Result<VecDeque<Session>> {
    let mut results = VecDeque::new();
    for file in files {
        if file.extension() != Some("onnx".as_ref()) {
            continue;
        }
        let bar = LoadingBarFactor::spinner(
            format!("Loading {:?}...", file.file_name().unwrap_or_default()).as_str(),
        );

        let result = Session::builder()?.commit_from_file(file)?;
        bar.finish_and_clear();
        results.push_back(result);
    }
    Ok(results)
}
