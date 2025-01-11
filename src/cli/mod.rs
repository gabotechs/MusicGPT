#[cfg(feature = "onnxruntime-from-source")]
mod onnxruntime_lib;

mod download;
mod gpu;
mod musicgen_models;
mod storage_ext;

use anyhow::anyhow;
use clap::{Parser, ValueEnum};
use directories::ProjectDirs;
use lazy_static::lazy_static;
use std::fmt::{Display, Formatter};
use tracing::warn;

use crate::backend::*;
use crate::storage::*;
use crate::terminal::{run_terminal_loop, RunTerminalOptions};

pub const INPUT_IDS_BATCH_PER_SECOND: usize = 50;

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

lazy_static! {
    static ref PROJECT_FS: AppFs = AppFs::new(
        ProjectDirs::from("com", "gabotechs", "musicgpt")
            .expect("Could not load project directory")
            .data_dir()
    );
}

impl Args {
    fn validate(&self) -> anyhow::Result<()> {
        if self.secs < 1 {
            return Err(anyhow!("--secs must > 0"));
        }
        if self.secs > 30 {
            return Err(anyhow!("--secs must <= 30"));
        }
        if self.no_interactive && self.prompt.is_empty() {
            return Err(anyhow!(
                "A prompt must be provided when not in interactive mode"
            ));
        }
        Ok(())
    }
}

pub async fn cli() -> anyhow::Result<()> {
    let args = Args::parse();
    args.validate()?;

    #[cfg(feature = "onnxruntime-from-source")]
    let mut ort_builder = ort::init_from(
        onnxruntime_lib::lookup_dynlib()
            .await?
            .to_str()
            .unwrap_or_default(),
    );
    #[cfg(not(feature = "onnxruntime-from-source"))]
    let mut ort_builder = ort::init();

    let device = if args.gpu {
        warn!("GPU support is experimental, it might not work on most platforms");
        let (gpu_device, provider) = gpu::init_gpu()?;
        ort_builder = ort_builder.with_execution_providers(&[provider]);
        gpu_device
    } else {
        "Cpu"
    };
    ort_builder.commit()?;

    let musicgen_models = musicgen_models::MusicGenModels::new(
        args.model,
        args.use_split_decoder,
        args.force_download,
    )
    .await?;

    if args.prompt.is_empty() {
        run_web_server(
            &PROJECT_FS.root,
            PROJECT_FS.clone(),
            musicgen_models,
            RunWebServerOptions {
                name: args.model.to_string(),
                device: device.to_string(),
                port: args.ui_port,
                auto_open: true,
                expose: args.ui_expose,
            },
        )
        .await
    } else {
        run_terminal_loop(
            &PROJECT_FS.root,
            PROJECT_FS.clone(),
            musicgen_models,
            RunTerminalOptions {
                init_prompt: args.prompt,
                init_secs: args.secs,
                init_output: args.output,
                no_playback: args.no_playback,
                no_interactive: args.no_interactive,
            },
        )
        .await
    }
}
