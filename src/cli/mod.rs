mod app_fs_ext;
mod download;
mod gpu;
mod loading_bar;
mod musicgen_builder;
mod musicgen_job_processor;

#[cfg(feature = "onnxruntime-from-source")]
mod onnxruntime_lib;

use anyhow::anyhow;
use clap::{Parser, ValueEnum};
use directories::ProjectDirs;
use lazy_static::lazy_static;
use regex::Regex;
use std::collections::VecDeque;
use std::fmt::{Display, Formatter};
use std::str::FromStr;
use text_io::read;
use tracing::warn;

use crate::audio::*;
use crate::backend::*;
use crate::storage::*;

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

    let mut device = "Cpu";
    if args.gpu {
        warn!("GPU support is experimental, it might not work on most platforms");
        let (gpu_device, provider) = gpu::init_gpu()?;
        device = gpu_device;
        ort_builder = ort_builder.with_execution_providers(&[provider]);
    }
    ort_builder.commit()?;

    let (text_encoder, decoder, audio_encodec) =
        musicgen_builder::musicgen_builder(args.model, args.use_split_decoder, args.force_download)
            .await?;

    if args.prompt.is_empty() {
        run(
            PROJECT_FS.clone(),
            musicgen_job_processor::MusicGenJobProcessor {
                name: args.model.to_string(),
                device: device.to_string(),
                text_encoder,
                decoder,
                audio_encodec,
            },
            RunOptions {
                port: args.ui_port,
                auto_open: true,
                expose: args.ui_expose,
            },
        )
        .await
    } else {
        let secs_re = Regex::new("--secs[ =](\\d+)")?;
        let output_re = Regex::new(r"--output[ =]([.a-zA-Z_-]+)")?;

        let audio_player = AudioManager::default();
        // This variable holds the audio stream. The stream stops when this is dropped,
        // so we need to maintain it referenced here.
        #[allow(unused_variables)]
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
            let token_stream =
                decoder.generate_tokens(last_hidden_state, attention_mask, max_len)?;
            let bar = loading_bar::LoadingBarFactory::bar("Generating audio");
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
                #[allow(unused_assignments)]
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
}