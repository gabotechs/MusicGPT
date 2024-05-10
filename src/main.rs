use std::collections::VecDeque;
use std::error::Error;
use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use half::f16;
use ort::{
    CUDAExecutionProvider, CoreMLExecutionProvider, ExecutionProvider, TensorRTExecutionProvider,
};
use tokio::sync::mpsc::Receiver;

use crate::audio_manager::AudioManager;
use crate::loading_bar_factory::LoadingBarFactor;
use crate::music_gen::{MusicGen, MusicGenMergedLoadOptions, MusicGenSplitLoadOptions};
use crate::music_gen_decoder::{MusicGenMergedDecoder, MusicGenSplitDecoder, MusicGenType};
use crate::storage::Storage;

mod audio_manager;
mod delay_pattern_mask_ids;
mod loading_bar_factory;
mod logits;
mod music_gen;
mod music_gen_audio_encodec;
mod music_gen_config;
mod music_gen_decoder;
mod music_gen_inputs;
mod music_gen_outputs;
mod music_gen_text_encoder;
mod storage;
mod tensor_ops;

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
}

fn io_err(msg: &str) -> Result<(), Box<dyn Error>> {
    Err(Box::new(std::io::Error::new(
        std::io::ErrorKind::Other,
        msg,
    )))
}

impl Args {
    fn validate(&self) -> Result<(), Box<dyn Error>> {
        if self.secs < 1 {
            return io_err("--secs must > 0");
        }
        if self.secs > 30 {
            return io_err("--secs must <= 30");
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    args.validate()?;

    if args.gpu {
        println!("WARNING: GPU support is experimental, it might not work on most platforms");
        init_gpu()?;
    }

    async fn split_stream<T: MusicGenType + 'static>(
        opts: MusicGenSplitLoadOptions,
        args: &Args,
    ) -> ort::Result<Receiver<ort::Result<f32>>> {
        let spinner = LoadingBarFactor::spinner("Loading models");
        let music_gen = MusicGen::<MusicGenSplitDecoder<T>>::load(opts).await?;
        spinner.finish_and_clear();
        let bar = LoadingBarFactor::bar("Generating audio");
        music_gen
            .generate(&args.prompt, args.secs, |el, t| {
                bar.update_elapsed_total(el, t)
            })
            .await
    }

    async fn merged_stream<T: MusicGenType + 'static>(
        opts: MusicGenMergedLoadOptions,
        args: &Args,
    ) -> ort::Result<Receiver<ort::Result<f32>>> {
        let spinner = LoadingBarFactor::spinner("Loading models");
        let music_gen = MusicGen::<MusicGenMergedDecoder<T>>::load(opts).await?;
        spinner.finish_and_clear();
        let bar = LoadingBarFactor::bar("Generating audio");
        music_gen
            .generate(&args.prompt, args.secs, |el, t| {
                bar.update_elapsed_total(el, t)
            })
            .await
    }

    let mut sample_stream = match (args.model, args.use_split_decoder) {
        (Model::Small, false) => {
            merged_stream::<f32>(model_to_music_gen_merged_load_opts(&args).await?, &args).await
        }
        (Model::Small, true) => {
            split_stream::<f32>(model_to_music_gen_split_load_opts(&args).await?, &args).await
        }
        (Model::SmallQuant, false) => {
            merged_stream::<f32>(model_to_music_gen_merged_load_opts(&args).await?, &args).await
        }
        (Model::SmallQuant, true) => {
            split_stream::<f32>(model_to_music_gen_split_load_opts(&args).await?, &args).await
        }
        (Model::SmallFp16, false) => {
            merged_stream::<f16>(model_to_music_gen_merged_load_opts(&args).await?, &args).await
        }
        (Model::SmallFp16, true) => {
            split_stream::<f16>(model_to_music_gen_split_load_opts(&args).await?, &args).await
        }
        (Model::Medium, false) => {
            merged_stream::<f32>(model_to_music_gen_merged_load_opts(&args).await?, &args).await
        }
        (Model::Medium, true) => {
            split_stream::<f32>(model_to_music_gen_split_load_opts(&args).await?, &args).await
        }
        (Model::MediumQuant, false) => {
            merged_stream::<f32>(model_to_music_gen_merged_load_opts(&args).await?, &args).await
        }
        (Model::MediumQuant, true) => {
            split_stream::<f32>(model_to_music_gen_split_load_opts(&args).await?, &args).await
        }
        (Model::MediumFp16, false) => {
            merged_stream::<f16>(model_to_music_gen_merged_load_opts(&args).await?, &args).await
        }
        (Model::MediumFp16, true) => {
            split_stream::<f16>(model_to_music_gen_split_load_opts(&args).await?, &args).await
        }
        (Model::Large, false) => {
            merged_stream::<f32>(model_to_music_gen_merged_load_opts(&args).await?, &args).await
        }
        (Model::Large, true) => {
            split_stream::<f32>(model_to_music_gen_split_load_opts(&args).await?, &args).await
        }
    }?;

    let output = if !args.output.ends_with(".wav") {
        args.output + ".wav"
    } else {
        args.output
    };

    let mut data = VecDeque::new();
    while let Some(sample) = sample_stream.recv().await {
        data.push_back(sample?);
    }

    let audio_player = AudioManager::default();
    if args.no_playback {
        audio_player.store_as_wav(data, output).await?;
    } else {
        let spinner = LoadingBarFactor::spinner("Playing audio...");
        let (_play_from_queue, store_as_wav) = tokio::join!(
            audio_player.play_from_queue(data.clone()),
            audio_player.store_as_wav(data, output)
        );
        // audio playback is just a best effort operation, we don't care if it fails.
        store_as_wav?;
        spinner.finish_and_clear();
    }

    Ok(())
}

fn init_gpu() -> Result<(), Box<dyn Error>> {
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
    for provider in candidates.iter() {
        if let Err(err) = provider.register(&dummy_builder) {
            println!("Could not load {}: {}", provider.as_str(), err);
        } else {
            println!("{} detected", provider.as_str());
            providers.push(provider)
        }
    }

    if providers.is_empty() {
        return io_err(
            "No hardware accelerator was detected, try running the program without the --gpu flag",
        );
    }

    ort::init().with_execution_providers(candidates).commit()?;
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

fn model_not_available<T>() -> Result<T, Box<dyn Error>> {
    Err(Box::new(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "Model not available",
    )))
}

async fn model_to_music_gen_merged_load_opts(
    args: &Args,
) -> Result<MusicGenMergedLoadOptions, Box<dyn Error>> {
    // Note that here some destination paths are exactly the same no matter the model.
    // This is because files are exactly the same, and if someone tried model "small",
    // we do not want to force them to re-download repeated files. The following files
    // are the same independently of the model:
    // - tokenizer.json
    // - text_encoder.onnx
    // - encodec_decode.onnx
    // That's why they are not prefixed by the model size. Files prefix by the model size
    // do vary depending on the model.
    let remote_file_spec = match args.model {
        Model::Small => vec![
            hf_url!("small/config.json"),
            hf_url!("small/tokenizer.json"),
            hf_url!("small_fp32/text_encoder.onnx"),
            hf_url!("small_fp32/decoder_model_merged.onnx"),
            hf_url!("small_fp32/encodec_decode.onnx"),
        ],
        Model::SmallQuant => vec![
            hf_url!("small/config.json"),
            hf_url!("small/tokenizer.json"),
            hf_url!("small_fp32/text_encoder.onnx"),
            hf_url!("small_i8/decoder_model_merged.onnx"),
            hf_url!("small_fp32/encodec_decode.onnx"),
        ],
        Model::SmallFp16 => vec![
            hf_url!("small/config.json"),
            hf_url!("small/tokenizer.json"),
            hf_url!("small_fp16/text_encoder.onnx"),
            hf_url!("small_fp16/decoder_model_merged.onnx"),
            hf_url!("small_fp16/encodec_decode.onnx"),
        ],
        Model::Medium => vec![
            hf_url!("medium/config.json"),
            hf_url!("medium/tokenizer.json"),
            hf_url!("medium_fp32/text_encoder.onnx"),
            hf_url!("medium_fp32/decoder_model_merged.onnx"),
            hf_url!("medium_fp32/encodec_decode.onnx"),
            // Files below will just be downloaded,
            hf_url!("medium_fp32/decoder_model_merged.onnx_data"),
        ],
        Model::MediumQuant => vec![
            hf_url!("medium/config.json"),
            hf_url!("medium/tokenizer.json"),
            hf_url!("medium_fp32/text_encoder.onnx"),
            hf_url!("medium_fpi8/decoder_model_merged.onnx"),
            hf_url!("medium_fpi8/encodec_decode.onnx"),
        ],
        Model::MediumFp16 => vec![
            hf_url!("medium/config.json"),
            hf_url!("medium/tokenizer.json"),
            hf_url!("medium_fp16/text_encoder.onnx"),
            hf_url!("medium_fp16/decoder_model_merged.onnx"),
            hf_url!("medium_fp16/encodec_decode.onnx"),
            // Files below will just be downloaded,
            hf_url!("medium_fp16/decoder_model_merged.onnx_data"),
        ],
        Model::Large => vec![
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

    let config = results.pop_front().unwrap();
    let tokenizer = results.pop_front().unwrap();
    let mut sessions = build_sessions(results).await?;
    let text_encoder = sessions.pop_front().unwrap();
    let decoder_model_merged = sessions.pop_front().unwrap();
    let audio_encodec_decode = sessions.pop_front().unwrap();

    Ok(MusicGenMergedLoadOptions {
        config,
        tokenizer,
        text_encoder,
        decoder_model_merged,
        audio_encodec_decode,
    })
}

async fn model_to_music_gen_split_load_opts(
    args: &Args,
) -> Result<MusicGenSplitLoadOptions, Box<dyn Error>> {
    let remote_file_spec = match args.model {
        Model::Small => vec![
            hf_url!("small/config.json"),
            hf_url!("small/tokenizer.json"),
            hf_url!("small_fp32/text_encoder.onnx"),
            hf_url!("small_fp32/decoder_model.onnx"),
            hf_url!("small_fp32/decoder_with_past_model.onnx"),
            hf_url!("small_fp32/encodec_decode.onnx"),
        ],
        Model::SmallQuant => vec![
            hf_url!("small/config.json"),
            hf_url!("small/tokenizer.json"),
            hf_url!("small_fp32/text_encoder.onnx"),
            hf_url!("small_i8/decoder_model.onnx"),
            hf_url!("small_i8/decoder_with_past_model.onnx"),
            hf_url!("small_fp32/encodec_decode.onnx"),
        ],
        Model::SmallFp16 => vec![
            hf_url!("small/config.json"),
            hf_url!("small/tokenizer.json"),
            hf_url!("small_fp16/text_encoder.onnx"),
            hf_url!("small_fp16/decoder_model.onnx"),
            hf_url!("small_fp16/decoder_with_past_model.onnx"),
            hf_url!("small_fp16/encodec_decode.onnx"),
        ],
        Model::Medium => vec![
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
        Model::MediumQuant => vec![
            hf_url!("medium/config.json"),
            hf_url!("medium/tokenizer.json"),
            hf_url!("medium_fp32/text_encoder.onnx"),
            hf_url!("medium_i8/decoder_model.onnx"),
            hf_url!("medium_i8/decoder_with_past_model.onnx"),
            hf_url!("medium_fp32/encodec_decode.onnx"),
        ],
        Model::MediumFp16 => vec![
            hf_url!("medium/config.json"),
            hf_url!("medium/tokenizer.json"),
            hf_url!("medium_fp16/text_encoder.onnx"),
            hf_url!("medium_fp16/decoder_model.onnx"),
            hf_url!("medium_fp16/decoder_with_past_model.onnx"),
            hf_url!("medium_fp16/encodec_decode.onnx"),
        ],
        Model::Large => vec![
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
    };

    let mut results = download(remote_file_spec, args.force_download).await?;

    let config = results.pop_front().unwrap();
    let tokenizer = results.pop_front().unwrap();
    let mut sessions = build_sessions(results).await?;
    let text_encoder = sessions.pop_front().unwrap();
    let decoder_model = sessions.pop_front().unwrap();
    let decoder_with_past_model = sessions.pop_front().unwrap();
    let audio_encodec_decode = sessions.pop_front().unwrap();

    Ok(MusicGenSplitLoadOptions {
        config,
        tokenizer,
        text_encoder,
        decoder_model,
        decoder_with_past_model,
        audio_encodec_decode,
    })
}

async fn download(
    remote_file_spec: Vec<(&'static str, &'static str)>,
    force_download: bool,
) -> Result<VecDeque<PathBuf>, Box<dyn Error>> {
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
            move |el, t| bar.update_elapsed_total(el, t),
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
) -> Result<VecDeque<ort::Session>, Box<dyn Error>> {
    let mut results = VecDeque::new();
    for file in files {
        let bar = LoadingBarFactor::spinner(
            format!("Loading {:?}...", file.file_name().unwrap_or_default()).as_str(),
        );

        let result = ort::Session::builder()?.commit_from_file(file)?;
        bar.finish_and_clear();
        results.push_back(result);
    }
    Ok(results)
}
