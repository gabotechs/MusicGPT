use std::collections::VecDeque;
use std::error::Error;

use clap::{Parser, ValueEnum};
use ort::{
    CUDAExecutionProvider, CoreMLExecutionProvider, DirectMLExecutionProvider,
    TensorRTExecutionProvider,
};
use tokio::sync::mpsc::Receiver;

use crate::audio_manager::AudioManager;
use crate::loading_bar_factory::LoadingBarFactor;
use crate::music_gen::{MusicGen, MusicGenLoadOptions, MusicGenType};
use crate::storage::Storage;

mod audio_manager;
mod delay_pattern_mask_ids;
mod loading_bar_factory;
mod logits;
mod music_gen;
mod music_gen_config;
mod music_gen_inputs;
mod music_gen_outputs;
mod storage;
mod music_gen_text_encoder;
mod music_gen_audio_encodec;
mod tensor_ops;

#[derive(Clone, ValueEnum)]
enum Model {
    Small,
    SmallQuant,
    Medium,
    Large,
}

#[derive(Parser)]
struct Args {
    prompt: String,

    #[arg(long, default_value = "10")]
    secs: usize,

    #[arg(long, default_value = "small-quant")]
    model: Model,

    #[arg(long, default_value = "")]
    output: String,

    #[arg(long, default_value = "false")]
    force_download: bool,

    #[arg(long, default_value = "false")]
    gpu: bool,

    #[arg(long, default_value = "false")]
    no_playback: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    if args.gpu {
        println!("WARNING: GPU support is experimental, it might not work on most platforms");
        ort::init()
            .with_execution_providers([
                // Prefer TensorRT over CUDA.
                TensorRTExecutionProvider::default().build(),
                CUDAExecutionProvider::default().build(),
                // Use DirectML on Windows if NVIDIA EPs are not available
                DirectMLExecutionProvider::default().build(),
                // Or use ANE on Apple platforms
                CoreMLExecutionProvider::default().build(),
            ])
            .commit()?;
    } else {
        ort::init().commit()?;
    };

    async fn gen_stream<T: MusicGenType + 'static>(
        opts: MusicGenLoadOptions,
        args: &Args,
    ) -> ort::Result<Receiver<ort::Result<f32>>> {
        let spinner = LoadingBarFactor::spinner("Loading models");
        let music_gen = MusicGen::<T>::load(opts).await?;
        spinner.finish_and_clear();
        let bar = LoadingBarFactor::bar("Generating audio");
        music_gen
            .generate(&args.prompt, args.secs, |el, t| {
                bar.update_elapsed_total(el, t)
            })
            .await
    }

    let opts = model_to_music_gen_load_opts(&args).await?;
    let mut sample_stream = match args.model {
        Model::Small => gen_stream::<f32>(opts, &args),
        Model::Medium => gen_stream::<f32>(opts, &args),
        Model::Large => gen_stream::<f32>(opts, &args),
        // Surprisingly, the quantized model also used f32 for inputs and outputs.
        Model::SmallQuant => gen_stream::<f32>(opts, &args),
    }
    .await?;

    let output = if args.output.is_empty() {
        "musicgpt-generated.wav".to_string()
    } else if !args.output.ends_with(".wav") {
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

// These constants reflect a 1:1 what's present in the https://huggingface.co/gabotechs/music_gen
// repo. Even if they have different URLs, a lot of the files turn out to be the same, like the
// tokenizers, de encodec_decode.onnx model and the text_encoder.onnx model.
#[rustfmt::skip]
mod urls {
    pub const CONFIG_SMALL: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_small/config.json";
    pub const TOKENIZER_JSON_SMALL: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_small/tokenizer.json";
    pub const TEXT_ENCODER_SMALL: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_small/text_encoder.onnx";
    pub const DECODER_SMALL: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_small/decoder_model_merged.onnx";
    pub const DECODER_SMALL_QUANTIZED: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_small/decoder_model_merged_quantized.onnx";
    pub const ENCODEC_DECODE_SMALL: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_small/encodec_decode.onnx";

    pub const CONFIG_MEDIUM: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_medium/config.json";
    pub const TOKENIZER_JSON_MEDIUM: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_medium/tokenizer.json";
    pub const TEXT_ENCODER_MEDIUM: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_medium/text_encoder.onnx";
    pub const DECODER_MEDIUM: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_medium/decoder_model_merged.onnx";
    pub const DECODER_DATA_MEDIUM: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_medium/decoder_model_merged.onnx_data";
    pub const ENCODEC_DECODE_MEDIUM: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_medium/encodec_decode.onnx";

    pub const CONFIG_LARGE: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_large/config.json";
    pub const TOKENIZER_JSON_LARGE: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_large/tokenizer.json";
    pub const TEXT_ENCODER_LARGE: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_large/text_encoder.onnx";
    pub const DECODER_LARGE: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_large/decoder_model_merged.onnx";
    pub const DECODER_DATA_LARGE: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_large/decoder_model_merged.onnx_data";
    pub const ENCODEC_DECODE_LARGE: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_large/encodec_decode.onnx";
}

async fn model_to_music_gen_load_opts(args: &Args) -> Result<MusicGenLoadOptions, Box<dyn Error>> {
    // Note that here some destination paths are exactly the same no matter the model.
    // This is because files are exactly the same, and if someone tried model "small",
    // we do not want to force them to re-download repeated files. The following files
    // are the same independently of the model:
    // - tokenizer_json.json
    // - text_encoder.onnx
    // - encodec_decode.onnx
    // That's why they are not prefixed by the model size. Files prefix by the model size
    // do vary depending on the model.
    #[rustfmt::skip]
        let remote_file_spec = match args.model {
        Model::Small => vec![
            (urls::CONFIG_SMALL, "v1/small/config.json"),
            (urls::TOKENIZER_JSON_SMALL, "v1/tokenizer_json.json"),
            (urls::TEXT_ENCODER_SMALL, "v1/text_encoder.onnx"),
            (urls::DECODER_SMALL, "v1/small/decoder_model_merged.onnx"),
            (urls::ENCODEC_DECODE_SMALL, "v1/encodec_decode.onnx"),
        ],
        Model::SmallQuant => vec![
            (urls::CONFIG_SMALL, "v1/small/config.json"),
            (urls::TOKENIZER_JSON_SMALL, "v1/tokenizer_json.json"),
            (urls::TEXT_ENCODER_SMALL, "v1/text_encoder.onnx"),
            (urls::DECODER_SMALL_QUANTIZED, "v1/small/decoder_model_merged_quantized.onnx"),
            (urls::ENCODEC_DECODE_SMALL, "v1/encodec_decode.onnx"),
        ],
        Model::Medium => vec![
            (urls::CONFIG_MEDIUM, "v1/medium/config.json"),
            (urls::TOKENIZER_JSON_MEDIUM, "v1/tokenizer_json.json"),
            (urls::TEXT_ENCODER_MEDIUM, "v1/text_encoder.onnx"),
            (urls::DECODER_MEDIUM, "v1/medium/decoder_model_merged.onnx"),
            (urls::ENCODEC_DECODE_MEDIUM, "v1/encodec_decode.onnx"),
            // Files below will just be downloaded,
            (urls::DECODER_DATA_MEDIUM, "v1/medium/decoder_model_merged.onnx_data"),
        ],
        Model::Large => vec![
            (urls::CONFIG_LARGE, "v1/large/config.json"),
            (urls::TOKENIZER_JSON_LARGE, "v1/tokenizer_json.json"),
            (urls::TEXT_ENCODER_LARGE, "v1/text_encoder.onnx"),
            (urls::DECODER_LARGE, "v1/large/decoder_model_merged.onnx"),
            (urls::ENCODEC_DECODE_LARGE, "v1/encodec_decode.onnx"),
            // Files below will just be downloaded,
            (urls::DECODER_DATA_LARGE, "v1/large/decoder_model_merged.onnx_data"),
        ],
    };

    let mut longest_name = 0;
    let mut has_to_download = args.force_download;
    for (_, local_filename) in remote_file_spec.iter() {
        longest_name = longest_name.max(local_filename.len());
        has_to_download = has_to_download || !Storage::exists(local_filename).await?
    }

    if has_to_download {
        println!("Some AI models need to be downloaded");
    }
    let m = LoadingBarFactor::multi();
    let mut tasks = vec![];
    for (remote_file, local_filename) in remote_file_spec {
        let name = format!("{local_filename: <width$}", width = longest_name);
        let bar = m.add(LoadingBarFactor::download_bar(&name));
        tasks.push(tokio::spawn(Storage::remote_data_file(
            remote_file,
            local_filename,
            args.force_download,
            move |el, t| bar.update_elapsed_total(el, t),
        )));
    }
    let mut results = VecDeque::new();
    for task in tasks {
        results.push_back(task.await??);
    }
    m.clear()?;

    Ok(MusicGenLoadOptions {
        config: results.pop_front().unwrap(),
        tokenizer: results.pop_front().unwrap(),
        text_encoder: results.pop_front().unwrap(),
        decoder_model_merged: results.pop_front().unwrap(),
        audio_encodec_decode: results.pop_front().unwrap(),
    })
}
