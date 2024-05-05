use std::collections::VecDeque;
use std::error::Error;
use std::fmt::Write;
use std::time::Duration;

use clap::{Parser, ValueEnum};
use indicatif::{MultiProgress, ProgressBar, ProgressState, ProgressStyle};
use ort::{CoreMLExecutionProvider, CUDAExecutionProvider, DirectMLExecutionProvider, TensorRTExecutionProvider};

use crate::audio_manager::AudioManager;
use crate::music_gen::{INPUT_IDS_BATCH_PER_SECOND, MusicGen, MusicGenLoadOptions};
use crate::storage::Storage;

mod audio_manager;
mod delay_pattern_mask_ids;
mod logits;
mod music_gen;
mod music_gen_config;
mod music_gen_inputs;
mod music_gen_outputs;
mod storage;

// These constants reflect a 1:1 what's present in the https://huggingface.co/gabotechs/music_gen
// repo. Even if they have different URLs, a lot of the files turn out to be the same, like the
// tokenizers, de encodec_decode.onnx model and the text_encoder.onnx model.
const CONFIG_SMALL: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_small/config.json";
const TOKENIZER_JSON_SMALL: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_small/tokenizer.json";
const TEXT_ENCODER_SMALL: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_small/text_encoder.onnx";
const DECODER_SMALL: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_small/decoder_model_merged.onnx";
const DECODER_SMALL_QUANTIZED: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_small/decoder_model_merged_quantized.onnx";
const ENCODEC_DECODE_SMALL: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_small/encodec_decode.onnx";

const CONFIG_MEDIUM: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_medium/config.json";
const TOKENIZER_JSON_MEDIUM: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_medium/tokenizer.json";
const TEXT_ENCODER_MEDIUM: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_medium/text_encoder.onnx";
const DECODER_MEDIUM: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_medium/decoder_model_merged.onnx";
const DECODER_DATA_MEDIUM: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_medium/decoder_model_merged.onnx_data";
const ENCODEC_DECODE_MEDIUM: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_medium/encodec_decode.onnx";

const CONFIG_LARGE: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_large/config.json";
const TOKENIZER_JSON_LARGE: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_large/tokenizer.json";
const TEXT_ENCODER_LARGE: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_large/text_encoder.onnx";
const DECODER_LARGE: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_large/decoder_model_merged.onnx";
const DECODER_DATA_LARGE: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_large/decoder_model_merged.onnx_data";
const ENCODEC_DECODE_LARGE: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_large/encodec_decode.onnx";

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
    cpu: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    if args.cpu {
        ort::init().commit()?
    } else {
        ort::init()
            .with_execution_providers([
                // Prefer TensorRT over CUDA.
                TensorRTExecutionProvider::default().build(),
                CUDAExecutionProvider::default().build(),
                // Use DirectML on Windows if NVIDIA EPs are not available
                DirectMLExecutionProvider::default().build(),
                // Or use ANE on Apple platforms
                CoreMLExecutionProvider::default().build(),
            ]).commit()?;
    };

    // Note that here some destination paths are exactly the same no matter the model.
    // This is because files are exactly the same, and if someone tried model "small",
    // we do not want to force them to re-download repeated files. The following files
    // are the same independently of the model:
    // - tokenizer_json.json
    // - text_encoder.onnx
    // - encodec_decode.onnx
    // That's why they are not prefixed by the model size. Files prefix by the model size
    // do vary depending on the model.
    let remote_file_spec = match args.model {
        Model::Small => vec![
            (CONFIG_SMALL, "v1/small/config.json"),
            (TOKENIZER_JSON_SMALL, "v1/tokenizer_json.json"),
            (TEXT_ENCODER_SMALL, "v1/text_encoder.onnx"),
            (DECODER_SMALL, "v1/small/decoder_model_merged.onnx"),
            (ENCODEC_DECODE_SMALL, "v1/encodec_decode.onnx"),
        ],
        Model::SmallQuant => vec![
            (CONFIG_SMALL, "v1/small/config.json"),
            (TOKENIZER_JSON_SMALL, "v1/tokenizer_json.json"),
            (TEXT_ENCODER_SMALL, "v1/text_encoder.onnx"),
            (DECODER_SMALL_QUANTIZED, "v1/small/decoder_model_merged_quantized.onnx"),
            (ENCODEC_DECODE_SMALL, "v1/encodec_decode.onnx"),
        ],
        Model::Medium => vec![
            (CONFIG_MEDIUM, "v1/medium/config.json"),
            (TOKENIZER_JSON_MEDIUM, "v1/tokenizer_json.json"),
            (TEXT_ENCODER_MEDIUM, "v1/text_encoder.onnx"),
            (DECODER_MEDIUM, "v1/medium/decoder_model_merged.onnx"),
            (ENCODEC_DECODE_MEDIUM, "v1/encodec_decode.onnx"),
            // Files below will just be downloaded,
            (DECODER_DATA_MEDIUM, "v1/medium/decoder_model_merged.onnx_data"),
        ],
        Model::Large => vec![
            (CONFIG_LARGE, "v1/large/config.json"),
            (TOKENIZER_JSON_LARGE, "v1/tokenizer_json.json"),
            (TEXT_ENCODER_LARGE, "v1/text_encoder.onnx"),
            (DECODER_LARGE, "v1/large/decoder_model_merged.onnx"),
            (ENCODEC_DECODE_LARGE, "v1/encodec_decode.onnx"),
            // Files below will just be downloaded,
            (DECODER_DATA_LARGE, "v1/large/decoder_model_merged.onnx_data"),
        ],
    };

    let m = MultiProgress::new();
    let mut tasks = vec![];
    let mut longest_name = 0;
    for (_, local_filename) in remote_file_spec.iter() {
        longest_name = longest_name.max(local_filename.len())
    }

    for (remote_file, local_filename) in remote_file_spec {
        let bar = m.add(make_bar(
            &format!("{local_filename: >width$}", width = longest_name),
            1,
        ));
        tasks.push(tokio::spawn(Storage::remote_data_file(
            remote_file,
            local_filename,
            args.force_download,
            move |elapsed, total| {
                bar.set_length(total as u64);
                bar.set_position(elapsed as u64);
            },
        )));
    }
    let mut results = VecDeque::new();
    for task in tasks {
        results.push_back(task.await??);
    }
    m.clear()?;

    let music_gen_load_options = MusicGenLoadOptions {
        config: results.pop_front().unwrap(),
        tokenizer: results.pop_front().unwrap(),
        text_encoder: results.pop_front().unwrap(),
        decoder_model_merged: results.pop_front().unwrap(),
        audio_encodec_decode: results.pop_front().unwrap(),
    };

    macro_rules! run_model {
        ($t: ty) => {{
            let spinner = make_spinner("Loading models");
            let music_gen = MusicGen::<$t>::load(music_gen_load_options).await?;
            spinner.finish_and_clear();
            let bar = make_bar("Generating audio", INPUT_IDS_BATCH_PER_SECOND);
            let stream = music_gen.generate(&args.prompt, args.secs, |elapsed, total| {
                    bar.set_length(total as u64);
                    bar.set_position(elapsed as u64);
                })
                .await?;
            bar.finish_and_clear();
            stream
        }};
    }
    let mut sample_stream = match args.model {
        Model::Small | Model::Medium | Model::Large => run_model!(f32),
        // Surprisingly, the quantized model also used f32 for inputs and outputs.
         Model::SmallQuant => run_model!(f32),
        // Whenever this can run other types (like fp16), more entries should go here.
    };

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

    let spinner = make_spinner("Playing audio...");
    let audio_player = AudioManager::default();
    let (_play_from_queue, store_as_wav) = tokio::join!(
        audio_player.play_from_queue(data.clone()),
        audio_player.store_as_wav(data, output)
    );
    // audio playback is just a best effort operation, we don't care if it fails.
    store_as_wav?;
    spinner.finish_and_clear();

    Ok(())
}

fn make_spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.enable_steady_tick(Duration::from_millis(120));
    pb.set_style(ProgressStyle::with_template("{spinner:.blue} {msg}").unwrap());
    pb.set_message(msg.to_string());
    pb
}

fn make_bar(prefix: &str, len: usize) -> ProgressBar {
    let pb = ProgressBar::new(len as u64);
    pb.set_style(
        ProgressStyle::with_template(
            &(prefix.to_string()
                + " {spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] ({eta})"),
        )
            .unwrap()
            .with_key("eta", |state: &ProgressState, w: &mut dyn Write| {
                write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap()
            })
            .progress_chars("#>-"),
    );
    pb
}
