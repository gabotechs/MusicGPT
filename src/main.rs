use std::collections::VecDeque;
use std::error::Error;
use std::fmt::Write;
use std::time::Duration;

use clap::{Parser, ValueEnum};
use indicatif::{MultiProgress, ProgressBar, ProgressState, ProgressStyle};

use crate::audio_manager::AudioManager;
use crate::music_gen::{MusicGen, MusicGenLoadOptions};
use crate::storage::Storage;

mod audio_manager;
mod delay_pattern_mask_ids;
mod logits;
mod music_gen;
mod music_gen_config;
mod music_gen_inputs;
mod music_gen_outputs;
mod storage;

const CONFIG_SMALL: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_small/config.json?download=true";
const TOKENIZER_JSON_SMALL: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_small/tokenizer.json?download=true";
const TEXT_ENCODER_SMALL: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_small/text_encoder.onnx?download=true";
const DECODER_SMALL: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_small/decoder_model_merged.onnx?download=true";
const ENCODEC_DECODE_SMALL: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_small/encodec_decode.onnx?download=true";

const CONFIG_MEDIUM: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_medium/config.json?download=true";
const TOKENIZER_JSON_MEDIUM: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_medium/tokenizer.json?download=true";
const TEXT_ENCODER_MEDIUM: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_medium/text_encoder.onnx?download=true";
const DECODER_MEDIUM: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_medium/decoder_model_merged.onnx?download=true";
const DECODER_DATA_MEDIUM: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_medium/decoder_model_merged.onnx_data?download=true";
const ENCODEC_DECODE_MEDIUM: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_medium/encodec_decode.onnx?download=true";

const CONFIG_LARGE: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_large/config.json?download=true";
const TOKENIZER_JSON_LARGE: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_large/tokenizer.json?download=true";
const TEXT_ENCODER_LARGE: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_large/text_encoder.onnx?download=true";
const DECODER_LARGE: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_large/decoder_model_merged.onnx?download=true";
const DECODER_DATA_LARGE: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_large/decoder_model_merged.onnx_data?download=true";
const ENCODEC_DECODE_LARGE: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_large/encodec_decode.onnx?download=true";

#[derive(Clone, ValueEnum)]
enum Model {
    Small,
    Medium,
    Large,
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    prompt: String,

    #[arg(long, default_value = "10")]
    secs: usize,

    #[arg(long, default_value = "small")]
    model: Model,

    #[arg(long, default_value = "")]
    output: String,

    #[arg(long, default_value = "false")]
    force_download: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let remote_file_spec = match args.model {
        Model::Small => vec![
            (CONFIG_SMALL, "small/config.json"),
            (TOKENIZER_JSON_SMALL, "small/tokenizer_json.json"),
            (TEXT_ENCODER_SMALL, "small/text_encoder.onnx"),
            (DECODER_SMALL, "small/decoder_model_merged.onnx"),
            (ENCODEC_DECODE_SMALL, "small/encodec_decode.onnx"),
        ],
        Model::Medium => vec![
            (CONFIG_MEDIUM, "medium/config.json"),
            (TOKENIZER_JSON_MEDIUM, "medium/tokenizer_json.json"),
            (TEXT_ENCODER_MEDIUM, "medium/text_encoder.onnx"),
            (DECODER_MEDIUM, "medium/decoder_model_merged.onnx"),
            (ENCODEC_DECODE_MEDIUM, "medium/encodec_decode.onnx"),
            // Files below will just be downloaded,
            (DECODER_DATA_MEDIUM, "medium/decoder_model_merged.onnx_data"),
        ],
        Model::Large => vec![
            (CONFIG_LARGE, "large/config.json"),
            (TOKENIZER_JSON_LARGE, "large/tokenizer_json.json"),
            (TEXT_ENCODER_LARGE, "large/text_encoder.onnx"),
            (DECODER_LARGE, "large/decoder_model_merged.onnx"),
            (ENCODEC_DECODE_LARGE, "large/encodec_decode.onnx"),
            // Files below will just be downloaded,
            (DECODER_DATA_LARGE, "large/decoder_model_merged.onnx_data"),
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

    let spinner = make_spinner("Loading models");
    let music_gen = MusicGen::load(MusicGenLoadOptions {
        config: results.pop_front().unwrap(),
        tokenizer: results.pop_front().unwrap(),
        text_encoder: results.pop_front().unwrap(),
        decoder_model_merged: results.pop_front().unwrap(),
        audio_encodec_decode: results.pop_front().unwrap(),
    })
        .await?;
    spinner.finish_and_clear();

    let bar = make_bar("Generating audio", 1);
    let mut sample_stream = music_gen
        .generate(&args.prompt, args.secs, |elapsed, total| {
            bar.set_length(total as u64);
            bar.set_position(elapsed as u64)
        })
        .await?;
    bar.finish_and_clear();

    let output = if args.output.is_empty() {
        "music-gen.wav".to_string()
    } else if !args.output.ends_with(".wav") {
        args.output + ".wav"
    } else {
        args.output
    };

    let mut data = VecDeque::new();
    while let Some(sample) = sample_stream.recv().await {
        data.push_back(sample?)
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
