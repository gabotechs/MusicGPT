use crate::config::Config;
use clap::{Parser, ValueEnum};
use indicatif::{MultiProgress, ProgressBar, ProgressState, ProgressStyle};
use std::error::Error;
use std::fmt::Write;
use std::time::Duration;

use crate::music_gen::{MusicGen, MusicGenLoadOptions};

mod config;
mod delay_pattern_mask_ids;
mod logits;
mod music_gen;
mod music_gen_inputs;
mod music_gen_outputs;

const SAMPLING_RATE: u32 = 32000;
const TOKENIZER_JSON_SMALL: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_small/tokenizer.json?download=true";
const TEXT_ENCODER_SMALL: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_small/text_encoder.onnx?download=true";
const DECODER_SMALL: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_small/decoder_model_merged.onnx?download=true";
const ENCODEC_DECODE_SMALL: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_small/encodec_decode.onnx?download=true";
const TOKENIZER_JSON_MEDIUM: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_medium/tokenizer.json?download=true";
const TEXT_ENCODER_MEDIUM: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_medium/text_encoder.onnx?download=true";
const DECODER_MEDIUM: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_medium/decoder_model_merged.onnx?download=true";
const ENCODEC_DECODE_MEDIUM: &str = "https://huggingface.co/gabotechs/music_gen/resolve/main/musicgen_onnx_medium/encodec_decode.onnx?download=true";

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
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let cfg = Config::new();

    let models = match args.model {
        Model::Small => (
            (TOKENIZER_JSON_SMALL, "tokenizer_json_small.json"),
            (TEXT_ENCODER_SMALL, "text_encoder_small.onnx"),
            (DECODER_SMALL, "decoder_model_merged_small.onnx"),
            (ENCODEC_DECODE_SMALL, "encodec_decode_small.onnx"),
        ),
        Model::Medium => (
            (TOKENIZER_JSON_MEDIUM, "tokenizer_json_medium.json"),
            (TEXT_ENCODER_MEDIUM, "text_encoder_medium.onnx"),
            (DECODER_MEDIUM, "decoder_model_merged_medium.onnx"),
            (ENCODEC_DECODE_MEDIUM, "encodec_decode_medium.onnx"),
        ),
        Model::Large => panic!("\"large\" model is not supported yet"),
    };

    let m = MultiProgress::new();
    let pb1 = m.add(make_bar("tokenizer   ", 1));
    let pb2 = m.insert_after(&pb1, make_bar("text_encoder", 1));
    let pb3 = m.insert_after(&pb2, make_bar("decoder     ", 1));
    let pb4 = m.insert_after(&pb3, make_bar("encodec     ", 1));
    let result = tokio::try_join!(
        cfg.remote_data_file(models.0 .0, models.0 .1, |elapsed, total| {
            pb1.set_length(total as u64);
            pb1.set_position(elapsed as u64);
        }),
        cfg.remote_data_file(models.1 .0, models.1 .1, |elapsed, total| {
            pb2.set_length(total as u64);
            pb2.set_position(elapsed as u64);
        }),
        cfg.remote_data_file(models.2 .0, models.2 .1, |elapsed, total| {
            pb3.set_length(total as u64);
            pb3.set_position(elapsed as u64);
        }),
        cfg.remote_data_file(models.3 .0, models.3 .1, |elapsed, total| {
            pb4.set_length(total as u64);
            pb4.set_position(elapsed as u64);
        }),
    )?;
    pb1.finish_with_message("done");
    pb2.finish_with_message("done");
    pb3.finish_with_message("done");
    pb4.finish_with_message("done");
    m.clear()?;

    let spinner = make_spinner("Loading models");
    let music_gen = MusicGen::load(MusicGenLoadOptions {
        tokenizer: result.0,
        text_encoder: result.1,
        decoder_model_merged: result.2,
        audio_encodec_decode: result.3,
    })
    .await?;
    spinner.finish_and_clear();

    let bar = make_bar("Generating audio", 1);
    let samples = music_gen
        .generate(&args.prompt, args.secs, |elapsed, total| {
            bar.set_length(total as u64);
            bar.set_position(elapsed as u64)
        })
        .await?;
    bar.finish_and_clear();

    let spinner = make_spinner("Encoding audio");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLING_RATE,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create("out.wav", spec).unwrap();
    for sample in samples {
        writer.write_sample(sample).unwrap();
    }
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
