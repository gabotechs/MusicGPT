use clap::Parser;

use crate::music_gen::MusicGen;

mod delay_pattern_mask_ids;
mod logits;
mod music_gen;
mod music_gen_inputs;
mod music_gen_outputs;

const SAMPLING_RATE: u32 = 32000;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    prompt: String,

    #[arg(long, default_value = "10")]
    secs: usize,
}

#[tokio::main]
async fn main() -> ort::Result<()> {
    let args = Args::parse();

    let music_gen = MusicGen::load().await?;

    let samples = music_gen.generate(&args.prompt, args.secs).await?;

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
    Ok(())
}
