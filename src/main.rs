use clap::Parser;

use crate::music_gen::MusicGen;

mod input_ids;
mod logits;
mod music_gen;
mod past_key_values;
mod session_input_builder;
mod tensor;

const SAMPLING_RATE: u32 = 32000;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "Create a relaxing LoFi song")]
    prompt: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let music_gen = MusicGen::load().await?;

    let mut all_ids = [(); 4].map(|()| vec![]);
    let mut generator = music_gen.generate(&args.prompt)?;
    while let Some(ids) = generator.recv().await {
        for (i, id) in ids.into_iter().enumerate() {
            all_ids[i].push(id)
        }
    }

    println!("generating audio...");
    let encoded = music_gen.encode_audio(all_ids)?;

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLING_RATE,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create("out.wav", spec).unwrap();
    for sample in encoded {
        writer.write_sample(sample).unwrap();
    }
    Ok(())
}
