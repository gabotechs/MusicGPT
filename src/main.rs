use clap::Parser;

use crate::music_gen::MusicGen;

mod delay_pattern_mask_ids;
mod logits;
mod music_gen;
mod music_gen_inputs;
mod music_gen_outputs;
mod tensor;

const SAMPLING_RATE: u32 = 32000;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "Create a relaxing LoFi song")]
    prompt: String,
}

#[tokio::main]
async fn main() -> ort::Result<()> {
    let args = Args::parse();

    let music_gen = MusicGen::load().await?;

    let mut all_ids = [(); 4].map(|()| vec![]);
    let mut generator = music_gen.generate(&args.prompt)?;
    while let Some(ids) = generator.recv().await {
        let ids = match ids {
            Ok(ids) => ids,
            Err(err) => return Err(err),
        };
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
