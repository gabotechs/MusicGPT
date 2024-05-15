use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::anyhow;
use cpal::{ChannelCount, SampleFormat, SampleRate, Stream, SupportedBufferSize, SupportedStreamConfig};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

const DEFAULT_SAMPLING_RATE: u32 = 32000;

pub struct AudioManager {
    host: cpal::Host,
    sample_format: SampleFormat,
    sampling_rate: u32,
    n_channels: u16,
}

impl Default for AudioManager {
    fn default() -> Self {
        let host = cpal::default_host();

        Self {
            host,
            sampling_rate: DEFAULT_SAMPLING_RATE,
            sample_format: SampleFormat::F32,
            n_channels: 1,
        }
    }
}

pub struct AudioStream {
    pub stream: Stream,
    pub duration: Duration
}


unsafe impl Send for AudioStream {}
unsafe impl Sync for AudioStream {}


impl AudioManager {
    pub fn play_from_queue(&self, mut v: VecDeque<f32>) -> anyhow::Result<AudioStream> {
        let time = 1000 * v.len() / self.sampling_rate as usize;
        let channels = self.n_channels;

        let config = SupportedStreamConfig::new(
            ChannelCount::from(channels),
            SampleRate(self.sampling_rate),
            SupportedBufferSize::Unknown,
            self.sample_format,
        );

        let device = match self.host.default_output_device() {
            None => return Err(anyhow!("No audio device")),
            Some(v) => v,
        };
        let stream = device.build_output_stream(
            &config.into(),
            move |output: &mut [f32], _: &cpal::OutputCallbackInfo| {
                for frame in output.chunks_mut(channels as usize) {
                    for sample in frame.iter_mut() {
                        *sample = v.pop_front().unwrap_or_default()
                    }
                }
            },
            |_err| {},
            None,
        )?;

        stream.play()?;
        Ok(AudioStream {
            stream,
            duration: Duration::from_millis(time as u64)
        })
    }

    pub fn store_as_wav(
        &self,
        v: VecDeque<f32>,
        out_path: impl Into<PathBuf>,
    ) -> hound::Result<()> {
        let spec = hound::WavSpec {
            channels: self.n_channels,
            sample_rate: self.sampling_rate,
            bits_per_sample: match self.sample_format {
                SampleFormat::I8 => 8,
                SampleFormat::I16 => 16,
                SampleFormat::I32 => 32,
                SampleFormat::I64 => 64,
                SampleFormat::U8 => 8,
                SampleFormat::U16 => 16,
                SampleFormat::U32 => 32,
                SampleFormat::U64 => 64,
                SampleFormat::F32 => 32,
                SampleFormat::F64 => 64,
                unknown => panic!("unknown sample format {unknown}"),
            },
            sample_format: match self.sample_format {
                SampleFormat::I8 => hound::SampleFormat::Int,
                SampleFormat::I16 => hound::SampleFormat::Int,
                SampleFormat::I32 => hound::SampleFormat::Int,
                SampleFormat::I64 => hound::SampleFormat::Int,
                SampleFormat::U8 => hound::SampleFormat::Int,
                SampleFormat::U16 => hound::SampleFormat::Int,
                SampleFormat::U32 => hound::SampleFormat::Int,
                SampleFormat::U64 => hound::SampleFormat::Int,
                SampleFormat::F32 => hound::SampleFormat::Float,
                SampleFormat::F64 => hound::SampleFormat::Float,
                unknown => panic!("unknown sample format {unknown}"),
            },
        };

        let mut writer = hound::WavWriter::create(out_path.into(), spec)?;
        for sample in v {
            writer.write_sample(sample)?;
        }

        Ok(())
    }
}
