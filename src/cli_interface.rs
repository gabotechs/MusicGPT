use std::collections::VecDeque;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use regex::Regex;
use text_io::read;

use crate::audio_manager::{AudioManager, AudioStream};
use crate::loading_bar_factory::LoadingBarFactor;
use crate::music_gen_audio_encodec::MusicGenAudioEncodec;
use crate::music_gen_decoder::MusicGenDecoder;
use crate::music_gen_text_encoder::MusicGenTextEncoder;

const INPUT_IDS_BATCH_PER_SECOND: usize = 50;

#[allow(unused_assignments, unused_variables)]
pub async fn cli_interface(
    text_encoder: MusicGenTextEncoder,
    decoder: Box<dyn MusicGenDecoder>,
    audio_encodec: MusicGenAudioEncodec,
    mut secs: usize,
    mut prompt: String,
    mut output: String,
    no_playback: bool,
) -> anyhow::Result<()> {
    let secs_re = Regex::new("--secs[ =](\\d+)")?;
    let output_re = Regex::new(r"--output[ =]([.a-zA-Z_-]+)")?;

    let audio_player = AudioManager::default();
    // This variable holds the audio stream. The stream stops when this is dropped,
    // so we need to maintain it referenced here.
    let mut curr_stream: Option<AudioStream> = None;

    loop {
        if prompt.is_empty() {
            print!(">>> ");
            prompt = read!("{}\n");
            if prompt == "exit" {
                return Ok(())
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
        let token_stream = decoder.generate_tokens(
            last_hidden_state,
            attention_mask,
            Arc::new(AtomicBool::default()),
            max_len,
        )?;
        let bar = LoadingBarFactor::bar("Generating audio");
        let mut data = VecDeque::new();
        while let Ok(tokens) = token_stream.recv() {
            data.push_back(tokens?);
            bar.update_elapsed_total(data.len(), max_len)
        }

        // Third, encode the tokens into audio.
        let samples = audio_encodec.encode(data)?;

        // Last, play the audio.
        if !output.ends_with(".wav") {
            output += ".wav";
        }
        if !no_playback {
            let samples_copy = samples.clone();
            let stream = audio_player.play_from_queue(samples_copy);
            if let Ok(stream) = stream {
                curr_stream = Some(stream);
            }
        }
        audio_player.store_as_wav(samples, output.clone())?;
        prompt = "".into();
    }
}
