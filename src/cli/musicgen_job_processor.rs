use std::collections::VecDeque;

use crate::backend::JobProcessor;
use crate::cli::musicgen_models::MusicGenModels;
use crate::cli::INPUT_IDS_BATCH_PER_SECOND;

pub struct MusicGenJobProcessor {
    pub name: String,
    pub device: String,
    pub musicgen_models: MusicGenModels,
}

impl JobProcessor for MusicGenJobProcessor {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn device(&self) -> String {
        self.device.clone()
    }

    fn process(
        &self,
        prompt: &str,
        secs: usize,
        on_progress: Box<dyn Fn(f32) -> bool + Sync + Send + 'static>,
    ) -> ort::Result<VecDeque<f32>> {
        let max_len = secs * INPUT_IDS_BATCH_PER_SECOND;

        let (lhs, am) = self.musicgen_models.encode_text(prompt)?;
        let token_stream = self.musicgen_models.generate_tokens(lhs, am, max_len)?;

        let mut data = VecDeque::new();
        while let Ok(tokens) = token_stream.recv() {
            data.push_back(tokens?);
            let should_exit = on_progress(data.len() as f32 / max_len as f32);
            if should_exit {
                return Err(ort::Error::new("Aborted"));
            }
        }

        self.musicgen_models.encode_audio(data)
    }
}
