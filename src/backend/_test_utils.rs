use std::collections::VecDeque;
use std::time::Duration;

use async_trait::async_trait;
use rand::distributions::Alphanumeric;
use rand::{Rng, thread_rng};

use crate::backend::audio_generation_fanout::{AudioGenerationError, AudioGenerationProgress, AudioGenerationResult, AudioGenerationStart, GenerationMessage};
use crate::backend::audio_generation_backend::{AudioGenerationRequest, BackendOutboundMsg, JobProcessor};
use crate::backend::music_gpt_chat::Chat;
use crate::backend::music_gpt_ws_handler::{Info, OutboundMsg};
use crate::storage::AppFs;

impl OutboundMsg {
    pub(crate) fn unwrap_info(self) -> Info {
        match self {
            OutboundMsg::Info(p) => p,
            _ => panic!("msg was not OutboundMsg::Init, it was {self:?}"),
        }
    }

    pub(crate) fn unwrap_chats(self) -> Vec<Chat> {
        match self {
            OutboundMsg::Chats(p) => p,
            _ => panic!("msg was not OutboundMsg::Chats, it was {self:?}"),
        }
    }

    pub(crate) fn unwrap_start(self) -> AudioGenerationStart {
        match self {
            OutboundMsg::Generation(GenerationMessage::Start(p)) => p,
            _ => panic!("msg was not GenerationMessage::Start, it was {self:?}"),
        }
    }

    pub(crate) fn unwrap_progress(self) -> AudioGenerationProgress {
        match self {
            OutboundMsg::Generation(GenerationMessage::Progress(p)) => p,
            _ => panic!("msg was not GenerationMessage::Progress, it was {self:?}"),
        }
    }

    pub(crate) fn unwrap_result(self) -> AudioGenerationResult {
        match self {
            OutboundMsg::Generation(GenerationMessage::Result(p)) => p,
            _ => panic!("msg was not GenerationMessage::Result, it was {self:?}"),
        }
    }

    pub(crate) fn unwrap_err(self) -> AudioGenerationError {
        match self {
            OutboundMsg::Generation(GenerationMessage::Error(p)) => p,
            _ => panic!("msg was not GenerationMessage::Error, it was {self:?}"),
        }
    }
}

impl BackendOutboundMsg {
    pub(crate) fn unwrap_start(self) -> AudioGenerationRequest {
        match self {
            BackendOutboundMsg::Start(p) => p,
            _ => panic!("msg was not Progress, it was {self:?}"),
        }
    }

    pub(crate) fn unwrap_progress(self) -> (String, f32) {
        match self {
            BackendOutboundMsg::Progress(p) => p,
            _ => panic!("msg was not Progress, it was {self:?}"),
        }
    }

    pub(crate) fn unwrap_response(self) -> (String, VecDeque<f32>) {
        match self {
            BackendOutboundMsg::Response(p) => p,
            _ => panic!("msg was not Response, it was {self:?}"),
        }
    }

    pub(crate) fn unwrap_err(self) -> (String, String) {
        match self {
            BackendOutboundMsg::Failure(p) => p,
            _ => panic!("msg was not Failure, it was {self:?}"),
        }
    }
}

#[derive(Default)]
pub struct DummyJobProcessor {
    wait_scale: Duration,
}

impl DummyJobProcessor {
    pub fn new(wait_scale: Duration) -> Self {
        Self { wait_scale }
    }
}

#[async_trait]
impl JobProcessor for DummyJobProcessor {
    fn name(&self) -> String {
        "Dummy".to_string()
    }

    fn device(&self) -> String {
        "Cpu".to_string()
    }

    fn process(
        &self,
        prompt: &str,
        secs: usize,
        on_progress: Box<dyn Fn(f32) -> bool + Sync + Send + 'static>,
    ) -> ort::Result<VecDeque<f32>> {
        let mut result = VecDeque::new();
        for i in 0..secs {
            if prompt == format!("fail at {i}") {
                return Err(ort::Error::CustomError(format!("Failed at {i}").into()));
            }
            std::thread::sleep(self.wait_scale);
            result.push_back(i as f32);
            let should_exit = on_progress(result.len() as f32 / secs as f32);
            if should_exit {
                return Err(ort::Error::CustomError("Aborted".into()));
            }
        }

        Ok(result)
    }
}

pub fn rand_string() -> String {
    thread_rng()
        .sample_iter(&Alphanumeric)
        .take(7)
        .map(char::from)
        .collect()
}

impl AppFs {
    pub fn new_tmp() -> Self {
        Self::new(format!("/tmp/musicgpt-tests/{}", rand_string()))
    }
}
