use std::collections::VecDeque;
use std::time::Duration;
use async_trait::async_trait;
use uuid::Uuid;
use crate::ui::backend_ai::{BackendAiOutboundMsg, JobProcessor};

impl BackendAiOutboundMsg {
    pub(crate) fn unwrap_progress(self) -> (Uuid, f32) {
        match self {
            BackendAiOutboundMsg::AudioGenerationProgress(p) => p,
            _ => panic!("msg was not AudioGenerationProgress, it was {self:?}"),
        }
    }

    pub(crate) fn unwrap_response(self) -> (Uuid, String) {
        match self {
            BackendAiOutboundMsg::AudioGenerationResponse(p) => p,
            _ => panic!("msg was not AudioGenerationResponse, it was {self:?}"),
        }
    }
    
    pub(crate) fn unwrap_err(self) -> (Uuid, String) {
        match self {
            BackendAiOutboundMsg::AudioGenerationFailure(p) => p,
            _ => panic!("msg was not AudioGenerationFailure, it was {self:?}"),
        }
    }
}

#[derive(Default)]
pub struct DummyJobProcessor {
    wait_scale: Duration
}

impl DummyJobProcessor {
    pub fn new(wait_scale: Duration)  -> Self {
        Self { wait_scale }
    }
}

#[async_trait]
impl JobProcessor for DummyJobProcessor {
    fn process(
        &self,
        prompt: &str,
        secs: usize,
        on_progress: Box<dyn Fn(f32) -> bool + Sync + Send + 'static>,
    ) -> ort::Result<VecDeque<f32>> {
        let mut result = VecDeque::new();
        for i in 0..secs {
            if prompt == format!("fail at {i}") {
                return Err(ort::Error::CustomError(format!("Failed at {i}").into()))
            }
            std::thread::sleep(self.wait_scale);
            result.push_back(i as f32);
            let should_exit = on_progress(result.len() as f32 / secs as f32);
            if should_exit {
                return Err(ort::Error::CustomError("Aborted".into()))
            }
        }

        Ok(result)
    }
}
