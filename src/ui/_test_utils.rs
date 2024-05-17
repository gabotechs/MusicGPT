use std::collections::VecDeque;
use std::time::Duration;

use async_trait::async_trait;

use crate::ui::backend_ai::{BackendAiOutboundMsg, JobProcessor};
use crate::ui::messages::{AudioGenerationError, AudioGenerationProgress, AudioGenerationResult, Init, OutboundMsg};

impl OutboundMsg {
    pub(crate) fn unwrap_init(self) -> Init {
        match self {
            OutboundMsg::Init(p) => p,
            _ => panic!("msg was not OutboundMsg::Init, it was {self:?}"),
        }
    }


    pub(crate) fn unwrap_progress(self) -> AudioGenerationProgress {
        match self {
            OutboundMsg::Progress(p) => p,
            _ => panic!("msg was not OutboundMsg::Progress, it was {self:?}"),
        }
    }

    pub(crate) fn unwrap_result(self) -> AudioGenerationResult {
        match self {
            OutboundMsg::Result(p) => p,
            _ => panic!("msg was not OutboundMsg::Result, it was {self:?}"),
        }
    }

    pub(crate) fn unwrap_err(self) -> AudioGenerationError {
        match self {
            OutboundMsg::Error(p) => p,
            _ => panic!("msg was not OutboundMsg::Error, it was {self:?}"),
        }
    }
}

impl BackendAiOutboundMsg {
    pub(crate) fn unwrap_progress(self) -> (String, f32) {
        match self {
            BackendAiOutboundMsg::Progress(p) => p,
            _ => panic!("msg was not Progress, it was {self:?}"),
        }
    }

    pub(crate) fn unwrap_response(self) -> (String, VecDeque<f32>) {
        match self {
            BackendAiOutboundMsg::Response(p) => p,
            _ => panic!("msg was not Response, it was {self:?}"),
        }
    }

    pub(crate) fn unwrap_err(self) -> (String, String) {
        match self {
            BackendAiOutboundMsg::Failure(p) => p,
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
