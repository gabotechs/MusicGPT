use std::collections::VecDeque;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::music_gen_audio_encodec::MusicGenAudioEncodec;
use crate::music_gen_decoder::MusicGenDecoder;
use crate::music_gen_text_encoder::MusicGenTextEncoder;

const INPUT_IDS_BATCH_PER_SECOND: usize = 50;

#[derive(Clone, Debug)]
pub struct AudioGenerationRequest {
    pub id: String,
    pub prompt: String,
    pub secs: usize,
}

#[derive(Clone, Debug)]
pub enum BackendInboundMsg {
    Request(AudioGenerationRequest),
    Abort(String),
}

#[derive(Clone, Debug)]
pub enum BackendOutboundMsg {
    Start(AudioGenerationRequest),
    Response((String, VecDeque<f32>)),
    Failure((String, String)),
    Progress((String, f32)),
}

#[derive(Clone, Debug)]
struct Job {
    req: AudioGenerationRequest,
    abort_token: CancellationToken,
}

impl Job {
    fn new(req: AudioGenerationRequest) -> Self {
        Self {
            req,
            abort_token: CancellationToken::new(),
        }
    }
}

pub trait JobProcessor: Send + Sync {
    fn name(&self) -> String;
    fn device(&self) -> String;
    fn process(
        &self,
        prompt: &str,
        secs: usize,
        on_progress: Box<dyn Fn(f32) -> bool + Sync + Send + 'static>,
    ) -> ort::Result<VecDeque<f32>>;
}

pub struct MusicGenJobProcessor {
    pub name: String,
    pub device: String,
    pub text_encoder: MusicGenTextEncoder,
    pub decoder: Box<dyn MusicGenDecoder>,
    pub audio_encodec: MusicGenAudioEncodec,
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

        let (lhs, am) = self.text_encoder.encode(prompt)?;
        let token_stream = self.decoder.generate_tokens(lhs, am, max_len)?;

        let mut data = VecDeque::new();
        while let Ok(tokens) = token_stream.recv() {
            data.push_back(tokens?);
            let should_exit = on_progress(data.len() as f32 / max_len as f32);
            if should_exit {
                return Err(ort::Error::CustomError("Aborted".into()));
            }
        }

        self.audio_encodec.encode(data)
    }
}

#[derive(Clone)]
pub struct AudioGenerationBackend {
    processor: Arc<dyn JobProcessor>,
    job_queue: Arc<RwLock<VecDeque<Job>>>,
    abort_token: CancellationToken,
}

impl AudioGenerationBackend {
    pub fn new<T: JobProcessor + 'static>(processor: T) -> Self {
        Self {
            processor: Arc::new(processor),
            job_queue: Arc::new(RwLock::new(VecDeque::new())),
            abort_token: CancellationToken::new(),
        }
    }

    fn job_processing_loop(self, outbound_tx: Sender<BackendOutboundMsg>) {
        loop {
            let front = {
                // Immediately drop jq so that the lock is released.
                let jq = self.job_queue.read().unwrap();
                jq.front().cloned()
            };
            let Some(job) = front else {
                if self.abort_token.is_cancelled() {
                    return;
                }
                std::thread::sleep(Duration::from_millis(10));
                continue;
            };

            let _ = outbound_tx.send(BackendOutboundMsg::Start(job.req.clone()));

            let output_tx_clone = outbound_tx.clone();
            let abort_token = self.abort_token.clone();
            let job_id = job.req.id.clone();
            let cbk = Box::new(move |p| {
                let msg = BackendOutboundMsg::Progress((job_id.clone(), p));
                let _ = output_tx_clone.send(msg);
                abort_token.is_cancelled() || job.abort_token.is_cancelled()
            });

            let msg = match self.processor.process(&job.req.prompt, job.req.secs, cbk) {
                Ok(filepath) => BackendOutboundMsg::Response((job.req.id, filepath)),
                Err(err) => BackendOutboundMsg::Failure((job.req.id, err.to_string())),
            };
            let _ = outbound_tx.send(msg);
            self.job_queue.write().unwrap().pop_front();
        }
    }

    fn msg_processing_loop(self, inbound_rx: Receiver<BackendInboundMsg>) {
        while let Ok(msg) = inbound_rx.recv() {
            match msg {
                BackendInboundMsg::Request(req) => {
                    self.job_queue.write().unwrap().push_back(Job::new(req));
                }
                BackendInboundMsg::Abort(id) => {
                    let mut queue = self.job_queue.write().unwrap();
                    let mut to_remove = None;
                    for (i, job) in queue.iter().enumerate() {
                        if job.req.id == id {
                            to_remove = Some(i);
                            job.abort_token.cancel();
                            break;
                        }
                    }
                    if let Some(to_remove) = to_remove {
                        queue.remove(to_remove);
                    }
                }
            }
        }
        self.abort_token.cancel()
    }

    pub fn run(self) -> (Sender<BackendInboundMsg>, Receiver<BackendOutboundMsg>) {
        let (inbound_tx, inbound_rx) = channel::<BackendInboundMsg>();
        let (outbound_tx, outbound_rx) = channel::<BackendOutboundMsg>();

        // Job processing loop.
        let self_clone = self.clone();
        std::thread::spawn(move || self_clone.job_processing_loop(outbound_tx));

        // Communications processing loop.
        std::thread::spawn(move || self.msg_processing_loop(inbound_rx));

        (inbound_tx, outbound_rx)
    }
}

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use crate::backend::_test_utils::DummyJobProcessor;

    use super::*;

    #[test]
    fn processes_job() -> anyhow::Result<()> {
        let backend = AudioGenerationBackend::new(DummyJobProcessor::default());

        let (tx, rx) = backend.run();

        let id = Uuid::new_v4().to_string();
        tx.send(BackendInboundMsg::Request(AudioGenerationRequest {
            id: id.clone(),
            prompt: "".to_string(),
            secs: 4,
        }))?;

        assert_eq!(rx.recv()?.unwrap_start().id, id);
        assert_eq!(rx.recv()?.unwrap_progress().1, 0.25);
        assert_eq!(rx.recv()?.unwrap_progress().1, 0.5);
        assert_eq!(rx.recv()?.unwrap_progress().1, 0.75);
        assert_eq!(rx.recv()?.unwrap_progress().1, 1.0);
        assert_eq!(
            rx.recv()?.unwrap_response().1,
            VecDeque::from([0.0, 1.0, 2.0, 3.0])
        );

        Ok(())
    }

    #[test]
    fn handles_job_failure() -> anyhow::Result<()> {
        let backend = AudioGenerationBackend::new(DummyJobProcessor::default());

        let (tx, rx) = backend.run();

        let id = Uuid::new_v4().to_string();
        tx.send(BackendInboundMsg::Request(AudioGenerationRequest {
            id: id.clone(),
            prompt: "fail at 2".to_string(),
            secs: 4,
        }))?;

        assert_eq!(rx.recv()?.unwrap_start().id, id);
        assert_eq!(rx.recv()?.unwrap_progress().1, 0.25);
        assert_eq!(rx.recv()?.unwrap_progress().1, 0.5);
        assert_eq!(rx.recv()?.unwrap_err().1, "Failed at 2");

        Ok(())
    }

    #[tokio::test]
    async fn handles_job_cancellation() -> anyhow::Result<()> {
        let backend = AudioGenerationBackend::new(DummyJobProcessor::new(Duration::from_millis(100)));

        let (tx, rx) = backend.run();

        let id = Uuid::new_v4().to_string();
        tx.send(BackendInboundMsg::Request(AudioGenerationRequest {
            id: id.clone(),
            prompt: "".to_string(),
            secs: 4,
        }))?;

        tokio::time::sleep(Duration::from_millis(150)).await;
        tx.send(BackendInboundMsg::Abort(id.clone()))?;

        assert_eq!(rx.recv()?.unwrap_start().id, id);
        assert_eq!(rx.recv()?.unwrap_progress().1, 0.25);
        assert_eq!(rx.recv()?.unwrap_progress().1, 0.5);
        assert_eq!(rx.recv()?.unwrap_err().1, "Aborted");

        let id = Uuid::new_v4().to_string();
        tx.send(BackendInboundMsg::Request(AudioGenerationRequest {
            id: id.clone(),
            prompt: "".to_string(),
            secs: 4,
        }))?;

        assert_eq!(rx.recv()?.unwrap_start().id, id);
        assert_eq!(rx.recv()?.unwrap_progress().1, 0.25);
        assert_eq!(rx.recv()?.unwrap_progress().1, 0.5);
        assert_eq!(rx.recv()?.unwrap_progress().1, 0.75);
        assert_eq!(rx.recv()?.unwrap_progress().1, 1.0);

        Ok(())
    }
}
