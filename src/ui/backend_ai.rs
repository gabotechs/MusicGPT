use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use async_trait::async_trait;
use uuid::Uuid;

use crate::music_gen_audio_encodec::MusicGenAudioEncodec;
use crate::music_gen_decoder::MusicGenDecoder;
use crate::music_gen_text_encoder::MusicGenTextEncoder;

const INPUT_IDS_BATCH_PER_SECOND: usize = 50;

#[derive(Clone, Debug)]
pub struct AudioGenerationRequest {
    id: Uuid,
    prompt: String,
    secs: usize,
}

#[derive(Clone, Debug)]
struct Job {
    req: AudioGenerationRequest,
    abort: Arc<AtomicBool>,
}

impl Job {
    fn new(req: AudioGenerationRequest) -> Self {
        Self {
            req,
            abort: Arc::new(AtomicBool::default()),
        }
    }
}

#[derive(Clone, Debug)]
pub enum BackendAiInboundMsg {
    AudioGenerationRequest(AudioGenerationRequest),
    Abort(Uuid),
    TearDown,
}

#[derive(Debug)]
pub enum BackendAiOutboundMsg {
    AudioGenerationResponse((Uuid, ort::Result<VecDeque<f32>>)),
    AudioGenerationProgress((Uuid, f32)),
}

#[async_trait]
pub trait JobProcessor: Send + Sync {
    fn process(
        &self,
        prompt: &str,
        secs: usize,
        abort: Arc<AtomicBool>,
        on_progress: Box<dyn Fn(f32) + Sync + Send + 'static>,
    ) -> ort::Result<VecDeque<f32>>;
}

pub struct MusicGenJobProcessor {
    pub text_encoder: MusicGenTextEncoder,
    pub decoder: Box<dyn MusicGenDecoder>,
    pub audio_encodec: MusicGenAudioEncodec,
}

impl JobProcessor for MusicGenJobProcessor {
    fn process(
        &self,
        prompt: &str,
        secs: usize,
        abort: Arc<AtomicBool>,
        on_progress: Box<dyn Fn(f32) + Sync + Send + 'static>,
    ) -> ort::Result<VecDeque<f32>> {
        let max_len = secs * INPUT_IDS_BATCH_PER_SECOND;

        let (lhs, am) = self.text_encoder.encode(prompt)?;
        let token_stream = self.decoder.generate_tokens(lhs, am, abort, max_len)?;

        let mut data = VecDeque::new();
        while let Ok(tokens) = token_stream.recv() {
            data.push_back(tokens?);
            on_progress(data.len() as f32 / max_len as f32);
        }

        self.audio_encodec.encode(data)
    }
}

pub struct BackendAi {
    processor: Arc<dyn JobProcessor>,
    job_queue: Arc<RwLock<VecDeque<Job>>>,
    should_exit: Arc<AtomicBool>,
}

impl BackendAi {
    pub fn new<T: JobProcessor + 'static>(processor: T) -> Self {
        Self {
            processor: Arc::new(processor),
            job_queue: Arc::new(RwLock::new(VecDeque::new())),
            should_exit: Arc::new(AtomicBool::default()),
        }
    }

    pub fn run(self) -> (Sender<BackendAiInboundMsg>, Receiver<BackendAiOutboundMsg>) {
        let (inbound_tx, inbound_rx) = channel::<BackendAiInboundMsg>();
        let (outbound_tx, outbound_rx) = channel::<BackendAiOutboundMsg>();

        let processor = self.processor.clone();

        // Job processing loop.
        let should_exit = self.should_exit.clone();
        let job_queue = self.job_queue.clone();
        std::thread::spawn(move || loop {
            let front = {
                // Immediately drop jq so that the lock is released.
                let jq = job_queue.read().unwrap();
                jq.front().cloned()
            };
            let job = match front {
                None => {
                    if should_exit.load(Ordering::SeqCst) {
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(200));
                    continue;
                }
                Some(job) => job,
            };
            if should_exit.load(Ordering::SeqCst) {
                return;
            }

            let output_tx_clone = outbound_tx.clone();
            let cbk = move |p| {
                let msg = BackendAiOutboundMsg::AudioGenerationProgress((job.req.id, p));
                let _ = output_tx_clone.send(msg);
            };
            let result = processor.process(&job.req.prompt, job.req.secs, job.abort, Box::new(cbk));

            let _ = outbound_tx.send(BackendAiOutboundMsg::AudioGenerationResponse((
                job.req.id, result,
            )));
            job_queue.write().unwrap().pop_front();
        });

        // Communications processing loop.
        let should_exit = self.should_exit.clone();
        let job_queue = self.job_queue.clone();
        std::thread::spawn(move || {
            while let Ok(msg) = inbound_rx.recv() {
                match msg {
                    BackendAiInboundMsg::AudioGenerationRequest(req) => {
                        job_queue.write().unwrap().push_back(Job::new(req));
                    }
                    BackendAiInboundMsg::Abort(id) => {
                        let queue = job_queue.read().unwrap();
                        let mut to_remove = None;
                        for (i, job) in queue.iter().enumerate() {
                            if job.req.id == id {
                                to_remove = Some(i);
                                job.abort.store(true, Ordering::SeqCst);
                                break;
                            }
                        }
                        if let Some(to_remove) = to_remove {
                            job_queue.write().unwrap().remove(to_remove);
                        }
                    }
                    BackendAiInboundMsg::TearDown => {
                        should_exit.store(true, Ordering::SeqCst);
                        return;
                    }
                }
            }
            should_exit.store(true, Ordering::SeqCst);
        });

        (inbound_tx, outbound_rx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn processes_job() -> anyhow::Result<()> {
        let backend = BackendAi::new(DummyJobProcessor);

        let (tx, rx) = backend.run();

        let id = Uuid::new_v4();
        tx.send(BackendAiInboundMsg::AudioGenerationRequest(
            AudioGenerationRequest {
                id,
                prompt: "".to_string(),
                secs: 4,
            },
        ))?;

        assert_eq!(rx.recv()?.unwrap_progress().1, 0.25);
        assert_eq!(rx.recv()?.unwrap_progress().1, 0.5);
        assert_eq!(rx.recv()?.unwrap_progress().1, 0.75);
        assert_eq!(rx.recv()?.unwrap_progress().1, 1.0);
        assert_eq!(
            rx.recv()?.unwrap_response().1.unwrap(),
            VecDeque::from([0.0, 1.0, 2.0, 3.0])
        );

        Ok(())
    }


    #[test]
    fn handles_job_failure() -> anyhow::Result<()> {
        let backend = BackendAi::new(DummyJobProcessor);

        let (tx, rx) = backend.run();

        let id = Uuid::new_v4();
        tx.send(BackendAiInboundMsg::AudioGenerationRequest(
            AudioGenerationRequest {
                id,
                prompt: "fail at 2".to_string(),
                secs: 4,
            },
        ))?;

        assert_eq!(rx.recv()?.unwrap_progress().1, 0.25);
        assert_eq!(rx.recv()?.unwrap_progress().1, 0.5);
        assert_eq!(
            rx.recv()?.unwrap_response().1.unwrap_err().to_string(),
            "Failed at 2"
        );

        Ok(())
    }

    impl BackendAiOutboundMsg {
        fn unwrap_progress(self) -> (Uuid, f32) {
            match self {
                BackendAiOutboundMsg::AudioGenerationProgress(p) => p,
                _ => panic!("msg was not AudioGenerationProgress"),
            }
        }

        fn unwrap_response(self) -> (Uuid, ort::Result<VecDeque<f32>>) {
            match self {
                BackendAiOutboundMsg::AudioGenerationResponse(p) => p,
                _ => panic!("msg was not AudioGenerationProgress"),
            }
        }
    }

    pub struct DummyJobProcessor;

    #[async_trait]
    impl JobProcessor for DummyJobProcessor {
        fn process(
            &self,
            prompt: &str,
            secs: usize,
            _abort: Arc<AtomicBool>,
            on_progress: Box<dyn Fn(f32) + Sync + Send + 'static>,
        ) -> ort::Result<VecDeque<f32>> {
            let mut result = VecDeque::new();
            for i in 0..secs {
                if prompt == format!("fail at {i}") {
                    return Err(ort::Error::CustomError(format!("Failed at {i}").into()))
                }
                result.push_back(i as f32);
                on_progress(result.len() as f32 / secs as f32);
            }

            Ok(result)
        }
    }
}
