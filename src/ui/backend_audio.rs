use crate::audio_manager::{AudioManager, AudioStream};
use std::path::PathBuf;
use tokio::sync::mpsc::{channel, Receiver, Sender};

pub enum BackendAudioInboundMsg {
    Play(PathBuf),
    Stop,
    TearDown,
}

pub enum BackendAudioOutboundMsg {
    Err(String),
}

#[derive(Default)]
pub struct BackendAudio {
    audio_manager: AudioManager,
    curr_audio_stream: Option<AudioStream>,
}

impl BackendAudio {
    pub fn run(
        mut self,
    ) -> (
        Sender<BackendAudioInboundMsg>,
        Receiver<BackendAudioOutboundMsg>,
    ) {
        let (inbound_tx, mut inbound_rx) = channel::<BackendAudioInboundMsg>(100);
        let (outbound_tx, outbound_rx) = channel::<BackendAudioOutboundMsg>(100);

        tokio::spawn(async move {
            while let Some(msg) = inbound_rx.recv().await {
                match msg {
                    BackendAudioInboundMsg::Play(path) => {
                        self.curr_audio_stream = None;
                        match self.audio_manager.play_from_wav(path) {
                            Ok(stream) => self.curr_audio_stream = Some(stream),
                            Err(err) => {
                                let _ = outbound_tx
                                    .send(BackendAudioOutboundMsg::Err(err.to_string()))
                                    .await;
                            }
                        }
                    }
                    BackendAudioInboundMsg::Stop => self.curr_audio_stream = None,
                    BackendAudioInboundMsg::TearDown => {
                        self.curr_audio_stream = None;
                        return
                    },
                }
            }
        });

        (inbound_tx, outbound_rx)
    }
}


#[cfg(test)]
mod tests {
    use std::time::Duration;
    use super::*;

    #[ignore]
    #[tokio::test]
    async fn test_audio_stop_works() -> anyhow::Result<()> {
        let backend_audio = BackendAudio::default();
        
        let (tx, _rx)  = backend_audio.run();
        tx.send(BackendAudioInboundMsg::Play("assets/test.wav".into())).await?;
        tokio::time::sleep(Duration::from_secs(1)).await;
        tx.send(BackendAudioInboundMsg::Stop).await?;
        tokio::time::sleep(Duration::from_secs(1)).await;
        Ok(())
    }

    #[ignore]
    #[tokio::test]
    async fn test_audio_play_twice_works() -> anyhow::Result<()> {
        let backend_audio = BackendAudio::default();

        let (tx, _rx)  = backend_audio.run();
        tx.send(BackendAudioInboundMsg::Play("assets/test.wav".into())).await?;
        tokio::time::sleep(Duration::from_secs(1)).await;
        tx.send(BackendAudioInboundMsg::Play("assets/test.wav".into())).await?;
        tokio::time::sleep(Duration::from_secs(1)).await;
        Ok(())
    }
}