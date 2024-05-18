use serde::{Deserialize, Serialize};
use specta::Type;
use uuid::Uuid;

use crate::audio_manager::AudioManager;
use crate::backend::audio_generation_backend::BackendOutboundMsg;
use crate::backend::music_gpt_chat::ChatEntry;
use crate::backend::music_gpt_ws_handler::IdPair;
use crate::storage::Storage;

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub struct AudioGenerationStart {
    pub id: Uuid,
    pub chat_id: Uuid,
    pub prompt: String,
    pub secs: usize,
}

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub struct AudioGenerationProgress {
    pub id: Uuid,
    pub chat_id: Uuid,
    pub progress: f32,
}

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub struct AudioGenerationError {
    pub id: Uuid,
    pub chat_id: Uuid,
    pub error: String,
}

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub struct AudioGenerationResult {
    pub id: Uuid,
    pub chat_id: Uuid,
    pub relpath: String,
}

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub enum GenerationMessage {
    Start(AudioGenerationStart),
    Progress(AudioGenerationProgress),
    Error(AudioGenerationError),
    Result(AudioGenerationResult),
}

pub fn audio_generation_fanout<S: Storage + 'static>(
    ai_rx: std::sync::mpsc::Receiver<BackendOutboundMsg>,
    storage: S,
) -> tokio::sync::broadcast::Sender<GenerationMessage> {
    let (ai_broadcast_tx, _) = tokio::sync::broadcast::channel(1000); // Arbitrary number.

    let mut ai_rx = std_to_tokio_receiver(ai_rx);
    let ai_broadcast_tx_clone = ai_broadcast_tx.clone();
    let audio_manager = AudioManager::default();
    tokio::spawn(async move {
        while let Some(msg) = ai_rx.recv().await {
            let outbound_msg = match msg {
                BackendOutboundMsg::Start(msg) => {
                    let IdPair(chat_id, id) = msg.id.into();
                    let entry = ChatEntry::new_user(chat_id, id, msg.prompt.clone());
                    let _ = entry.save(&storage).await;
                    GenerationMessage::Start(AudioGenerationStart {
                        id,
                        chat_id,
                        prompt: msg.prompt,
                        secs: msg.secs,
                    })
                }
                BackendOutboundMsg::Response((id, queue)) => {
                    let IdPair(chat_id, id) = id.into();
                    let relpath = format!("audios/{}.wav", id);
                    let save_audio = || async {
                        let bytes = audio_manager.to_wav(queue)?;
                        storage.write(&relpath, bytes).await?;
                        Ok::<(), anyhow::Error>(())
                    };
                    // If audio failed to be saved, do not count as a success.
                    if let Err(err) = save_audio().await {
                        let entry = ChatEntry::new_ai_err(chat_id, id, err.to_string());
                        let _ = entry.save(&storage).await;
                        GenerationMessage::Error(AudioGenerationError {
                            id,
                            chat_id,
                            error: err.to_string(),
                        })
                    } else {
                        let entry = ChatEntry::new_ai_success(chat_id, id, relpath.clone());
                        let _ = entry.save(&storage).await;
                        GenerationMessage::Result(AudioGenerationResult {
                            id,
                            chat_id,
                            relpath,
                        })
                    }
                }
                BackendOutboundMsg::Failure((id, error)) => {
                    let IdPair(chat_id, id) = id.into();
                    let entry = ChatEntry::new_ai_err(chat_id, id, error.clone());
                    let _ = entry.save(&storage).await;
                    GenerationMessage::Error(AudioGenerationError { id, chat_id, error })
                }
                BackendOutboundMsg::Progress((id, progress)) => {
                    let IdPair(chat_id, id) = id.into();
                    GenerationMessage::Progress(AudioGenerationProgress {
                        id,
                        chat_id,
                        progress,
                    })
                }
            };
            let _ = ai_broadcast_tx.send(outbound_msg);
        }
    });

    ai_broadcast_tx_clone
}

fn std_to_tokio_receiver<T: Send + 'static>(
    std_rx: std::sync::mpsc::Receiver<T>,
) -> tokio::sync::mpsc::UnboundedReceiver<T> {
    let (tokio_tx, tokio_rx) = tokio::sync::mpsc::unbounded_channel();
    tokio::task::spawn_blocking(move || {
        for msg in std_rx {
            let _ = tokio_tx.send(msg);
        }
    });
    tokio_rx
}
