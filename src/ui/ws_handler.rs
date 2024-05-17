use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::{CloseFrame, Message, WebSocket};
use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use scopeguard::defer;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::audio_manager::AudioManager;
use crate::storage::Storage;
use crate::ui::backend_ai::{AudioGenerationRequest, BackendAiInboundMsg, BackendAiOutboundMsg};
use crate::ui::messages::{
    AiHistoryEntry, AudioGenerationError, AudioGenerationProgress, AudioGenerationResult,
    GenerateAudio, HistoryEntry, InboundMsg, Init, OutboundMsg, UserHistoryEntry,
};

#[derive(Clone)]
pub struct WsHandler<S: Storage> {
    pub storage: S,
    pub ai_rx: Arc<Mutex<Receiver<BackendAiOutboundMsg>>>,
    pub ai_tx: Sender<BackendAiInboundMsg>,
    pub info: Init,
}

impl<S: Storage + 'static> WsHandler<S> {
    async fn ai_reception_loop(
        self,
        tx: Arc<Mutex<SplitSink<WebSocket, Message>>>,
        end: Arc<AtomicBool>,
    ) {
        let ai_rx = self.ai_rx.lock().await;
        let audio_manager = AudioManager::default();
        loop {
            // keep polling the ai channel until there's a message.
            let Ok(ai_msg) = ai_rx.try_recv() else {
                if end.load(Ordering::SeqCst) {
                    return;
                } else {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    continue;
                }
            };
            // Map the message to something sendable to the WS connection.
            let outbound_msg = match ai_msg {
                BackendAiOutboundMsg::Response((id, queue)) => {
                    let IdPair((chat_id, id)) = id.into();
                    let relpath = format!("audios/{}.wav", id);
                    let f = || async {
                        let bytes = audio_manager.to_wav(queue)?;
                        self.storage.write(&relpath, bytes).await?;
                        let entry = HistoryEntry::Ai(AiHistoryEntry {
                            id,
                            chat_id,
                            relpath: relpath.clone(),
                        });
                        entry.save(&self.storage).await?;
                        Ok::<(), anyhow::Error>(())
                    };
                    if let Err(err) = f().await {
                        OutboundMsg::Error(AudioGenerationError {
                            id,
                            chat_id,
                            error: err.to_string(),
                        })
                    } else {
                        OutboundMsg::Result(AudioGenerationResult {
                            id,
                            chat_id,
                            relpath,
                        })
                    }
                }
                BackendAiOutboundMsg::Failure((id, error)) => {
                    let IdPair((chat_id, id)) = id.into();
                    OutboundMsg::Error(AudioGenerationError { id, chat_id, error })
                }
                BackendAiOutboundMsg::Progress((id, progress)) => {
                    let IdPair((chat_id, id)) = id.into();
                    OutboundMsg::Progress(AudioGenerationProgress {
                        id,
                        chat_id,
                        progress,
                    })
                }
            };
            // Send it to the ws connection.
            {
                let mut tx = tx.lock().await;
                let _ = tx.send(outbound_msg.to_msg()).await;
            }
        }
    }

    async fn ws_reception_loop(
        self,
        tx: Arc<Mutex<SplitSink<WebSocket, Message>>>,
        mut rx: SplitStream<WebSocket>,
        end: Arc<AtomicBool>,
    ) -> anyhow::Result<()> {
        defer! {
            // When this function finishes, any other asynchronous task should also end.
            end.store(true, Ordering::SeqCst)
        }

        {
            // Send some initialization messages
            let mut tx = tx.lock().await;
            let _ = tx.send(OutboundMsg::Init(self.info.clone()).to_msg()).await;
        }

        while let Some(msg_or_err) = rx.next().await {
            let msg = msg_or_err?;
            let msg = match msg {
                Message::Text(_) | Message::Binary(_) => InboundMsg::from_msg(msg)?,
                Message::Close(_) => break,
                _ => continue,
            };
            match msg {
                InboundMsg::GenerateAudio(req) => {
                    // An audio generation was requested this will:
                    // - store a new entry in the chat history
                    // - send the audio generation request to the AI
                    let entry: HistoryEntry = req.clone().into();
                    let _ = self.ai_tx.send(req.into());
                    let _ = entry.save(&self.storage).await;
                }
                InboundMsg::AbortGeneration(req) => {
                    let id = IdPair((req.chat_id, req.id)).to_string();
                    let _ = self.ai_tx.send(BackendAiInboundMsg::Abort(id));
                }
                InboundMsg::RetrieveHistory(req) => {
                    let history = HistoryEntry::load_from_chat(&self.storage, req.chat_id)
                        .await
                        .unwrap_or_default();
                    let msg = OutboundMsg::History((req.chat_id, history));
                    {
                        let mut tx = tx.lock().await;
                        let _ = tx.send(msg.to_msg()).await;
                    }
                }
            }
        }
        Ok(())
    }

    pub async fn handle(self, mut ws: WebSocket) {
        {
            let lock = self.ai_rx.try_lock();
            if lock.is_err() {
                let _ = ws
                    .send(Message::Close(Some(CloseFrame {
                        // https://www.rfc-editor.org/rfc/rfc6455.html#section-7.4.1
                        code: 1000,
                        reason: "Already in use".into(),
                    })))
                    .await;
                return;
            }
        };
        let (tx, rx) = ws.split();
        let tx = Arc::new(Mutex::new(tx));
        let end = Arc::new(AtomicBool::default());

        tokio::spawn(self.clone().ai_reception_loop(tx.clone(), end.clone()));
        tokio::spawn(self.ws_reception_loop(tx, rx, end));
    }
}

impl From<GenerateAudio> for BackendAiInboundMsg {
    fn from(value: GenerateAudio) -> Self {
        BackendAiInboundMsg::Request(AudioGenerationRequest {
            id: IdPair((value.chat_id, value.id)).to_string(),
            prompt: value.prompt,
            secs: value.secs,
        })
    }
}

impl From<GenerateAudio> for HistoryEntry {
    fn from(value: GenerateAudio) -> Self {
        HistoryEntry::User(UserHistoryEntry {
            id: value.id,
            chat_id: value.chat_id,
            text: value.prompt,
        })
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct IdPair((Uuid, Uuid));

impl Display for IdPair {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let serialized = serde_json::to_string(self).expect("Could not serialize IdPair");
        write!(f, "{serialized}")
    }
}

impl From<String> for IdPair {
    fn from(value: String) -> Self {
        serde_json::from_str(value.as_str()).expect("Could not deserialize IdPair")
    }
}
