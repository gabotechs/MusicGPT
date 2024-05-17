use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::{CloseFrame, Message, WebSocket};
use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::audio_manager::AudioManager;
use crate::storage::Storage;
use crate::ui::backend_ai::{AudioGenerationRequest, BackendAiInboundMsg, BackendAiOutboundMsg};
use crate::ui::messages::{
    AiHistoryEntry, AudioGenerationError, AudioGenerationProgress, AudioGenerationResult,
    HistoryEntry, InboundMsg, Init, OutboundMsg, UserHistoryEntry,
};

#[derive(Clone)]
pub struct WsHandler<S: Storage> {
    pub storage: S,
    pub ai_rx: Arc<Mutex<Receiver<BackendAiOutboundMsg>>>,
    pub ai_tx: Sender<BackendAiInboundMsg>,
    pub info: Init,
    pub end: Arc<AtomicBool>,
}

impl<S: Storage + 'static> WsHandler<S> {
    async fn ai_reception_loop(self, tx: Arc<Mutex<SplitSink<WebSocket, Message>>>) {
        let ai_rx = self.ai_rx.lock().await;
        let audio_manager = AudioManager::default();
        loop {
            // keep polling the ai channel until there's a message.
            let ai_msg = if let Ok(ai_msg) = ai_rx.try_recv() {
                ai_msg
            } else if self.end.load(Ordering::SeqCst) {
                return;
            } else {
                tokio::time::sleep(Duration::from_millis(10)).await;
                continue;
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
                let _ = tx.lock().await.send(outbound_msg.to_msg()).await;
            }
        }
    }

    async fn ws_reception_loop(
        self,
        tx: Arc<Mutex<SplitSink<WebSocket, Message>>>,
        mut rx: SplitStream<WebSocket>,
    ) -> anyhow::Result<()> {
        while let Some(msg_or_err) = rx.next().await {
            let msg = msg_or_err?;
            let msg = match msg {
                Message::Text(_) | Message::Binary(_) => InboundMsg::from_msg(msg)?,
                Message::Close(_) => break,
                Message::Ping(_) => continue,
                Message::Pong(_) => continue,
            };
            match msg {
                InboundMsg::GenerateAudio(req) => {
                    let _ = self
                        .ai_tx
                        .send(BackendAiInboundMsg::Request(AudioGenerationRequest {
                            id: IdPair((req.chat_id, req.id)).to_string(),
                            prompt: req.prompt.clone(),
                            secs: req.secs,
                        }));
                    let entry = HistoryEntry::User(UserHistoryEntry {
                        id: req.id,
                        chat_id: req.chat_id,
                        text: req.prompt,
                    });
                    let _ = entry.save(&self.storage).await;
                }
                InboundMsg::AbortGeneration(req) => {
                    let _ = self.ai_tx.send(BackendAiInboundMsg::Abort(
                        IdPair((req.chat_id, req.id)).to_string(),
                    ));
                }
                InboundMsg::RetrieveHistory(req) => {
                    let history = HistoryEntry::load_from_chat(&self.storage, req.chat_id)
                        .await
                        .unwrap_or_default();
                    let msg = OutboundMsg::History((req.chat_id, history));
                    {
                        let _ = tx.lock().await.send(msg.to_msg()).await;
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
        let (mut tx, rx) = ws.split();
        let _ = tx.send(OutboundMsg::Init(self.info.clone()).to_msg()).await;
        let tx = Arc::new(Mutex::new(tx));

        tokio::spawn(self.clone().ai_reception_loop(tx.clone()));
        tokio::spawn(self.ws_reception_loop(tx, rx));
    }
}

impl<S: Storage> Drop for WsHandler<S> {
    fn drop(&mut self) {
        self.end.store(true, Ordering::SeqCst)
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
