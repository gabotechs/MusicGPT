use std::fmt::{Display, Formatter};
use std::sync::mpsc::Sender;
use async_trait::async_trait;
use futures_util::StreamExt;
use serde::de::Error;
use serde::{Deserialize, Serialize};
use specta::Type;
use uuid::Uuid;
use crate::storage::Storage;
use crate::backend::audio_generation_fanout::GenerationMessage;
use crate::backend::audio_generation_backend::{AudioGenerationRequest, BackendInboundMsg};
use crate::backend::music_gpt_chat_entry::ChatEntry;
use crate::backend::ws_handler::{WsHandler};

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub struct RetrieveHistory {
    pub chat_id: Uuid,
}

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub struct GenerateAudio {
    pub id: Uuid,
    pub chat_id: Uuid,
    pub prompt: String,
    pub secs: usize,
}

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub struct AbortGeneration {
    pub id: Uuid,
    pub chat_id: Uuid,
}

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub struct Info {
    pub model: String,
    pub device: String
}

// === Inbound ===

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub enum InboundMsg {
    GenerateAudio(GenerateAudio),
    AbortGeneration(AbortGeneration),
    RetrieveHistory(RetrieveHistory)
}

// === Outbound ===

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub enum OutboundMsg {
    Generation(GenerationMessage),
    Init(Info),
    ChatHistory((Uuid, Vec<ChatEntry>))
}


#[derive(Clone)]
pub struct MusicGptWsHandler<S: Storage> {
    pub storage: S,
    pub ai_broadcast_tx: tokio::sync::broadcast::Sender<GenerationMessage>,
    pub ai_tx: Sender<BackendInboundMsg>,
    pub info: Info,
}

#[async_trait]
impl<S: Storage + 'static> WsHandler for MusicGptWsHandler<S> {
    type Inbound = InboundMsg;
    type Outbound = OutboundMsg;

    async fn handle_init(&self) -> Vec<OutboundMsg> {
        vec![OutboundMsg::Init(self.info.clone())]
    }

    async fn handle_inbound_msg(&self, msg: InboundMsg) -> Option<OutboundMsg> {
        match msg {
            InboundMsg::GenerateAudio(req) => {
                let _ = self
                    .ai_tx
                    .send(BackendInboundMsg::Request(AudioGenerationRequest {
                        id: IdPair(req.chat_id, req.id).to_string(),
                        prompt: req.prompt.clone(),
                        secs: req.secs,
                    }));
                None
            }
            InboundMsg::AbortGeneration(req) => {
                let id = IdPair(req.chat_id, req.id).to_string();
                let _ = self.ai_tx.send(BackendInboundMsg::Abort(id));
                None
            }
            InboundMsg::RetrieveHistory(req) => {
                let history = ChatEntry::load_from_chat(&self.storage, req.chat_id)
                    .await
                    .unwrap_or_default();
                Some(OutboundMsg::ChatHistory((req.chat_id, history)))
            }
        }
    }

    fn handle_subscription(&self) -> impl StreamExt<Item = OutboundMsg> + Send + 'static {
        let mut rx = self.ai_broadcast_tx.subscribe();
        async_stream::stream! {
            while let Ok(msg) = rx.recv().await {
                yield OutboundMsg::Generation(msg)
            }
        }
    }

    async fn handle_error(&self, _: impl Error + Send) -> Option<OutboundMsg> {
        None
    }
}


#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct IdPair(pub Uuid, pub Uuid);

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
