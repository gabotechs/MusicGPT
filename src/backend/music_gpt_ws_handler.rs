use std::fmt::{Display, Formatter};
use std::sync::mpsc::Sender;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use specta::Type;
use tracing::info;
use uuid::Uuid;

use crate::backend::audio_generation_backend::{AudioGenerationRequest, BackendInboundMsg};
use crate::backend::audio_generation_fanout::GenerationMessage;
use crate::backend::music_gpt_chat::{Chat, ChatEntry};
use crate::backend::ws_handler::WsHandler;
use crate::storage::Storage;

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub struct ChatRequest {
    pub chat_id: Uuid,
}

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub struct GenerateAudioRequest {
    pub id: Uuid,
    pub chat_id: Uuid,
    pub prompt: String,
    pub secs: usize,
}

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub struct AbortGenerationRequest {
    pub id: Uuid,
    pub chat_id: Uuid,
}

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub struct Info {
    pub model: String,
    pub device: String,
}

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub struct SetChatMetadataRequest {
    pub chat_id: Uuid,
    pub name: Option<String>,
}

// === Inbound ===

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub enum InboundMsg {
    GenerateAudioNewChat(GenerateAudioRequest),
    GenerateAudio(GenerateAudioRequest),
    AbortGeneration(AbortGenerationRequest),
    GetChat(ChatRequest),
    SetChatMetadata(SetChatMetadataRequest),
    DelChat(ChatRequest),
}

// === Outbound ===

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub enum OutboundMsg {
    Generation(GenerationMessage),
    Info(Info),
    Chat((Chat, Vec<ChatEntry>)),
    Chats(Vec<Chat>),
    Error(String),
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
        let chats = Chat::load_all(&self.storage).await.unwrap_or_default();
        vec![
            OutboundMsg::Info(self.info.clone()),
            OutboundMsg::Chats(chats),
        ]
    }

    async fn handle_inbound_msg(&self, msg: InboundMsg) -> Option<OutboundMsg> {
        async move {
            let res = match msg {
                InboundMsg::GenerateAudioNewChat(req) => {
                    info!("Generating audio for new chat");
                    let chat = Chat {
                        chat_id: req.chat_id,
                        name: req.prompt.clone(),
                        created_at: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_millis(),
                    };
                    chat.save(&self.storage).await?;
                    self.ai_tx
                        .send(BackendInboundMsg::Request(AudioGenerationRequest {
                            id: IdPair(req.chat_id, req.id).to_string(),
                            prompt: req.prompt.clone(),
                            secs: req.secs,
                        }))?;
                    let chats = Chat::load_all(&self.storage).await?;
                    Some(OutboundMsg::Chats(chats))
                }
                InboundMsg::GenerateAudio(req) => {
                    info!("Generating audio for existing chat");
                    self.ai_tx
                        .send(BackendInboundMsg::Request(AudioGenerationRequest {
                            id: IdPair(req.chat_id, req.id).to_string(),
                            prompt: req.prompt.clone(),
                            secs: req.secs,
                        }))?;
                    None
                }
                InboundMsg::AbortGeneration(req) => {
                    info!("Aborting audio generation");
                    let id = IdPair(req.chat_id, req.id).to_string();
                    self.ai_tx.send(BackendInboundMsg::Abort(id))?;
                    None
                }
                InboundMsg::GetChat(req) => {
                    let chat = Chat::load(&self.storage, req.chat_id).await?;
                    let history = Chat::load_entries(&self.storage, req.chat_id).await?;
                    Some(OutboundMsg::Chat((chat, history)))
                }
                InboundMsg::SetChatMetadata(req) => {
                    info!("Modifying the chat's metadata");
                    let mut chat = Chat::load(&self.storage, req.chat_id).await?;
                    chat.update_metadata(&self.storage, req.name).await?;
                    None
                }
                InboundMsg::DelChat(req) => {
                    info!("Deleting chat");
                    let chat = Chat::load(&self.storage, req.chat_id).await?;
                    chat.delete(&self.storage).await?;
                    let chats = Chat::load_all(&self.storage).await?;
                    Some(OutboundMsg::Chats(chats))
                }
            };
            Ok::<Option<OutboundMsg>, anyhow::Error>(res)
        }
        .await
        .unwrap_or_else(|err| Some(OutboundMsg::Error(err.to_string())))
    }

    fn handle_subscription(&self) -> impl StreamExt<Item = OutboundMsg> + Send + 'static {
        let mut rx = self.ai_broadcast_tx.subscribe();
        async_stream::stream! {
            while let Ok(msg) = rx.recv().await {
                yield OutboundMsg::Generation(msg)
            }
        }
    }

    async fn handle_error(&self, err: impl Display + Send) -> Option<OutboundMsg> {
        Some(OutboundMsg::Error(err.to_string()))
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
