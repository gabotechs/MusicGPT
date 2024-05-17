use axum::extract::ws::Message;
use serde::{Deserialize, Serialize};
use serde::de::{DeserializeOwned, Error};
use specta::Type;
use uuid::Uuid;

// === Inbound ===

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub struct RetrieveHistory {
    pub chat_id: Uuid
}

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub struct GenerateAudio {
    pub id: Uuid,
    pub chat_id: Uuid,
    pub prompt: String,
    pub secs: usize
}

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub struct AbortGeneration {
    pub id: Uuid,
    pub chat_id: Uuid,
}

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub enum InboundMsg {
    GenerateAudio(GenerateAudio),
    AbortGeneration(AbortGeneration),
    RetrieveHistory(RetrieveHistory)
}

// === Outbound ===

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub struct Init {
    pub model: String,
    pub device: String
}

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub struct UserHistoryEntry {
    pub id: Uuid,
    pub chat_id: Uuid,
    pub text: String
}

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub struct AiHistoryEntry {
    pub id: Uuid,
    pub chat_id: Uuid,
    pub relpath: String
}

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub enum HistoryEntry {
    User(UserHistoryEntry),
    Ai(AiHistoryEntry)
}

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub struct AudioGenerationProgress {
    pub id: Uuid,
    pub chat_id: Uuid,
    pub progress: f32
}

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub struct AudioGenerationError {
    pub id: Uuid,
    pub chat_id: Uuid,
    pub error: String
}

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub struct AudioGenerationResult {
    pub id: Uuid,
    pub chat_id: Uuid,
    pub relpath: String
}

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub enum OutboundMsg {
    Progress(AudioGenerationProgress),
    Error(AudioGenerationError),
    Result(AudioGenerationResult),
    Init(Init),
    History((Uuid, Vec<HistoryEntry>))
}

// === Ser-De ===

impl OutboundMsg  {
    pub fn to_msg(&self) -> Message {
        to_msg(self)
    }
    #[allow(dead_code)]
    pub fn from_msg(msg: Message) -> serde_json::error::Result<Self> {
        from_msg(msg)
    }
}


impl InboundMsg {
    #[allow(dead_code)]
    pub fn to_msg(&self) -> Message {
        to_msg(self)
    }
    pub fn from_msg(msg: Message) -> serde_json::error::Result<Self> {
        from_msg(msg)
    }
}


pub fn to_msg<T: Serialize>(msg: &T) -> Message {
    let msg = serde_json::to_string(msg).expect("Could not serialize msg");
    Message::Text(msg)
}

fn from_msg<T: DeserializeOwned>(msg: Message) -> serde_json::error::Result<T> {
    match msg {
        Message::Text(text) => {
            serde_json::from_str(&text)
        }
        Message::Binary(bin) => {
            serde_json::from_slice(&bin)
        }
        _ => Err(serde_json::error::Error::custom(format!("Invalid message type {msg:?}")))
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use specta::ts::{BigIntExportBehavior, ExportConfiguration};


    #[ignore]
    #[test]
    fn export_bindings() -> anyhow::Result<()> {
        specta::export::ts_with_cfg(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("web/src/bindings.ts")
                .to_str()
                .unwrap(),
            &ExportConfiguration::default().bigint(BigIntExportBehavior::Number),
        )?;
        Ok(())
    }
}