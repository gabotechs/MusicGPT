use crate::storage::Storage;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use specta::Type;
use std::time::SystemTime;
use uuid::Uuid;

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub struct UserChatEntry {
    pub id: Uuid,
    pub chat_id: Uuid,
    pub text: String,
}

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub struct AiChatEntry {
    pub id: Uuid,
    pub chat_id: Uuid,
    pub relpath: String,
    pub error: String,
}

#[derive(Clone, Debug, Type, Serialize, Deserialize)]
pub enum ChatEntry {
    User(UserChatEntry),
    Ai(AiChatEntry),
}

impl ChatEntry {
    pub fn new_ai_success(chat_id: Uuid, id: Uuid, relpath: String) -> Self {
        Self::Ai(AiChatEntry {
            id,
            chat_id,
            relpath,
            error: "".to_string(),
        })
    }

    pub fn new_ai_err(chat_id: Uuid, id: Uuid, error: String) -> Self {
        Self::Ai(AiChatEntry {
            id,
            chat_id,
            relpath: "".to_string(),
            error,
        })
    }

    pub fn new_user(chat_id: Uuid, id: Uuid, text: String) -> Self {
        Self::User(UserChatEntry { id, chat_id, text })
    }

    pub async fn save<S: Storage>(&self, storage: &S) -> anyhow::Result<()> {
        let (chat_id, id, is_ai) = match self {
            ChatEntry::User(v) => (v.chat_id, v.id, 0),
            ChatEntry::Ai(v) => (v.chat_id, v.id, 1),
        };
        let now: DateTime<Utc> = SystemTime::now().into();
        let path = format!("chats/{chat_id}/{now}_{id}_{is_ai}.json");
        Ok(storage.write(&path, serde_json::to_vec(self)?).await?)
    }

    pub async fn load_from_chat<S: Storage>(
        storage: &S,
        chat_id: Uuid,
    ) -> anyhow::Result<Vec<Self>> {
        let mut result = vec![];
        for file in storage.list(&format!("chats/{chat_id}")).await? {
            if let Ok(Some(content)) = storage.read(&file).await {
                match serde_json::from_slice::<Self>(&content) {
                    Ok(entry) => result.push(entry),
                    Err(_err) => { /* do something? */ }
                };
            }
        }

        Ok(result)
    }
}
