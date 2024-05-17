use crate::storage::Storage;
use crate::ui::messages::HistoryEntry;
use chrono::{DateTime, Utc};
use std::time::SystemTime;
use uuid::Uuid;

impl HistoryEntry {
    pub async fn save<S: Storage>(&self, storage: &S) -> anyhow::Result<()> {
        let (chat_id, id, is_ai) = match self {
            HistoryEntry::User(v) => (v.chat_id, v.id, 0),
            HistoryEntry::Ai(v) => (v.chat_id, v.id, 1),
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
            if let Some(content) = storage.read(&file).await? {
                result.push(serde_json::from_slice(&content)?)
            }
        }

        Ok(result)
    }
}
