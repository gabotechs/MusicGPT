use crate::storage::Storage;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use specta::Type;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

#[derive(Clone, Debug, Type, Serialize, Deserialize, PartialEq)]
pub struct UserChatEntry {
    pub id: Uuid,
    pub chat_id: Uuid,
    pub text: String,
}

#[derive(Clone, Debug, Type, Serialize, Deserialize, PartialEq)]
pub struct AiChatEntry {
    pub id: Uuid,
    pub chat_id: Uuid,
    pub relpath: String,
    pub error: String,
}

#[derive(Clone, Debug, Type, Serialize, Deserialize, PartialEq)]
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
}

#[derive(Clone, Debug, Type, Serialize, Deserialize, PartialEq)]
pub struct Chat {
    pub chat_id: Uuid,
    pub name: String,
    pub created_at: u128,
}

const METADATA_FILE: &str = ".metadata.json";

impl Chat {
    pub async fn load<S: Storage>(storage: &S, chat_id: Uuid) -> anyhow::Result<Self> {
        let metadata_file = format!("chats/{chat_id}/{METADATA_FILE}");

        if let Some(this_serial) = storage.read(&metadata_file).await? {
            if let Ok(this) = serde_json::de::from_slice(&this_serial) {
                return Ok(this);
            }
        }

        let this = Self {
            chat_id,
            name: "".to_string(),
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis(),
        };
        let this_serial = serde_json::to_string(&this)?;
        storage.write(&metadata_file, this_serial).await?;
        Ok(this)
    }

    pub async fn load_all<S: Storage>(storage: &S) -> anyhow::Result<Vec<Self>> {
        let mut result = vec![];

        for dir in storage.list("chats").await? {
            let chat_id = dir.strip_prefix("chats/").unwrap_or(&dir);
            let chat_id = match Uuid::parse_str(chat_id) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let chat = match Chat::load(storage, chat_id).await {
                Ok(v) => v,
                Err(_) => continue,
            };
            result.push(chat)
        }
        result.sort_by_key(|v| v.created_at);
        result.reverse();

        Ok(result)
    }

    pub async fn save<S: Storage>(&self, storage: &S) -> anyhow::Result<()> {
        let this_serial = serde_json::to_string(self)?;
        storage
            .write(
                &format!("chats/{}/{METADATA_FILE}", self.chat_id),
                this_serial,
            )
            .await?;
        Ok(())
    }

    pub async fn update_metadata<S: Storage>(
        &mut self,
        storage: &S,
        name: Option<String>,
    ) -> anyhow::Result<()> {
        if let Some(name) = name {
            self.name = name;
        }
        let this_serial = serde_json::to_string(self)?;
        storage
            .write(
                &format!("chats/{}/{METADATA_FILE}", self.chat_id),
                this_serial,
            )
            .await?;
        Ok(())
    }

    pub async fn load_entries<S: Storage>(
        storage: &S,
        chat_id: Uuid,
    ) -> anyhow::Result<Vec<ChatEntry>> {
        let mut result = vec![];
        for file in storage.list(&format!("chats/{chat_id}")).await? {
            if file.ends_with(METADATA_FILE) {
                continue;
            }
            if let Ok(Some(content)) = storage.read(&file).await {
                match serde_json::from_slice::<ChatEntry>(&content) {
                    Ok(entry) => result.push(entry),
                    Err(_err) => { /* do something? */ }
                };
            }
        }
        Ok(result)
    }

    pub async fn delete<S: Storage>(self, storage: &S) -> anyhow::Result<()> {
        storage.rm_rf(&format!("chats/{}", self.chat_id)).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::music_gpt_chat::{Chat, ChatEntry};
    use crate::storage::AppFs;
    use std::time::Duration;
    use uuid::Uuid;

    #[tokio::test]
    async fn loads_a_chat_even_if_it_does_not_exist() -> anyhow::Result<()> {
        let storage = AppFs::new_tmp();
        let chat_id = Uuid::new_v4();
        let chat = Chat::load(&storage, chat_id).await?;
        assert_eq!(chat.chat_id, chat_id);
        Ok(())
    }

    #[tokio::test]
    async fn updates_a_chat_that_does_not_exist_creates_it() -> anyhow::Result<()> {
        let storage = AppFs::new_tmp();
        let chat_id = Uuid::new_v4();
        let mut chat = Chat::load(&storage, chat_id).await?;
        chat.update_metadata(&storage, Some("foo".to_string()))
            .await?;

        let chat = Chat::load(&storage, chat_id).await?;
        assert_eq!(chat.name, "foo");
        Ok(())
    }

    #[tokio::test]
    async fn creates_multiple_chats_and_lists_them() -> anyhow::Result<()> {
        let storage = AppFs::new_tmp();
        let mut chats = vec![];
        for _ in 0..5 {
            let chat_id = Uuid::new_v4();
            let chat = Chat::load(&storage, chat_id).await?;
            tokio::time::sleep(Duration::from_millis(1)).await;
            chats.push(chat)
        }
        chats.reverse();

        let all = Chat::load_all(&storage).await?;
        assert_eq!(all, chats);

        Ok(())
    }

    #[tokio::test]
    async fn list_messages_in_non_existing_chat_returns_empty_vec() -> anyhow::Result<()> {
        let storage = AppFs::new_tmp();
        let chat_id = Uuid::new_v4();
        let history = Chat::load_entries(&storage, chat_id).await?;
        assert_eq!(history, vec![]);
        Ok(())
    }

    #[tokio::test]
    async fn list_messages_in_chat() -> anyhow::Result<()> {
        let storage = AppFs::new_tmp();
        let chat_id = Uuid::new_v4();

        let msg1 = ChatEntry::new_user(chat_id, Uuid::new_v4(), "u1".to_string());
        msg1.save(&storage).await?;
        let msg2 = ChatEntry::new_ai_success(chat_id, Uuid::new_v4(), "a1".to_string());
        msg2.save(&storage).await?;
        let msg3 = ChatEntry::new_ai_success(Uuid::new_v4(), Uuid::new_v4(), "BAD".to_string());
        msg3.save(&storage).await?;
        let msg4 = ChatEntry::new_user(chat_id, Uuid::new_v4(), "u2".to_string());
        msg4.save(&storage).await?;
        let msg5 = ChatEntry::new_user(Uuid::new_v4(), Uuid::new_v4(), "BAD".to_string());
        msg5.save(&storage).await?;
        let msg6 = ChatEntry::new_ai_err(chat_id, Uuid::new_v4(), "a2".to_string());
        msg6.save(&storage).await?;

        let history = Chat::load_entries(&storage, chat_id).await?;
        assert_eq!(history, vec![msg1, msg2, msg4, msg6]);

        Ok(())
    }

    #[tokio::test]
    async fn deletes_chat() -> anyhow::Result<()> {
        let storage = AppFs::new_tmp();
        let chat_id = Uuid::new_v4();

        let msg1 = ChatEntry::new_user(chat_id, Uuid::new_v4(), "u1".to_string());
        msg1.save(&storage).await?;
        let msg2 = ChatEntry::new_ai_success(chat_id, Uuid::new_v4(), "a1".to_string());
        msg2.save(&storage).await?;

        let history = Chat::load_entries(&storage, chat_id).await?;
        assert_eq!(history, vec![msg1, msg2]);

        let chat = Chat::load(&storage, chat_id).await?;
        chat.delete(&storage).await?;

        let history = Chat::load_entries(&storage, chat_id).await?;
        assert_eq!(history, vec![]);

        Ok(())
    }
}
