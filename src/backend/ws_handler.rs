use std::fmt::Display;
use std::sync::Arc;

use async_trait::async_trait;
use axum::extract::ws::{Message, WebSocket};
use futures_util::{pin_mut, SinkExt, StreamExt};
use serde::de::DeserializeOwned;
use serde::Serialize;
use tokio::sync::Mutex;

#[async_trait]
pub trait WsHandler: Sized {
    type Inbound: DeserializeOwned + Send + Sync;
    type Outbound: Serialize + Send + Sync;

    async fn handle_init(&self) -> Vec<Self::Outbound>;
    async fn handle_inbound_msg(&self, msg: Self::Inbound) -> Option<Self::Outbound>;
    fn handle_subscription(&self) -> impl StreamExt<Item = Self::Outbound> + Send + 'static;
    async fn handle_error(&self, _: impl Display + Send) -> Option<Self::Outbound>;

    async fn handle(self, ws: WebSocket) {
        let (tx, mut rx) = ws.split();
        let tx = Arc::new(Mutex::new(tx));

        // Initialization messages.
        {
            let mut tx = tx.lock().await;
            for msg in self.handle_init().await {
                let msg = serde_json::to_string(&msg).expect("Could not serialize msg");
                let _ = tx.send(Message::Text(msg)).await;
            }
            // <- drop tx
        }

        // Subscriptions messages.
        let tx_clone = tx.clone();
        let subscription = self.handle_subscription();
        let task = tokio::spawn(async move {
            pin_mut!(subscription);
            while let Some(msg) = subscription.next().await {
                let msg = serde_json::to_string(&msg).expect("Could not serialize msg");
                let _ = tx_clone.lock().await.send(Message::Text(msg)).await;
            }
        });

        // Inbound messages.
        while let Some(Ok(msg)) = rx.next().await {
            let msg = match msg {
                Message::Text(text) => serde_json::from_str(&text),
                Message::Binary(bin) => serde_json::from_slice(&bin),
                Message::Close(_) => break,
                _ => continue,
            };
            let maybe_response = match msg {
                Ok(msg) => self.handle_inbound_msg(msg).await,
                Err(err) => self.handle_error(err).await,
            };
            if let Some(response) = maybe_response {
                let mut tx = tx.lock().await;
                let msg = serde_json::to_string(&response).expect("Could not serialize msg");
                let _ = tx.send(Message::Text(msg)).await;
                // <- drop tx
            }
        }
        // TODO: use a cancellation token?
        task.abort()
    }
}
