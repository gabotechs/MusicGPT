use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::time::Duration;
use axum::extract::ws::{CloseFrame, Message, WebSocket};
use futures_util::{SinkExt, StreamExt};
use tokio::sync::Mutex;
use crate::ui::backend_ai::{BackendAiInboundMsg, BackendAiOutboundMsg};

pub async fn ws_handler(
    mut ws: WebSocket,
    ai_tx: Sender<BackendAiInboundMsg>,
    ai_rx: Arc<Mutex<Receiver<BackendAiOutboundMsg>>>,
) {
    {
        let lock = ai_rx.try_lock();
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
    let (mut tx, mut rx) = ws.split();
    let end_rx = Arc::new(AtomicBool::default());
    let end_tx = end_rx.clone();

    let ai_rx = ai_rx.clone();
    tokio::spawn(async move {
        let ai_rx = ai_rx.lock().await;
        loop {
            if let Ok(msg) = ai_rx.try_recv() {
                let msg = serde_json::to_string(&msg).expect("Could not serialize msg");
                let _ = tx.send(Message::Text(msg)).await;
            } else if end_rx.load(Ordering::SeqCst) {
                return Ok::<(), anyhow::Error>(());
            } else {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
    });

    let ai_tx = ai_tx.clone();
    tokio::spawn(async move {
        let res = async move {
            while let Some(msg) = rx.next().await {
                let msg = msg?;
                match msg {
                    Message::Text(text) => {
                        ai_tx.send(serde_json::from_str(&text)?)?;
                    }
                    Message::Binary(bin) => {
                        ai_tx.send(serde_json::from_slice(&bin)?)?;
                    }
                    Message::Close(_) => break,
                    Message::Ping(_) => {}
                    Message::Pong(_) => {}
                }
            }
            Ok::<(), anyhow::Error>(())
        }
            .await;
        end_tx.store(true, Ordering::SeqCst);
        res
    });
}

