use axum::extract::WebSocketUpgrade;
use axum::response::Html;
use axum::routing::get;
use axum::Router;
use tower_http::services::ServeDir;
use tracing::info;

use crate::backend::audio_generation_backend::{AudioGenerationBackend, JobProcessor};
use crate::backend::audio_generation_fanout::audio_generation_fanout;
use crate::backend::music_gpt_ws_handler::{Info, MusicGptWsHandler};
use crate::backend::ws_handler::WsHandler;
use crate::storage::AppFs;

pub struct RunOptions {
    pub port: usize,
    pub auto_open: bool,
    pub expose: bool,
}

pub async fn run<T: JobProcessor + 'static>(
    storage: AppFs,
    processor: T,
    opts: RunOptions,
) -> anyhow::Result<()> {
    let model = processor.name();
    let device = processor.device();

    let (ai_tx, ai_rx) = AudioGenerationBackend::new(processor).run();
    let ai_broadcast_tx = audio_generation_fanout(ai_rx, storage.clone());

    let root_dir = storage.root.clone();
    let ws_handler = MusicGptWsHandler {
        ai_tx,
        storage,
        info: Info { model, device },
        ai_broadcast_tx,
    };

    let app = Router::new()
        .fallback(get(web_app))
        .nest_service("/files", ServeDir::new(root_dir))
        .route(
            "/ws",
            get(|ws: WebSocketUpgrade| async move {
                let ws_handler = ws_handler.clone();
                ws.on_upgrade(move |ws| ws_handler.handle(ws))
            }),
        );

    let port = opts.port;
    let host = if opts.expose { "0.0.0.0" } else { "127.0.0.1" };
    let advertised = if opts.expose {
        hostname::get()
            .unwrap_or_default()
            .to_str()
            .unwrap_or("localhost")
            .to_string()
    } else {
        "localhost".to_string()
    };
    let listener = tokio::net::TcpListener::bind(format!("{host}:{port}")).await?;
    let addr = format!("http://{advertised}:{port}");
    info!("MusicGPT running at {addr}");
    if opts.auto_open {
        let _ = open::that(addr);
    }

    Ok(axum::serve(listener, app).await?)
}

async fn web_app() -> Html<&'static str> {
    Html(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/web/dist/index.html"
    )))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU16, Ordering};
    use std::time::Duration;

    use futures_util::{SinkExt, StreamExt};
    use serde::de::DeserializeOwned;
    use serde::Serialize;
    use tokio_tungstenite::connect_async;
    use uuid::Uuid;

    use crate::backend::_test_utils::DummyJobProcessor;
    use crate::backend::music_gpt_ws_handler::{
        AbortGenerationRequest, GenerateAudioRequest, InboundMsg, OutboundMsg,
    };

    use super::*;

    static PORT: AtomicU16 = AtomicU16::new(8643);

    fn spawn<P: JobProcessor + 'static>(processor: P) -> usize {
        let app_fs = AppFs::new_tmp();
        let port = PORT.fetch_add(1, Ordering::SeqCst) as usize;
        let run_options = RunOptions {
            port,
            auto_open: false,
            expose: false,
        };
        tokio::spawn(run(app_fs, processor, run_options));
        port
    }

    #[tokio::test]
    async fn sending_a_job_processes_it() -> anyhow::Result<()> {
        let port = spawn(DummyJobProcessor::default());

        let (mut ws_stream, _) = connect_async(&format!("ws://localhost:{port}/ws")).await?;
        let id = Uuid::new_v4();
        let chat_id = Uuid::new_v4();
        let msg = InboundMsg::GenerateAudio(GenerateAudioRequest {
            id,
            chat_id,
            prompt: "Create a cool song".to_string(),
            secs: 4,
        });
        ws_stream.send(msg.to_tungstenite_msg()).await?;

        let msg = OutboundMsg::from_tungstenite_msg(ws_stream.next().await.unwrap()?)?;
        msg.unwrap_info();

        let msg = OutboundMsg::from_tungstenite_msg(ws_stream.next().await.unwrap()?)?;
        msg.unwrap_chats();

        let msg = OutboundMsg::from_tungstenite_msg(ws_stream.next().await.unwrap()?)?;
        msg.unwrap_start();

        let msg = OutboundMsg::from_tungstenite_msg(ws_stream.next().await.unwrap()?)?;
        let p = msg.unwrap_progress();
        assert_eq!(p.id, id);
        assert_eq!(p.chat_id, chat_id);
        assert_eq!(p.progress, 0.25);

        let msg = OutboundMsg::from_tungstenite_msg(ws_stream.next().await.unwrap()?)?;
        let p = msg.unwrap_progress();
        assert_eq!(p.id, id);
        assert_eq!(p.chat_id, chat_id);
        assert_eq!(p.progress, 0.5);

        let msg = OutboundMsg::from_tungstenite_msg(ws_stream.next().await.unwrap()?)?;
        let p = msg.unwrap_progress();
        assert_eq!(p.id, id);
        assert_eq!(p.chat_id, chat_id);
        assert_eq!(p.progress, 0.75);

        let msg = OutboundMsg::from_tungstenite_msg(ws_stream.next().await.unwrap()?)?;
        let p = msg.unwrap_progress();
        assert_eq!(p.id, id);
        assert_eq!(p.chat_id, chat_id);
        assert_eq!(p.progress, 1.0);

        let msg = OutboundMsg::from_tungstenite_msg(ws_stream.next().await.unwrap()?)?;
        let p = msg.unwrap_result();
        assert_eq!(p.id, id);
        assert_eq!(p.chat_id, chat_id);
        assert_eq!(p.relpath, format!("audios/{id}.wav"));

        let res = reqwest::get(format!("http://localhost:{port}/files/audios/{id}.wav")).await?;
        assert_eq!(res.status(), 200);

        Ok(())
    }

    #[tokio::test]
    async fn can_abort_a_job() -> anyhow::Result<()> {
        let port = spawn(DummyJobProcessor::new(Duration::from_millis(100)));

        let (mut ws_stream, _) = connect_async(&format!("ws://localhost:{port}/ws")).await?;
        let id = Uuid::new_v4();
        let chat_id = Uuid::new_v4();
        let msg = InboundMsg::GenerateAudio(GenerateAudioRequest {
            id,
            chat_id,
            prompt: "Create a cool song".to_string(),
            secs: 4,
        });
        ws_stream.send(msg.to_tungstenite_msg()).await?;

        tokio::time::sleep(Duration::from_millis(150)).await;
        ws_stream
            .send(
                InboundMsg::AbortGeneration(AbortGenerationRequest { id, chat_id })
                    .to_tungstenite_msg(),
            )
            .await?;

        let msg = OutboundMsg::from_tungstenite_msg(ws_stream.next().await.unwrap()?)?;
        msg.unwrap_info();

        let msg = OutboundMsg::from_tungstenite_msg(ws_stream.next().await.unwrap()?)?;
        msg.unwrap_chats();

        let msg = OutboundMsg::from_tungstenite_msg(ws_stream.next().await.unwrap()?)?;
        msg.unwrap_start();

        let msg = OutboundMsg::from_tungstenite_msg(ws_stream.next().await.unwrap()?)?;
        let p = msg.unwrap_progress();
        assert_eq!(p.id, id);
        assert_eq!(p.chat_id, chat_id);
        assert_eq!(p.progress, 0.25);

        let msg = OutboundMsg::from_tungstenite_msg(ws_stream.next().await.unwrap()?)?;
        let p = msg.unwrap_progress();
        assert_eq!(p.id, id);
        assert_eq!(p.chat_id, chat_id);
        assert_eq!(p.progress, 0.5);

        let msg = OutboundMsg::from_tungstenite_msg(ws_stream.next().await.unwrap()?)?;
        let p = msg.unwrap_err();
        assert_eq!(p.id, id);
        assert_eq!(p.chat_id, chat_id);
        assert_eq!(p.error, "Aborted");

        Ok(())
    }

    #[tokio::test]
    async fn handles_job_failures() -> anyhow::Result<()> {
        let port = spawn(DummyJobProcessor::default());

        let (mut ws_stream, _) = connect_async(&format!("ws://localhost:{port}/ws")).await?;
        let id = Uuid::new_v4();
        let chat_id = Uuid::new_v4();
        let msg = InboundMsg::GenerateAudio(GenerateAudioRequest {
            id,
            chat_id,
            prompt: "fail at 2".to_string(),
            secs: 4,
        });
        ws_stream.send(msg.to_tungstenite_msg()).await?;

        let msg = OutboundMsg::from_tungstenite_msg(ws_stream.next().await.unwrap()?)?;
        msg.unwrap_info();

        let msg = OutboundMsg::from_tungstenite_msg(ws_stream.next().await.unwrap()?)?;
        msg.unwrap_chats();

        let msg = OutboundMsg::from_tungstenite_msg(ws_stream.next().await.unwrap()?)?;
        msg.unwrap_start();

        let msg = OutboundMsg::from_tungstenite_msg(ws_stream.next().await.unwrap()?)?;
        let p = msg.unwrap_progress();
        assert_eq!(p.id, id);
        assert_eq!(p.chat_id, chat_id);
        assert_eq!(p.progress, 0.25);

        let msg = OutboundMsg::from_tungstenite_msg(ws_stream.next().await.unwrap()?)?;
        let p = msg.unwrap_progress();
        assert_eq!(p.id, id);
        assert_eq!(p.chat_id, chat_id);
        assert_eq!(p.progress, 0.5);

        let msg = OutboundMsg::from_tungstenite_msg(ws_stream.next().await.unwrap()?)?;
        let p = msg.unwrap_err();
        assert_eq!(p.id, id);
        assert_eq!(p.chat_id, chat_id);
        assert_eq!(p.error, "Failed at 2");

        Ok(())
    }

    trait TungsteniteMsg: Sized {
        fn to_tungstenite_msg(&self) -> tokio_tungstenite::tungstenite::Message;
        fn from_tungstenite_msg(
            msg: tokio_tungstenite::tungstenite::Message,
        ) -> anyhow::Result<Self>;
    }

    impl<T: Serialize + DeserializeOwned> TungsteniteMsg for T {
        fn to_tungstenite_msg(&self) -> tokio_tungstenite::tungstenite::Message {
            tokio_tungstenite::tungstenite::Message::Text(
                serde_json::to_string(self).expect("Could not serialize msg"),
            )
        }

        fn from_tungstenite_msg(
            msg: tokio_tungstenite::tungstenite::Message,
        ) -> anyhow::Result<Self> {
            let msg = msg.to_text()?;
            Ok(serde_json::de::from_str(msg)?)
        }
    }
}
