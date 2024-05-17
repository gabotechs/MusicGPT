use std::sync::Arc;

use axum::extract::WebSocketUpgrade;
use axum::response::Html;
use axum::routing::get;
use axum::Router;
use tokio::sync::Mutex;
use tower_http::services::ServeDir;

use crate::storage::AppFs;
use crate::ui::backend_ai::{BackendAi, JobProcessor};
use crate::ui::messages::Init;
use crate::ui::ws_handler::WsHandler;

async fn web_app() -> Html<&'static str> {
    Html(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/web/dist/index.html"
    )))
}

pub async fn run<T: JobProcessor + 'static>(
    app_fs: AppFs,
    processor: T,
    port: usize,
    open: bool,
) -> anyhow::Result<()> {
    let model = processor.name();
    let device = processor.device();
    let init = Init { model, device };

    let (ai_tx, ai_rx) = BackendAi::new(processor).run();
    let ai_rx = Arc::new(Mutex::new(ai_rx));

    let app = Router::new()
        .route("/", get(web_app))
        .nest_service("/files", ServeDir::new(app_fs.root.clone()))
        .route(
            "/ws",
            get(|ws: WebSocketUpgrade| async move {
                let handler = WsHandler {
                    ai_tx: ai_tx.clone(),
                    ai_rx: ai_rx.clone(),
                    storage: app_fs.clone(),
                    info: init,
                };
                ws.on_upgrade(move |ws| handler.handle(ws))
            }),
        );

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await?;
    if open {
        let _ = open::that(format!("http://localhost:{port}"));
    }

    Ok(axum::serve(listener, app).await?)
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::time::Duration;

    use futures_util::{SinkExt, StreamExt};
    use serde::de::DeserializeOwned;
    use serde::Serialize;
    use tokio::net::TcpStream;
    use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};
    use uuid::Uuid;

    use crate::ui::_test_utils::DummyJobProcessor;
    use crate::ui::messages::{AbortGeneration, GenerateAudio, InboundMsg, OutboundMsg};

    use super::*;

    #[tokio::test]
    async fn sending_a_job_processes_it() -> anyhow::Result<()> {
        let app_fs = AppFs::new(Path::new("/tmp/ws_server_tests"));
        let processor = DummyJobProcessor::default();
        tokio::spawn(run(app_fs, processor, 8642, false));

        ws_works("ws://localhost:8642/ws").await?;
        Ok(())
    }

    #[tokio::test]
    async fn second_connection_fails_if_first_is_still_active() -> anyhow::Result<()> {
        let app_fs = AppFs::new(Path::new("/tmp/ws_server_tests"));
        let processor = DummyJobProcessor::default();
        tokio::spawn(run(app_fs, processor, 8643, false));

        let _ = ws_works("ws://localhost:8643/ws").await?;
        ws_closes_connection("ws://localhost:8643/ws").await?;

        Ok(())
    }

    #[tokio::test]
    async fn second_connection_succeeds_if_first_releases() -> anyhow::Result<()> {
        let app_fs = AppFs::new(Path::new("/tmp/ws_server_tests"));
        let processor = DummyJobProcessor::default();
        tokio::spawn(run(app_fs, processor, 8644, false));

        {
            let mut ws = ws_works("ws://localhost:8644/ws").await?;
            ws_closes_connection("ws://localhost:8644/ws").await?;
            ws.close(None).await?;
            // leave some room for tearing down old connection.
            tokio::time::sleep(Duration::from_millis(300)).await;
        }

        ws_works("ws://localhost:8644/ws").await?;

        Ok(())
    }

    #[tokio::test]
    async fn can_abort_a_job() -> anyhow::Result<()> {
        let app_fs = AppFs::new(Path::new("/tmp/ws_server_tests"));
        let processor = DummyJobProcessor::new(Duration::from_millis(100));
        tokio::spawn(run(app_fs, processor, 8645, false));

        let (mut ws_stream, _) = connect_async("ws://localhost:8645/ws").await?;
        let id = Uuid::new_v4();
        let chat_id = Uuid::new_v4();
        let msg = InboundMsg::GenerateAudio(GenerateAudio {
            id,
            chat_id,
            prompt: "Create a cool song".to_string(),
            secs: 4,
        });
        ws_stream.send(msg.to_tungstenite_msg()).await?;

        tokio::time::sleep(Duration::from_millis(150)).await;
        ws_stream
            .send(InboundMsg::AbortGeneration(AbortGeneration { id, chat_id }).to_tungstenite_msg())
            .await?;
        let msg = OutboundMsg::from_tungstenite_msg(ws_stream.next().await.unwrap()?)?;
        msg.unwrap_init();

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

    async fn ws_closes_connection(url: &str) -> anyhow::Result<()> {
        let (mut ws, _) = connect_async(url).await?;
        assert!(ws.next().await.unwrap()?.is_close());
        Ok(())
    }

    async fn ws_works(url: &str) -> anyhow::Result<WebSocketStream<MaybeTlsStream<TcpStream>>> {
        let (mut ws_stream, _) = connect_async(url).await?;
        let id = Uuid::new_v4();
        let chat_id = Uuid::new_v4();
        let msg = InboundMsg::GenerateAudio(GenerateAudio {
            id,
            chat_id,
            prompt: "Create a cool song".to_string(),
            secs: 4,
        });
        ws_stream.send(msg.to_tungstenite_msg()).await?;

        let msg = OutboundMsg::from_tungstenite_msg(ws_stream.next().await.unwrap()?)?;
        msg.unwrap_init();

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

        Ok(ws_stream)
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
