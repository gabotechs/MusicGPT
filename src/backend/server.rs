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
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicU16, Ordering};
    use std::time::Duration;

    use futures_util::{SinkExt, StreamExt};
    use serde::de::DeserializeOwned;
    use serde::Serialize;
    use tokio::net::TcpStream;
    use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};
    use uuid::Uuid;

    use crate::backend::_test_utils::DummyJobProcessor;
    use crate::backend::music_gpt_chat::{AiChatEntry, ChatEntry, UserChatEntry};
    use crate::backend::music_gpt_ws_handler::{
        AbortGenerationRequest, ChatRequest, GenerateAudioRequest, InboundMsg, OutboundMsg,
    };

    use super::*;

    #[tokio::test]
    async fn sending_a_job_processes_it() -> anyhow::Result<()> {
        let (mut ws, host) = spawn(DummyJobProcessor::default()).await?;

        let id = Uuid::new_v4();
        let chat_id = Uuid::new_v4();
        InboundMsg::GenerateAudio(GenerateAudioRequest {
            id,
            chat_id,
            prompt: "Create a cool song".to_string(),
            secs: 4,
        })
        .to_ws(&mut ws)
        .await?;

        OutboundMsg::from_ws(&mut ws).await?.info();
        OutboundMsg::from_ws(&mut ws).await?.chats();
        OutboundMsg::from_ws(&mut ws).await?.start();

        let p = OutboundMsg::from_ws(&mut ws).await?.progress();
        assert_eq!(p.id, id);
        assert_eq!(p.chat_id, chat_id);
        assert_eq!(p.progress, 0.25);

        let p = OutboundMsg::from_ws(&mut ws).await?.progress();
        assert_eq!(p.progress, 0.5);

        let p = OutboundMsg::from_ws(&mut ws).await?.progress();
        assert_eq!(p.progress, 0.75);

        let p = OutboundMsg::from_ws(&mut ws).await?.progress();
        assert_eq!(p.progress, 1.0);

        let p = OutboundMsg::from_ws(&mut ws).await?.result();
        assert_eq!(p.id, id);
        assert_eq!(p.chat_id, chat_id);
        assert_eq!(p.relpath, format!("audios/{id}.wav"));

        let res = reqwest::get(format!("http://{host}/files/audios/{id}.wav")).await?;
        assert_eq!(res.status(), 200);

        Ok(())
    }

    #[tokio::test]
    async fn can_abort_a_job() -> anyhow::Result<()> {
        let (mut ws, _) = spawn(DummyJobProcessor::new(Duration::from_millis(200))).await?;

        let id = Uuid::new_v4();
        let chat_id = Uuid::new_v4();
        InboundMsg::GenerateAudio(GenerateAudioRequest {
            id,
            chat_id,
            prompt: "Create a cool song".to_string(),
            secs: 4,
        })
        .to_ws(&mut ws)
        .await?;

        tokio::time::sleep(Duration::from_millis(50)).await;

        InboundMsg::AbortGeneration(AbortGenerationRequest { id, chat_id })
            .to_ws(&mut ws)
            .await?;

        OutboundMsg::from_ws(&mut ws).await?.info();
        OutboundMsg::from_ws(&mut ws).await?.chats();
        OutboundMsg::from_ws(&mut ws).await?.start();

        let p = OutboundMsg::from_ws(&mut ws).await?.progress();
        assert_eq!(p.id, id);
        assert_eq!(p.chat_id, chat_id);
        assert_eq!(p.progress, 0.25);

        let p = OutboundMsg::from_ws(&mut ws).await?.error();
        assert_eq!(p.id, id);
        assert_eq!(p.chat_id, chat_id);
        assert_eq!(p.error, "Aborted");

        Ok(())
    }

    #[tokio::test]
    async fn handles_job_failures() -> anyhow::Result<()> {
        let (mut ws, _) = spawn(DummyJobProcessor::default()).await?;

        let id = Uuid::new_v4();
        let chat_id = Uuid::new_v4();
        InboundMsg::GenerateAudio(GenerateAudioRequest {
            id,
            chat_id,
            prompt: "fail at 2".to_string(),
            secs: 4,
        })
        .to_ws(&mut ws)
        .await?;

        OutboundMsg::from_ws(&mut ws).await?.info();
        OutboundMsg::from_ws(&mut ws).await?.chats();
        OutboundMsg::from_ws(&mut ws).await?.start();

        let p = OutboundMsg::from_ws(&mut ws).await?.progress();
        assert_eq!(p.id, id);
        assert_eq!(p.chat_id, chat_id);
        assert_eq!(p.progress, 0.25);

        let p = OutboundMsg::from_ws(&mut ws).await?.progress();
        assert_eq!(p.progress, 0.5);

        let p = OutboundMsg::from_ws(&mut ws).await?.error();
        assert_eq!(p.id, id);
        assert_eq!(p.chat_id, chat_id);
        assert_eq!(p.error, "Failed at 2");

        Ok(())
    }

    #[tokio::test]
    async fn handles_chats() -> anyhow::Result<()> {
        let (mut ws, _) = spawn(DummyJobProcessor::default()).await?;


        OutboundMsg::from_ws(&mut ws).await?.info();
        OutboundMsg::from_ws(&mut ws).await?.chats();

        let id = Uuid::new_v4();
        let chat_id = Uuid::new_v4();
        InboundMsg::GenerateAudioNewChat(GenerateAudioRequest {
            id,
            chat_id,
            prompt: "foo".to_string(),
            secs: 1,
        })
            .to_ws(&mut ws)
            .await?;
        OutboundMsg::from_ws(&mut ws).await?.chats();

        OutboundMsg::from_ws(&mut ws).await?.start();
        OutboundMsg::from_ws(&mut ws).await?.progress();
        OutboundMsg::from_ws(&mut ws).await?.result();

        InboundMsg::GetChat(ChatRequest { chat_id })
            .to_ws(&mut ws)
            .await?;
        
        let (chat, entries) = OutboundMsg::from_ws(&mut ws).await?.chat();
        assert_eq!(chat.chat_id, chat_id);
        assert_eq!(chat.name, "foo");
        assert_eq!(entries.len(), 2);
        
        assert_eq!(entries[0], ChatEntry::User(UserChatEntry {
            id,
            chat_id,
            text: "foo".to_string(),
        }));

        assert_eq!(entries[1], ChatEntry::Ai(AiChatEntry {
            id,
            chat_id,
            relpath: format!("audios/{id}.wav"),
            error: "".to_string(),
        }));

        Ok(())
    }

    #[async_trait]
    trait TungsteniteMsg: Sized {
        async fn to_ws(
            self,
            ws: &mut WebSocketStream<MaybeTlsStream<TcpStream>>,
        ) -> anyhow::Result<()>;

        async fn from_ws(
            ws: &mut WebSocketStream<MaybeTlsStream<TcpStream>>,
        ) -> anyhow::Result<Self>;
    }

    #[async_trait]
    impl<T: Serialize + DeserializeOwned + Send> TungsteniteMsg for T {
        async fn to_ws(
            self,
            ws: &mut WebSocketStream<MaybeTlsStream<TcpStream>>,
        ) -> anyhow::Result<()> {
            let msg = serde_json::to_string(&self).expect("Could not serialize msg");
            Ok(ws
                .send(tokio_tungstenite::tungstenite::Message::Text(msg))
                .await?)
        }

        async fn from_ws(
            ws: &mut WebSocketStream<MaybeTlsStream<TcpStream>>,
        ) -> anyhow::Result<Self> {
            let msg = ws.next().await.unwrap().unwrap();
            Ok(serde_json::de::from_str(msg.to_text()?)?)
        }
    }

    static PORT: AtomicU16 = AtomicU16::new(8643);

    async fn spawn<P: JobProcessor + 'static>(
        processor: P,
    ) -> anyhow::Result<(WebSocketStream<MaybeTlsStream<TcpStream>>, String)> {
        let app_fs = AppFs::new_tmp();
        let port = PORT.fetch_add(1, Ordering::SeqCst) as usize;
        let run_options = RunOptions {
            port,
            auto_open: false,
            expose: false,
        };
        tokio::spawn(run(app_fs, processor, run_options));
        let (ws_stream, _) = connect_async(&format!("ws://localhost:{port}/ws")).await?;
        Ok((ws_stream, format!("localhost:{port}")))
    }
}
