pub use audio_generation_backend::JobProcessor;
pub use server::*;

#[cfg(test)]
mod _test_utils;
mod audio_generation_backend;
mod audio_generation_fanout;
mod music_gpt_chat;
mod music_gpt_ws_handler;
mod server;
mod ws_handler;

#[cfg(test)]
mod tests {
    use specta::ts::{BigIntExportBehavior, ExportConfiguration};
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    use crate::backend::RunOptions;
    use crate::backend::_test_utils::DummyJobProcessor;
    use crate::backend::server::run_web_server;
    use crate::storage::AppFs;

    #[ignore]
    #[tokio::test]
    async fn spawn_dummy_server() -> anyhow::Result<()> {
        let storage = AppFs::new(Path::new("/tmp/dummy-server"));
        let processor = DummyJobProcessor::new(Duration::from_millis(100));
        let options = RunOptions {
            device: "Cpu".to_string(),
            name: "Dummy".to_string(),
            port: 8642,
            auto_open: false,
            expose: false,
        };
        run_web_server(storage.root.clone(), storage, processor, options).await
    }

    #[ignore]
    #[test]
    fn export_bindings() -> anyhow::Result<()> {
        specta::export::ts_with_cfg(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("web/src/backend/bindings.ts")
                .to_str()
                .unwrap(),
            &ExportConfiguration::default().bigint(BigIntExportBehavior::Number),
        )?;
        Ok(())
    }
}
