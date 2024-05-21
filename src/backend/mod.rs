pub use audio_generation_backend::MusicGenJobProcessor;
pub use server::*;

mod audio_generation_backend;
mod server;
#[cfg(test)]
mod _test_utils;
mod music_gpt_chat;
mod audio_generation_fanout;
mod ws_handler;
mod music_gpt_ws_handler;

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::time::Duration;
    use specta::ts::{BigIntExportBehavior, ExportConfiguration};

    use crate::storage::AppFs;
    use crate::backend::_test_utils::DummyJobProcessor;
    use crate::backend::RunOptions;
    use crate::backend::server::run;

    #[ignore]
    #[tokio::test]
    async fn spawn_dummy_server() -> anyhow::Result<()> {
        let storage = AppFs::new(Path::new("/tmp/dummy-server"));
        let processor = DummyJobProcessor::new(Duration::from_millis(100));
        let options = RunOptions {
            port: 8642,
            auto_open: false,
            expose: false,
        };
        run(storage, processor, options).await
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