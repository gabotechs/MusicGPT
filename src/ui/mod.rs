mod backend_ai;
mod server;
#[cfg(test)]
mod _test_utils;
mod ws_handler;

pub use server::*;
pub use backend_ai::MusicGenJobProcessor;

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::Duration;
    use specta::ts::{BigIntExportBehavior, ExportConfiguration};
    use crate::ui::_test_utils::DummyJobProcessor;
    use crate::ui::server::run;


    #[ignore]
    #[tokio::test]
    async fn spawn_dummy_server() -> anyhow::Result<()> {
        run(DummyJobProcessor::new(Duration::from_millis(100)), 8642, false).await
    }

    #[ignore]
    #[test]
    fn export_bindings() -> anyhow::Result<()> {
        specta::export::ts_with_cfg(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("web/src/bindings.ts")
                .to_str()
                .unwrap(),
            &ExportConfiguration::default().bigint(BigIntExportBehavior::Number),
        )?;
        Ok(())
    }
}