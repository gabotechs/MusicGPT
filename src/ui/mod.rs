pub use backend_ai::MusicGenJobProcessor;
pub use server::*;

mod backend_ai;
mod server;
#[cfg(test)]
mod _test_utils;
mod ws_handler;
mod messages;
mod history_entry;

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::time::Duration;
    use crate::storage::AppFs;

    use crate::ui::_test_utils::DummyJobProcessor;
    use crate::ui::server::run;

    #[ignore]
    #[tokio::test]
    async fn spawn_dummy_server() -> anyhow::Result<()> {
        let app_fs = AppFs::new(Path::new("/tmp/dummy-server"));
        run(app_fs, DummyJobProcessor::new(Duration::from_millis(100)), 8642, false).await
    }
}