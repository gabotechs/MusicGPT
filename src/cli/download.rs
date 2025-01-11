use log::info;
use std::collections::VecDeque;
use std::fmt::Display;
use std::path::PathBuf;

use crate::cli::loading_bar::LoadingBarFactory;
use crate::cli::PROJECT_FS;
use crate::storage::Storage;

pub async fn download_many<T: Display>(
    remote_file_spec: Vec<(T, T)>,
    force_download: bool,
    on_download_msg: &str,
    on_finished_msg: &str,
) -> anyhow::Result<VecDeque<PathBuf>> {
    let mut has_to_download = force_download;
    for (_, local_filename) in remote_file_spec.iter() {
        has_to_download = has_to_download || !PROJECT_FS.exists(&local_filename.to_string()).await?
    }

    if has_to_download {
        info!("{on_download_msg}");
    }
    let m = LoadingBarFactory::multi();
    let mut tasks = vec![];
    for (remote_file, local_filename) in remote_file_spec {
        let remote_file = remote_file.to_string();
        let local_filename = local_filename.to_string();
        let bar = m.add(LoadingBarFactory::download_bar(&local_filename));
        tasks.push(tokio::spawn(async move {
            PROJECT_FS
                .fetch_remote_data_file(
                    &remote_file,
                    &local_filename,
                    force_download,
                    bar.into_update_callback(),
                )
                .await
        }));
    }
    let mut results = VecDeque::new();
    for task in tasks {
        results.push_back(task.await??);
    }
    m.clear()?;
    if has_to_download {
        info!("{on_finished_msg}");
    }
    Ok(results)
}
