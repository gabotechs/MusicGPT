use log::info;
use std::collections::VecDeque;
use std::fmt::{Display, Write};
use std::path::PathBuf;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressState, ProgressStyle};

use crate::cli::storage_ext::StorageExt;
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
    let m = MultiProgress::new();
    let mut tasks = vec![];
    for (remote_file, local_filename) in remote_file_spec {
        let remote_file = remote_file.to_string();
        let local_filename = local_filename.to_string();
        let bar = m.add(download_bar(&local_filename));
        tasks.push(tokio::spawn(async move {
            PROJECT_FS
                .fetch_remote_data_file(
                    &remote_file,
                    &local_filename,
                    force_download,
                    Box::new(move |el, t| {
                        bar.set_length(t as u64);
                        bar.set_position(el as u64);
                    })
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

pub fn download_bar(file: &str) -> ProgressBar {
    const NAME_LEN: usize = 32;
    const NAME_SHIFT_INTERVAL: usize = 300;
    let pb = ProgressBar::with_draw_target(None, ProgressDrawTarget::stderr());
    let file_string = file.to_string();
    pb.set_style(
        ProgressStyle::with_template(
            "{file:>32} {spinner:.green} [{wide_bar:.cyan/blue}] {bytes}/{total_bytes}",
        )
            .unwrap()
            .with_key("file", move |state: &ProgressState, w: &mut dyn Write| {
                if file_string.len() > NAME_LEN {
                    let el = state.elapsed().as_millis() as usize;
                    let offset = (el / NAME_SHIFT_INTERVAL) % (file_string.len() - NAME_LEN + 1);
                    let view = &file_string[offset..offset + NAME_LEN];
                    write!(w, "{view: >w$}", w = NAME_LEN).unwrap();
                } else {
                    write!(w, "{file_string: >w$}", w = NAME_LEN).unwrap();
                }
            })
            .with_key("eta", |state: &ProgressState, w: &mut dyn Write| {
                write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap()
            })
            .progress_chars("#>-"),
    );
    pb
}
