use async_trait::async_trait;
use axum::http::StatusCode;
use futures_util::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressState, ProgressStyle};
use log::info;
use std::collections::VecDeque;
use std::error;
use std::fmt::{Display, Write};
use std::path::PathBuf;
use tokio::io::AsyncWriteExt;

use crate::storage::Storage;


#[async_trait]
pub trait StorageExt: Storage {
    async fn download_many<
        T1: Display + Send + Sync + 'static,
        T2: Display + Send + Sync + 'static,
    >(
        &self,
        remote_file_spec: Vec<(T1, T2)>,
        force_download: bool,
        on_download_msg: impl Display + Send + Sync + 'static,
        on_finished_msg: impl Display + Send + Sync + 'static,
    ) -> anyhow::Result<VecDeque<PathBuf>> {
        let mut has_to_download = force_download;
        for (_, local_filename) in remote_file_spec.iter() {
            has_to_download = has_to_download || !self.exists(&local_filename.to_string()).await?
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
            let this = self.clone();
            tasks.push(tokio::spawn(async move {
                this.fetch_remote_data_file(
                    &remote_file,
                    &local_filename,
                    force_download,
                    Box::new(move |el, t| {
                        bar.set_length(t as u64);
                        bar.set_position(el as u64);
                    }),
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

    /// Loads a remote from the local data directory, downloading it from
    /// the remote endpoint if necessary
    ///
    /// # Arguments
    ///
    /// * `url`: The URL of the remote file
    /// * `file_name`: The filename in the local data directory
    /// * `force`: Force the download even if the file exists
    /// * `cbk`: A callback for tracking progress of the download (elapsed, total)
    ///
    /// returns: Result<PathBuf, Error>
    async fn fetch_remote_data_file<Cb: Fn(usize, usize) + Send + Sync + 'static>(
        &self,
        url: &str,
        local_file: &str,
        force: bool,
        cbk: Cb,
    ) -> std::io::Result<PathBuf> {
        // At this point, the file might already exist on disk, so nothing else to do.
        if self.exists(local_file).await? && !force {
            return Ok(self.path_buf(local_file));
        }

        // If the file was not in disk, we need to download it.
        let resp = reqwest::get(url).await.map_err(io_err)?;
        let status_code = resp.status();
        if status_code != StatusCode::OK {
            return Err(io_err(format!(
                "Error downloading {url}. Invalid status code {status_code}"
            )));
        }
        let total_bytes = resp.content_length().unwrap_or_default() as usize;

        // The file will be first downloaded to a temporary file, to avoid corruptions.
        let temp_file = format!("{local_file}.temp");
        let mut file = self.create(&temp_file).await?;

        // Stream the HTTP response to the file stream.
        let mut stream = resp.bytes_stream();
        let mut downloaded_bytes = 0;
        while let Some(item) = stream.next().await {
            match item {
                Ok(chunk) => {
                    downloaded_bytes += chunk.len();
                    cbk(downloaded_bytes, total_bytes);
                    file.write_all(&chunk).await?
                }
                Err(err) => return Err(io_err(err)),
            }
        }

        // If everything succeeded, we are fine to promote the newly stored temporary
        // file to the actual destination.
        self.mv(&temp_file, local_file).await?;

        Ok(self.path_buf(local_file))
    }
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

impl<T: Storage + 'static> StorageExt for T {}

fn io_err<E>(e: E) -> std::io::Error
where
    E: Into<Box<dyn error::Error + Send + Sync>>,
{
    std::io::Error::new(std::io::ErrorKind::Other, e)
}

#[cfg(test)]
mod tests {
    use rand::distributions::Alphanumeric;
    use rand::{thread_rng, Rng};
    use std::path::Path;
    use std::time::SystemTime;

    use crate::storage::AppFs;
    use crate::storage_ext::StorageExt;

    fn rand_string() -> String {
        thread_rng()
            .sample_iter(&Alphanumeric)
            .take(7)
            .map(char::from)
            .collect()
    }

    #[tokio::test]
    async fn downloads_remote_file() -> std::io::Result<()> {
        let remote_file = "https://raw.githubusercontent.com/seanmonstar/reqwest/master/README.md";
        let file_name = format!("foo/{}.txt", rand_string());

        let app_fs = AppFs::new(Path::new("/tmp/downloads_remote_file_test"));

        let time = SystemTime::now();
        app_fs
            .fetch_remote_data_file(remote_file, &file_name, false, |_, _| {})
            .await?;
        let download_elapsed = SystemTime::now().duration_since(time).unwrap().as_micros();

        let time = SystemTime::now();
        app_fs
            .fetch_remote_data_file(remote_file, &file_name, false, |_, _| {})
            .await?;
        let cached_elapsed = SystemTime::now().duration_since(time).unwrap().as_micros();

        assert!(download_elapsed / cached_elapsed > 10);

        Ok(())
    }
}
