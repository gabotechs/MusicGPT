use std::error;
use std::path::PathBuf;

use axum::http::StatusCode;
use futures_util::StreamExt;
use tokio::io::AsyncWriteExt;

use crate::storage::{AppFs, Storage};

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
impl AppFs {
    pub async fn fetch_remote_data_file<Cb: Fn(usize, usize)>(
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
            return Err(io_err(format!("Invalid status code {status_code}")));
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

fn io_err<E>(e: E) -> std::io::Error
where
    E: Into<Box<dyn error::Error + Send + Sync>>,
{
    std::io::Error::new(std::io::ErrorKind::Other, e)
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::time::SystemTime;

    use rand::distributions::Alphanumeric;
    use rand::{thread_rng, Rng};

    use crate::storage::AppFs;

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
