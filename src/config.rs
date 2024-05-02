use directories::ProjectDirs;
use futures_util::StreamExt;
use reqwest::StatusCode;
use std::error;
use std::path::PathBuf;
use tokio::fs;
use tokio::io::AsyncWriteExt;

pub struct Config {
    dirs: ProjectDirs,
}

impl Config {
    pub fn new() -> Self {
        let dirs = ProjectDirs::from("com", "gabotechs", "music-gen")
            .expect("Could not load project directory");

        Self { dirs }
    }

    /// Loads a remote from the local data directory, downloading it from
    /// the remote endpoint if necessary
    ///
    /// # Arguments
    ///
    /// * `url`: The URL of the remote file
    /// * `file_name`: The filename in the local data directory
    /// * `cbk`: A callback for tracking progress of the download (elapsed, total)
    ///
    /// returns: Result<PathBuf, Error>
    pub async fn remote_data_file<Cb: Fn(usize, usize)>(
        &self,
        url: &str,
        file_name: &str,
        cbk: Cb,
    ) -> std::io::Result<PathBuf> {
        let data_dir = self.dirs.data_dir();
        let abs_file_path = data_dir.join(file_name);
        if fs::try_exists(abs_file_path.clone()).await? {
            return Ok(abs_file_path);
        }

        let resp = reqwest::get(url).await.map_err(io_err)?;
        let status_code = resp.status();
        if status_code != StatusCode::OK {
            return Err(io_err(format!("Invalid status code {status_code}")));
        }
        let total_bytes = resp.content_length().unwrap_or_default() as usize;

        let _ = fs::create_dir_all(data_dir).await;
        let mut file = fs::File::create(abs_file_path.clone()).await?;
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

        Ok(abs_file_path)
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
    use super::*;
    use rand::distributions::Alphanumeric;
    use rand::{thread_rng, Rng};
    use std::time::SystemTime;

    fn rand_string() -> String {
        thread_rng()
            .sample_iter(&Alphanumeric)
            .take(7)
            .map(char::from)
            .collect()
    }

    #[tokio::test]
    async fn downloads_remote_file() -> std::io::Result<()> {
        let config = Config::new();
        let remote_file = "https://raw.githubusercontent.com/seanmonstar/reqwest/master/README.md";
        let file_name = rand_string() + ".txt";

        let time = SystemTime::now();
        let file = config
            .remote_data_file(remote_file, &file_name, |_, _| {})
            .await?;
        let download_elapsed = SystemTime::now().duration_since(time).unwrap().as_micros();

        assert!(fs::try_exists(file.clone()).await?);

        let time = SystemTime::now();
        config
            .remote_data_file(remote_file, &file_name, |_, _| {})
            .await?;
        let cached_elapsed = SystemTime::now().duration_since(time).unwrap().as_micros();

        assert!(download_elapsed / cached_elapsed > 10);

        fs::remove_file(file).await?;
        Ok(())
    }
}
