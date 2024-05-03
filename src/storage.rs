use directories::ProjectDirs;
use futures_util::StreamExt;
use reqwest::StatusCode;
use std::error;
use std::path::PathBuf;
use tokio::fs;
use tokio::io::AsyncWriteExt;

pub struct Storage {
    dirs: ProjectDirs,
}

impl Storage {
    fn new() -> Self {
        let dirs = ProjectDirs::from("com", "gabotechs", "musicgpt")
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
        url: &str,
        relative_file: &str,
        force: bool,
        cbk: Cb,
    ) -> std::io::Result<PathBuf> {
        let cfg = Storage::new();
        let mut abs_file_dir = cfg.dirs.data_dir().to_path_buf();

        // The provided `relative_path` might contain directories separated with /
        let mut relative_file_elements = relative_file.split('/').collect::<Vec<_>>();
        // so take the file name...
        let file_name = relative_file_elements
            .pop()
            .expect("provided path was empty");
        // ... and append the rest of the elements to the base directory.
        for element in relative_file_elements {
            abs_file_dir = abs_file_dir.join(element);
        }
        let abs_file_path = abs_file_dir.join(file_name);
        // At this point, the file might already exist on disk, so nothing else to do.
        if fs::try_exists(abs_file_path.clone()).await? && !force {
            return Ok(abs_file_path);
        }

        // If the file was not in disk, we need to download it.
        let resp = reqwest::get(url).await.map_err(io_err)?;
        let status_code = resp.status();
        if status_code != StatusCode::OK {
            return Err(io_err(format!("Invalid status code {status_code}")));
        }
        let total_bytes = resp.content_length().unwrap_or_default() as usize;

        // The file will be first downloaded to a temporary file, to avoid corruptions.
        fs::create_dir_all(abs_file_dir.clone()).await?;
        let temp_abs_file_path = abs_file_dir.join(file_name.to_string() + ".temp");
        let mut file = fs::File::create(temp_abs_file_path.clone()).await?;

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
        fs::rename(temp_abs_file_path, abs_file_path.clone()).await?;

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
        let remote_file = "https://raw.githubusercontent.com/seanmonstar/reqwest/master/README.md";
        let file_name = "foo/".to_string() + &rand_string() + ".txt";

        let time = SystemTime::now();
        let file = Storage::remote_data_file(remote_file, &file_name, false, |_, _| {}).await?;
        let download_elapsed = SystemTime::now().duration_since(time).unwrap().as_micros();

        assert!(fs::try_exists(file.clone()).await?);

        let time = SystemTime::now();
        Storage::remote_data_file(remote_file, &file_name, false, |_, _| {}).await?;
        let cached_elapsed = SystemTime::now().duration_since(time).unwrap().as_micros();

        assert!(download_elapsed / cached_elapsed > 10);

        fs::remove_file(file).await?;
        Ok(())
    }
}
