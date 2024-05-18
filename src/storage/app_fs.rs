use async_trait::async_trait;
use directories::ProjectDirs;
use lazy_static::lazy_static;

use crate::storage::{Storage, StorageFile};

#[derive(Clone)]
pub struct AppFs {
    pub root: std::path::PathBuf,
}

impl StorageFile for tokio::fs::File {}

#[async_trait]
impl Storage for AppFs {
    type File = tokio::fs::File;

    async fn exists(&self, path: &str) -> std::io::Result<bool> {
        let (abs_filepath, _, _) = self.relative_file_to_path_buf(path);
        tokio::fs::try_exists(abs_filepath).await
    }

    async fn read(&self, path: &str) -> std::io::Result<Option<Vec<u8>>> {
        let (abs_filepath, _, _) = self.relative_file_to_path_buf(path);
        match tokio::fs::read(abs_filepath).await {
            Ok(v) => Ok(Some(v)),
            Err(err) => {
                if err.kind() == std::io::ErrorKind::NotFound {
                    Ok(None)
                } else {
                    Err(err)
                }
            }
        }
    }

    async fn write(&self, path: &str, content: impl AsRef<[u8]> + Send) -> std::io::Result<()> {
        let (abs_filepath, abs_filedir, _) = self.relative_file_to_path_buf(path);
        tokio::fs::create_dir_all(abs_filedir).await?;
        tokio::fs::write(abs_filepath, content).await?;
        Ok(())
    }

    async fn create(&self, path: &str) -> std::io::Result<Self::File> {
        let (abs_filepath, abs_filedir, _) = self.relative_file_to_path_buf(path);
        tokio::fs::create_dir_all(abs_filedir).await?;
        tokio::fs::File::create(abs_filepath).await
    }

    async fn list(&self, path: &str) -> std::io::Result<Vec<String>> {
        let (abs_dir, _, _) = self.relative_file_to_path_buf(path);
        let mut files = vec![];
        let mut dir = match tokio::fs::read_dir(abs_dir).await {
            Ok(v) => v,
            Err(err) => {
                if err.kind() == std::io::ErrorKind::NotFound {
                    return Ok(vec![])
                }
                return Err(err)
            }
        };
        while let Some(entry) = dir.next_entry().await? {
            let entry_path = entry.path();
            let Ok(rel) = entry_path.strip_prefix(&self.root) else {
                continue;
            };
            files.push(rel.display().to_string())
        }
        // TODO: doesn't the OS apis already return this sorted?
        files.sort();
        Ok(files)
    }

    async fn mv(&self, from: &str, to: &str) -> std::io::Result<()> {
        let (from_filepath, _, _) = self.relative_file_to_path_buf(from);
        let (to_filepath, to_dirpath, _) = self.relative_file_to_path_buf(to);
        tokio::fs::create_dir_all(to_dirpath).await?;
        tokio::fs::rename(from_filepath, to_filepath).await
    }

    async fn rm(&self, path: &str) -> std::io::Result<bool> {
        let (abs_filepath, _, _) = self.relative_file_to_path_buf(path);
        match tokio::fs::remove_file(abs_filepath).await {
            Ok(_) => Ok(true),
            Err(err) => {
                if err.kind() == std::io::ErrorKind::NotFound {
                    Ok(false)
                } else {
                    Err(err)
                }
            }
        }
    }

    async fn rm_rf(&self, path: &str) -> std::io::Result<bool> {
        let (abs_dirpath, _, _) = self.relative_file_to_path_buf(path);
        match tokio::fs::remove_dir_all(abs_dirpath).await {
            Ok(_) => Ok(true),
            Err(err) => {
                if err.kind() == std::io::ErrorKind::NotFound {
                    Ok(false)
                } else {
                    Err(err)
                }
            }
        }
    }
}

lazy_static! {
    static ref PROJECT_DIRS: ProjectDirs = ProjectDirs::from("com", "gabotechs", "musicgpt")
        .expect("Could not load project directory");
}

impl AppFs {
    pub fn new(value: impl Into<std::path::PathBuf>) -> Self {
        Self { root: value.into() }
    }

    pub fn path_buf(&self, path: &str) -> std::path::PathBuf {
        let (abs_filepath, _, _) = self.relative_file_to_path_buf(path);
        abs_filepath
    }

    /// Gets a / separated relative path and returns:
    /// - The absolute path in the disk
    /// - The absolute path of the dir containing the file in the dis
    /// - The filename
    ///
    /// # Arguments
    ///
    /// * `relative_file`: Path relative to the applications data dir
    ///
    /// returns: (PathBuf, PathBuf, &str) (abs file path, abs dir path, file name)
    fn relative_file_to_path_buf(
        &self,
        relative_file: &str,
    ) -> (std::path::PathBuf, std::path::PathBuf, String) {
        let mut abs_file_dir = self.root.to_path_buf();

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
        (
            abs_file_dir.join(file_name),
            abs_file_dir,
            file_name.to_string(),
        )
    }
}

#[cfg(test)]
mod tests {
    use rand::distributions::Alphanumeric;
    use rand::{thread_rng, Rng};

    use crate::storage::tests::test_storage;
    use crate::storage::AppFs;

    fn rand_string() -> String {
        thread_rng()
            .sample_iter(&Alphanumeric)
            .take(7)
            .map(char::from)
            .collect()
    }

    #[tokio::test]
    async fn app_fs_works() -> std::io::Result<()> {
        let app_fs = AppFs::new(format!("/tmp/{}", rand_string()));
        test_storage(app_fs).await
    }
}
