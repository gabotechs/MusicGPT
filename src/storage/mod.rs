mod app_fs;

pub use app_fs::*;

use async_trait::async_trait;
use tokio::io::AsyncWriteExt;

// Don't know why clippy says that this is dead code,
// it's used in app_fs.rs
#[allow(dead_code)]
pub trait StorageFile: AsyncWriteExt + Unpin {}

#[async_trait]
pub trait Storage: Sync + Send + Clone {
    type File: StorageFile;

    async fn exists(&self, path: &str) -> std::io::Result<bool>;
    async fn read(&self, path: &str) -> std::io::Result<Option<Vec<u8>>>;
    async fn write(&self, path: &str, content: impl AsRef<[u8]> + Send) -> std::io::Result<()>;
    async fn create(&self, path: &str) -> std::io::Result<Self::File>;
    async fn list(&self, path: &str) -> std::io::Result<Vec<String>>;
    async fn mv(&self, from: &str, to: &str) -> std::io::Result<()>;
    async fn rm(&self, path: &str) -> std::io::Result<bool>;
    async fn rm_rf(&self, path: &str) -> std::io::Result<bool>;
}

#[cfg(test)]
mod tests {
    use super::*;

    pub async fn test_storage<S: Storage>(s: S) -> std::io::Result<()> {
        // it should create a file
        s.write("foo/bar.txt", "test content").await?;

        // it should very that it exists
        assert!(s.exists("foo/bar.txt").await?);
        // it should return false if file does not exist
        assert!(!s.exists("foo/NON_EXISTING.txt").await?);
        assert!(!s.exists("NON_EXISTING/bar.txt").await?);

        // it should return None if reading a non-existing file
        let content = s.read("foo/NON_EXISTING.txt").await?;
        assert!(content.is_none());

        // it should return the content if reading an actual file
        let content = s.read("foo/bar.txt").await?;
        assert!(content.is_some());
        let content = content.unwrap();
        assert_eq!(String::from_utf8_lossy(&content), "test content");

        // it should move the file
        s.mv("foo/bar.txt", "bar/foo.txt").await?;
        assert!(!s.exists("foo/bar.txt").await?);
        assert!(s.exists("bar/foo.txt").await?);
        let content = s.read("bar/foo.txt").await?;
        assert!(content.is_some());
        let content = content.unwrap();
        assert_eq!(String::from_utf8_lossy(&content), "test content");

        // it should list files
        for i in 0..3 {
            let mut file = s.create(&format!("list/{i}.txt")).await?;
            file.write_all(format!("{i}").as_bytes()).await?;
        }
        let list = s.list("list/").await?;
        assert_eq!(list, vec!["list/0.txt", "list/1.txt", "list/2.txt"]);
        let list = s.list("list").await?;
        assert_eq!(list, vec!["list/0.txt", "list/1.txt", "list/2.txt"]);
        
        // listing empty directory returns empty array
        let list = s.list("NON_EXISTING/").await?;
        assert_eq!(list, Vec::<String>::new());
        
        // it should remove a single file
        s.write("to_remove/foo.txt", "foo").await?;
        assert!(s.exists("to_remove/foo.txt").await?);
        let is_removed = s.rm("to_remove/foo.txt").await?;
        assert!(is_removed);
        assert!(!s.exists("to_remove/foo.txt").await?);
        
        // it should allow removing non-existing files without failing
        let is_removed = s.rm("NON_EXISTING/foo.txt").await?;
        assert!(!is_removed);
        
        // it should remove multiple files
        for i in 0..3 {
            let mut file = s.create(&format!("to_remove/{i}.txt")).await?;
            file.write_all(format!("{i}").as_bytes()).await?;
        }
        for i in 0..3 {
            assert!(s.exists(&format!("to_remove/{i}.txt")).await?);
        }
        let is_removed = s.rm_rf("to_remove").await?;
        assert!(is_removed);
        for i in 0..3 {
            assert!(!s.exists(&format!("to_remove/{i}.txt")).await?);
        }

        Ok(())
    }
}
