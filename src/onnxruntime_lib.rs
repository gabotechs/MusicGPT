#[cfg(feature = "onnxruntime-from-source")]
pub mod init {
    use std::path::PathBuf;
    use ort::environment::EnvironmentBuilder;
    
    use crate::storage::Storage;
    use crate::storage_ext::StorageExt;

    include!(concat!(env!("OUT_DIR"), "/built.rs"));
    include!(concat!(env!("OUT_DIR"), "/build_info.rs"));

    pub async fn init<S: Storage>(storage: S) -> anyhow::Result<EnvironmentBuilder> {
        Ok(ort::init_from(
            lookup_dynlib(storage)
                .await?
                .to_str()
                .unwrap_or_default())
        )
    }

    async fn lookup_dynlib<S: Storage>(storage: S) -> anyhow::Result<PathBuf> {
        // If running with Cargo, build.rs have set this ONNXRUNTIME_LOCAL_FILES env to the
        // path of the generated dynamic library files compiled from source.
        // If not running with cargo, this will not be set.
        for local_filepath in LOCAL_DYNLIB_FILEPATHS {
            if local_filepath.ends_with(&MAIN_DYNLIB_FILENAME)
                && tokio::fs::try_exists(&local_filepath).await?
            {
                return Ok(PathBuf::from(local_filepath));
            }
        }

        // If there's no local file, attempt to download it from a GitHub release.
        let remote_file_spec = DYNLIB_FILENAMES
            .iter()
            .map(|v| {
                (
                    // It's very important that the remote filename matches what the pipelines upload here:
                    // https://github.com/gabotechs/MusicGPT/blob/main/.github/workflows/ci.yml#L188
                    format!("{PKG_REPOSITORY}/releases/download/v{PKG_VERSION}/{TARGET}-{v}"),
                    format!("dynlibs/{ONNXRUNTIME_VERSION}/{v}"),
                )
            })
            .collect::<Vec<_>>();
        storage.download_many(
            remote_file_spec,
            false,
            &format!("Dynamic libraries not found in path set by ONNXRUNTIME_LOCAL_FILES env variable. Downloading them from GitHub release {PKG_VERSION}..."),
            "Dynamic libraries downloaded successfully",
        )
            .await?;
        let main_dynlib_file = storage.path_buf(&format!(
            "dynlibs/{ONNXRUNTIME_VERSION}/{MAIN_DYNLIB_FILENAME}"
        ));
        if !tokio::fs::try_exists(&main_dynlib_file).await? {
            return Err(anyhow::anyhow!(
            "dynamic library file {main_dynlib_file:?} not found"
        ));
        }
        Ok(main_dynlib_file)
    }
}


#[cfg(not(feature = "onnxruntime-from-source"))]
pub mod init {
    use ort::environment::EnvironmentBuilder;
    use crate::storage::Storage;
    
    pub async fn init<S: Storage>(_: S) -> anyhow::Result<EnvironmentBuilder> {
        Ok(ort::init())
    }
}
