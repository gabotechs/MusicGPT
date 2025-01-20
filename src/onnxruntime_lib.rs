#[cfg(feature = "onnxruntime-from-source")]
pub mod init {
    use ort::environment::EnvironmentBuilder;
    use std::path::PathBuf;

    use crate::storage::Storage;
    use crate::storage_ext::StorageExt;

    include!(concat!(env!("OUT_DIR"), "/built.rs"));
    include!(concat!(env!("OUT_DIR"), "/build_info.rs"));

    pub async fn init<S: Storage>(storage: S) -> anyhow::Result<EnvironmentBuilder> {
        Ok(ort::init_from(
            lookup_dynlib(storage).await?.to_str().unwrap_or_default(),
        ))
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

#[cfg(feature = "onnxruntime-from-github")]
pub mod init {
    use crate::storage::Storage;
    use crate::storage_ext::StorageExt;
    use flate2::bufread::GzDecoder;
    use ort::environment::EnvironmentBuilder;
    use std::env;
    use std::env::consts::OS;
    use std::fs::File;
    use std::io::BufReader;
    use std::path::PathBuf;
    use tar::Archive;
    use zip::ZipArchive;

    macro_rules! onnxruntime_version {
        () => {
            "1.20.1"
        };
    }
    const ONNXRUNTIME_VERSION: &str = onnxruntime_version!();

    include!(concat!(env!("OUT_DIR"), "/built.rs"));

    pub async fn init<S: Storage>(storage: S) -> anyhow::Result<EnvironmentBuilder> {
        Ok(ort::init_from(
            lookup_dynlib(storage).await?.to_str().unwrap_or_default(),
        ))
    }

    macro_rules! gh_release {
        ($file: literal, $ext: literal) => {
            concat!(
                "https://github.com/microsoft/onnxruntime/releases/download/v",
                onnxruntime_version!(),
                "/",
                $file,
                "-",
                onnxruntime_version!(),
                ".",
                $ext,
            )
        };
    }

    async fn lookup_dynlib<S: Storage>(storage: S) -> anyhow::Result<PathBuf> {
        let url = match (TARGET, cfg!(feature = "cuda")) {
            ("aarch64-apple-darwin", _) => gh_release!("onnxruntime-osx-arm64", "tgz"),
            ("aarch64-pc-windows-gnullvm", _) => gh_release!("onnxruntime-win-arm64", "zip"),
            ("aarch64-pc-windows-msvc", _) => gh_release!("onnxruntime-win-arm64", "zip"),
            ("aarch64-unknown-linux-gnu", _) => gh_release!("onnxruntime-linux-aarch64", "tgz"),
            ("aarch64-unknown-linux-musl", _) => gh_release!("onnxruntime-linux-aarch64", "tgz"),
            ("x86_64-apple-darwin", _) => gh_release!("onnxruntime-osx-x86_64", "tgz"),
            ("x86_64-pc-windows-gnu", false) => gh_release!("onnxruntime-win-x64", "zip"),
            ("x86_64-pc-windows-gnullvm", false) => gh_release!("onnxruntime-win-x64", "zip"),
            ("x86_64-pc-windows-msvc", false) => gh_release!("onnxruntime-win-x64", "zip"),
            ("x86_64-pc-windows-gnu", true) => gh_release!("onnxruntime-win-x64-gpu", "zip"),
            ("x86_64-pc-windows-gnullvm", true) => gh_release!("onnxruntime-win-x64-gpu", "zip"),
            ("x86_64-pc-windows-msvc", true) => gh_release!("onnxruntime-win-x64-gpu", "zip"),
            ("x86_64-unknown-linux-gnu", false) => gh_release!("onnxruntime-linux-x64", "tgz"),
            ("x86_64-unknown-linux-musl", false) => gh_release!("onnxruntime-linux-x64", "tgz"),
            ("x86_64-unknown-linux-gnu", true) => gh_release!("onnxruntime-linux-x64-gpu", "tgz"),
            ("x86_64-unknown-linux-musl", true) => gh_release!("onnxruntime-linux-x64-gpu", "tgz"),
            (target, _) => anyhow::bail!("Unknown target: {target}"),
        };

        let filename = url.split("/").last().unwrap();
        let filepath = format!("dynlibs/{ONNXRUNTIME_VERSION}/{filename}");
        let archive_path = storage.path_buf(&filepath);
        let mainlib = match OS {
            "macos" => "libonnxruntime.dylib",
            "windows" => "onnxruntime.dll",
            "linux" => "libonnxruntime.so",
            _ => unreachable!(),
        };
        let uncompressed_dirname = filename.replace(".zip", "").replace(".tgz", "");
        let mainlib_path = storage.path_buf(&format!(
            "dynlibs/{ONNXRUNTIME_VERSION}/{uncompressed_dirname}/lib/{mainlib}"
        ));

        if mainlib_path.exists() {
            return Ok(mainlib_path);
        }

        // If there's no local file, attempt to download it from a GitHub release.
        storage
            .download_many(
                vec![(url, filepath)],
                false,
                format!("Dynamic libraries not found, downloading them from Github release {url}"),
                "Dynamic libraries downloaded successfully",
            )
            .await?;

        extract(
            archive_path,
            storage.path_buf(&format!("dynlibs/{ONNXRUNTIME_VERSION}")),
        )?;

        if !mainlib_path.exists() {
            anyhow::bail!(
                "Main dynamic library {} not found after un-compressing archive",
                mainlib_path.display()
            );
        }

        Ok(mainlib_path)
    }

    fn extract(archive_path: PathBuf, output_dir: PathBuf) -> anyhow::Result<()> {
        if archive_path.extension() == Some(std::ffi::OsStr::new("zip")) {
            extract_zip(archive_path, output_dir)
        } else {
            extract_tar_gz(archive_path, output_dir)
        }
    }

    fn extract_tar_gz(archive_path: PathBuf, output_dir: PathBuf) -> anyhow::Result<()> {
        let tar_gz = File::open(archive_path)?;
        let buf_reader = BufReader::new(tar_gz);
        let gz_decoder = GzDecoder::new(buf_reader);
        let mut archive = Archive::new(gz_decoder);

        for entry in archive.entries()? {
            let mut entry = entry?;
            entry.unpack_in(&output_dir)?;
        }
        Ok(())
    }

    fn extract_zip(archive_path: PathBuf, output_dir: PathBuf) -> anyhow::Result<()> {
        let file = File::open(archive_path)?;
        let buf_reader = BufReader::new(file);
        let mut zip = ZipArchive::new(buf_reader)?;

        for i in 0..zip.len() {
            let mut file = zip.by_index(i)?;
            let out_path = output_dir.join(file.name());

            if file.is_dir() {
                std::fs::create_dir_all(&out_path)?;
            } else {
                if let Some(parent) = out_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let mut outfile = File::create(&out_path)?;
                std::io::copy(&mut file, &mut outfile)?;
            }
        }
        Ok(())
    }
}

#[cfg(feature = "onnxruntime-from-cdn")]
pub mod init {
    use crate::storage::Storage;
    use ort::environment::EnvironmentBuilder;

    pub async fn init<S: Storage>(_: S) -> anyhow::Result<EnvironmentBuilder> {
        Ok(ort::init())
    }
}
