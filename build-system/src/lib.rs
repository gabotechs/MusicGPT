use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use serde_json::Serializer;
use std::fmt::Display;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::{env, fs};
use tar::Archive;

const ONNX_RELEASE: &str = "1.20.1";
const PROFILE: &str = "Release";

#[cfg(target_os = "macos")]
const DYN_LIB_EXT: &str = "dylib";
#[cfg(target_os = "macos")]
const MAIN_DYNLIB_FILENAME: &str = "libonnxruntime.dylib";

#[cfg(target_os = "windows")]
const DYN_LIB_EXT: &str = "dll";
#[cfg(target_os = "windows")]
const MAIN_DYNLIB_FILENAME: &str = "onnxruntime.dll";

#[cfg(target_os = "linux")]
const DYN_LIB_EXT: &str = "so";
#[cfg(target_os = "linux")]
const MAIN_DYNLIB_FILENAME: &str = "libonnxruntime.so";

pub enum Accelerators {
    COREML,
    TENSORRT,
    CUDA,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct BuildInfo {
    /// absolute path to all the compiled dynamic library files.
    pub local_dynlib_filepaths: Vec<PathBuf>,
    /// version of the onnxruntime.
    pub onnxruntime_version: String,
    /// filename of the main library.
    pub main_dynlib_filename: String,
    /// filenames of the generated dynamic libraries.
    pub dynlib_filenames: Vec<String>,

    cmd: String,
}

impl Display for BuildInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "pub const LOCAL_DYNLIB_FILEPATHS: [&str; {}] = [{}];",
            self.local_dynlib_filepaths.len(),
            self.local_dynlib_filepaths
                .iter()
                .map(|v| format!("{}", serde_json::ser::to_string(v).unwrap()))
                .reduce(|a, b| format!("{a}, {b}"))
                .unwrap_or_default()
        )?;
        writeln!(
            f,
            "pub const ONNXRUNTIME_VERSION: &str = \"{}\";",
            self.onnxruntime_version
        )?;
        writeln!(
            f,
            "pub const MAIN_DYNLIB_FILENAME: &str = \"{}\";",
            self.main_dynlib_filename
        )?;
        writeln!(
            f,
            "pub const DYNLIB_FILENAMES: [&str; {}] = [{}];",
            self.dynlib_filenames.len(),
            self.dynlib_filenames
                .iter()
                .map(|v| format!("\"{v}\""))
                .reduce(|a, b| format!("{a}, {b}"))
                .unwrap_or_default()
        )?;

        Ok(())
    }
}

impl BuildInfo {
    /// Loads the BuildInfo from the out dir.
    fn from_dir(dir: &PathBuf) -> Option<Self> {
        let file = dir.join("onnxruntime-build-info.json");
        if file_exists(&file) {
            let content =
                fs::read(&file).expect(&format!("Could not read build info from {file:?}"));
            match serde_json::from_slice::<BuildInfo>(&content) {
                Ok(info) => Some(info),
                Err(_) => None,
            }
        } else {
            None
        }
    }

    /// Dumps the BuildInfo into out dir.
    fn to_out_dir(&self, dir: &PathBuf) {
        let file = dir.join("onnxruntime-build-info.json");
        fs::write(
            &file,
            serde_json::to_string_pretty(&self).expect("could not serialize build info"),
        )
        .expect(&format!("could not dump build info into {file:?}"));
    }

    fn dynlibs_exist(&self) -> bool {
        for file in &self.local_dynlib_filepaths {
            if !file_exists(file) {
                return false;
            }
        }
        true
    }

    /// Dumps the BuildInfo into out dir. This function must always be called
    /// with the `build_info.to_out_dir()` argument.
    pub fn write_build_info(&self) {
        let file = PathBuf::from(env::var("OUT_DIR").unwrap()).join("build_info.rs");
        fs::write(&file, self.to_string())
            .expect(&format!("could not dump build info into {file:?}"));
    }
}

/// builds onnxruntime from source using a C++ toolchain installed in the system,
/// and copies all the generated dynamic libs to `dir`.
///
/// # Arguments
///
/// * `dir`: the folder in which the onnxruntime project will be built.
/// * `accelerators`: all the accelerators with which the onnxruntime project will be compiled.
///
/// returns: all the information regarding the compilation artifacts.
pub fn build(
    dir: PathBuf,
    accelerators: Vec<Accelerators>,
) -> Result<BuildInfo, Box<dyn std::error::Error>> {
    let url = format!(
        "https://github.com/microsoft/onnxruntime/archive/refs/tags/v{ONNX_RELEASE}.tar.gz"
    );
    let name = format!("onnxruntime-{ONNX_RELEASE}");

    fs::create_dir_all(&dir)?;
    let dir = fs::canonicalize(&dir)?;
    let tar_gz = dir.join(format!("{name}.tar.gz"));
    let tar_gz = tar_gz.to_str().unwrap();

    let repo = dir.join(&name);
    let build_dir = repo.join("build");

    let mut cmd = if cfg!(target_os = "windows") {
        Command::new("cmd")
    } else {
        Command::new("./build.sh")
    };

    let mut cmd = if cfg!(target_os = "windows") {
        // I have no idea why I need to do this, but without this shit just
        // doesn't work.
        fn windows_fix_path(p: &PathBuf) -> String {
            p.to_str().unwrap_or_default().replace("\\\\?\\", "")
        }
        cmd.current_dir(windows_fix_path(&repo))
            .arg("/C")
            .arg("build.bat")
            .arg("--build_dir")
            .arg(windows_fix_path(&build_dir))
    } else {
        cmd.current_dir(&repo).arg("--build_dir").arg(&build_dir)
    };

    cmd.arg("--config")
        .arg(PROFILE)
        .arg("--build_shared_lib")
        .arg("--parallel")
        .arg("--compile_no_warning_as_error")
        .arg("--skip_submodule_sync")
        .arg("--skip_tests");

    let build_dir = build_dir.join(PROFILE);

    for accelerator in accelerators {
        match accelerator {
            Accelerators::COREML => cmd.arg("--use_coreml"),
            Accelerators::TENSORRT => cmd.arg("--use_tensorrt"),
            Accelerators::CUDA => cmd.arg("--use_cuda"),
        };
    }

    let cmd_str = format!("{:?}", cmd);

    if let Some(build_info) = BuildInfo::from_dir(&dir) {
        if build_info.dynlibs_exist() {
            if build_info.cmd != cmd_str {
                println!("All dynlib files exist, but the where build with a different command, recompiling...");
            } else {
                println!("All dynlib files already exist, nothing to do");
                return Ok(build_info);
            }
        }
    }

    if !file_exists(tar_gz) {
        println!("Downloading {url}...");
        download_file(&url, tar_gz)?;
    }
    if !dir_exists(&repo) {
        println!("Extracting {tar_gz}...");
        extract_tar_gz(tar_gz, dir.to_str().unwrap())?;
    }

    run_command_with_output(&mut cmd)?;

    let mut local_dynlib_filepaths = vec![];
    let mut dynlib_filenames = vec![];
    for file in fs::read_dir(&build_dir)? {
        let file = match file {
            Ok(v) => v.file_name().to_str().unwrap_or("").to_string(),
            Err(_) => continue,
        };

        if file.ends_with(DYN_LIB_EXT) {
            let src = build_dir.join(&file);
            let dst = dir.join(&file);
            fs::copy(&src, &dst)?;
            local_dynlib_filepaths.push(dst);
            dynlib_filenames.push(file.clone());
        }
    }
    let build_info = BuildInfo {
        local_dynlib_filepaths,
        onnxruntime_version: ONNX_RELEASE.to_string(),
        main_dynlib_filename: MAIN_DYNLIB_FILENAME.to_string(),
        dynlib_filenames,
        cmd: cmd_str,
    };
    build_info.to_out_dir(&dir);
    Ok(build_info)
}

fn download_file(url: &str, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Create a temporary directory
    let temp_dir = env::temp_dir();
    fs::create_dir_all(&temp_dir)?;
    let temp_file_path = temp_dir.join("temp_download_file");

    // Send the GET request
    let mut response = reqwest::blocking::get(url)?;
    if !response.status().is_success() {
        return Err(format!("Failed to download file: HTTP {}", response.status()).into());
    }

    // Get the content length, if available
    let total_size = response.content_length().unwrap_or_default();

    // Set up a progress bar
    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")?
            .progress_chars("#>-"),
    );

    // Write to a temporary file while updating progress
    let mut temp_file = File::create(&temp_file_path)?;
    let mut buffer = [0; 8192]; // 8 KB buffer
    let mut downloaded = 0;

    loop {
        let bytes_read = response.read(&mut buffer)?;
        if bytes_read == 0 {
            break; // EOF reached
        }

        temp_file.write_all(&buffer[..bytes_read])?;
        downloaded += bytes_read as u64;
        pb.set_position(downloaded);
    }

    // Finish the progress bar
    pb.finish_with_message("Download complete!\n");

    // Ensure the temporary file is closed before copying
    drop(temp_file);

    // Copy the downloaded file to the final output path
    let output_path = Path::new(output_path);
    fs::create_dir_all(output_path.parent().unwrap_or_else(|| Path::new("")))?;
    fs::copy(&temp_file_path, &output_path)?;

    // Optionally remove the temporary file (if needed)
    fs::remove_file(temp_file_path)?;

    Ok(())
}

fn extract_tar_gz(archive_path: &str, output_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Open the .tar.gz file
    let tar_gz = File::open(archive_path)?;
    let buf_reader = BufReader::new(tar_gz);

    // Create a GzDecoder to handle the .gz compression
    let gz_decoder = GzDecoder::new(buf_reader);

    // Create a Tar Archive to read the entries
    let mut archive = Archive::new(gz_decoder);

    // First pass: count the number of entries
    let entries_count = archive.entries()?.count();
    drop(archive); // Drop the archive to reset the iterator

    // Reopen the archive for extraction
    let tar_gz = File::open(archive_path)?;
    let buf_reader = BufReader::new(tar_gz);
    let gz_decoder = GzDecoder::new(buf_reader);
    let mut archive = Archive::new(gz_decoder);

    // Set up a progress bar
    let pb = ProgressBar::new(entries_count as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )?
            .progress_chars("#>-"),
    );

    // Extract each entry and update the progress bar
    for (index, entry) in archive.entries()?.enumerate() {
        let mut entry = entry?;
        entry.unpack_in(output_dir)?;
        pb.set_position((index + 1) as u64);
    }

    // Finish the progress bar
    pb.finish_with_message("Extraction complete!");

    Ok(())
}

fn run_command_with_output(cmd: &mut Command) -> std::io::Result<()> {
    // Configure the command to pipe stdout and stderr
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    // Spawn the command
    let mut child = cmd.spawn()?;

    // Get stdout and stderr streams
    let stdout = child.stdout.take().expect("Failed to capture stdout");
    let stderr = child.stderr.take().expect("Failed to capture stderr");

    // Spawn threads to handle stdout and stderr in real-time
    let stdout_thread = std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            if let Ok(line) = line {
                println!("{}", line); // Redirect to stdout
            }
        }
    });

    let stderr_thread = std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            if let Ok(line) = line {
                eprintln!("{}", line); // Redirect to stderr
            }
        }
    });

    // Wait for the command to finish
    let status = child.wait()?;

    // Wait for the threads to finish
    stdout_thread.join().expect("Failed to join stdout thread");
    stderr_thread.join().expect("Failed to join stderr thread");

    if !status.success() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Command failed with status: {}", status),
        ));
    }

    Ok(())
}

fn file_exists<P: AsRef<Path>>(path: P) -> bool {
    fs::metadata(path)
        .map(|metadata| metadata.is_file())
        .unwrap_or(false)
}

fn dir_exists<P: AsRef<Path>>(path: P) -> bool {
    fs::metadata(path)
        .map(|metadata| metadata.is_dir())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build() -> Result<(), Box<dyn std::error::Error>> {
        let mut p = PathBuf::from(env::current_dir()?);
        p.pop();
        p.push("target");
        build(p, vec![])?;
        Ok(())
    }
}
