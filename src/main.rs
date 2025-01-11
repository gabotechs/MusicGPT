mod onnxruntime_lib;
mod audio;
mod backend;
mod cli;
mod musicgen;
mod storage;
mod terminal;
mod musicgen_models;
mod gpu;
mod storage_ext;

use log::error;
use std::process::exit;
use directories::ProjectDirs;
use lazy_static::lazy_static;
use tracing_subscriber::fmt::time::UtcTime;
use tracing_subscriber::{fmt, EnvFilter};

use crate::storage::AppFs;

#[tokio::main]
async fn main() {
    let time_format = time::format_description::parse(
        "[year]-[month]-[day] [hour]:[minute]:[second].[subsecond digits:3]",
    )
    .expect("Failed to create timestamp format");
    let format = fmt::format()
        .with_target(false)
        .with_timer(UtcTime::new(time_format));
    let filter = EnvFilter::new("info,ort=off");

    tracing_subscriber::fmt()
        .event_format(format)
        .with_max_level(tracing::Level::INFO)
        .with_env_filter(filter)
        .init();
    if let Err(err) = cli::cli(&PROJECT_FS.root, PROJECT_FS.clone()).await {
        error!("{err}");
        exit(1)
    }
}

lazy_static! {
    static ref PROJECT_FS: AppFs = AppFs::new(
        ProjectDirs::from("com", "gabotechs", "musicgpt")
            .expect("Could not load project directory")
            .data_dir()
    );
}
