use log::error;
use std::process::exit;
use tracing_subscriber::fmt::time::UtcTime;
use tracing_subscriber::{fmt, EnvFilter};

mod audio;
mod backend;
mod cli;
mod musicgen;
mod storage;
mod terminal;

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
    if let Err(err) = cli::cli().await {
        error!("{err}");
        exit(1)
    }
}
