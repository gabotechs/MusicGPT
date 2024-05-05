use std::fmt::Write;
use std::ops::{Deref, DerefMut};
use std::time::Duration;

use indicatif::{MultiProgress, ProgressBar, ProgressFinish, ProgressState, ProgressStyle};

pub struct LoadingBarFactor;

pub struct Bar(ProgressBar);

impl Bar {
    pub fn update_elapsed_total(&self, elapsed: usize, total: usize) {
        self.0.set_length(total as u64);
        self.0.set_position(elapsed as u64);
    }
}

impl Deref for Bar {
    type Target = ProgressBar;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Bar {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub struct MultiBar(MultiProgress);

impl MultiBar {
    pub fn add(&self, bar: Bar) -> Bar {
        Bar(self.0.add(bar.0))
    }
}

impl Deref for MultiBar {
    type Target = MultiProgress;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for MultiBar {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl LoadingBarFactor {
    pub fn multi() -> MultiBar {
        MultiBar(MultiProgress::new())
    }

    pub fn spinner(msg: &str) -> Bar {
        let pb = ProgressBar::new_spinner();
        pb.enable_steady_tick(Duration::from_millis(120));
        pb.set_style(ProgressStyle::with_template("{spinner:.blue} {msg}").unwrap());
        pb.set_message(msg.to_string());
        Bar(pb)
    }

    pub fn bar(prefix: &str) -> Bar {
        Self::fixed_bar(prefix, 1)
    }

    pub fn fixed_bar(prefix: &str, len: usize) -> Bar {
        let pb = ProgressBar::new(len as u64);
        pb.set_style(
            ProgressStyle::with_template(
                &(prefix.to_string()
                    + " {spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] ({eta})"),
            )
            .unwrap()
            .with_key("eta", |state: &ProgressState, w: &mut dyn Write| {
                write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap()
            })
            .progress_chars("#>-"),
        );
        Bar(pb)
    }

    pub fn download_bar(file: &str) -> Bar {
        let pb = ProgressBar::new(1);
        pb.set_style(
            ProgressStyle::with_template(
                &format!("{file} {{spinner:.green}} [{{wide_bar:.cyan/blue}}] {{bytes}}/{{total_bytes}} ({{bytes_per_sec}}, {{eta}})")
            )
                .unwrap()
                .with_key("eta", |state: &ProgressState, w: &mut dyn Write| {
                    write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap()
                })
                .progress_chars("#>-"),
        );
        let pb = pb.with_finish(ProgressFinish::WithMessage(
            format!("{file} downloaded").into(),
        ));
        Bar(pb)
    }
}
