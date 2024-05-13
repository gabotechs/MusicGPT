use crate::music_gen_audio_encodec::MusicGenAudioEncodec;
use crate::music_gen_decoder::MusicGenDecoder;
use crate::music_gen_text_encoder::MusicGenTextEncoder;
use crate::ui::backend_ai::{BackendAi, BackendAiInboundMsg, BackendAiOutboundMsg, MusicGenJobProcessor};
use crate::ui::backend_audio::{BackendAudio, BackendAudioInboundMsg, BackendAudioOutboundMsg};
use anyhow::Context;
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::prelude::*;
use std::io;
use std::io::Stdout;
use std::time::{Duration, Instant};
use crossterm::event::{KeyCode, KeyEventKind};

pub struct App {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    audio_tx: tokio::sync::mpsc::Sender<BackendAudioInboundMsg>,
    audio_rx: tokio::sync::mpsc::Receiver<BackendAudioOutboundMsg>,
    ai_tx: std::sync::mpsc::Sender<BackendAiInboundMsg>,
    ai_rx: std::sync::mpsc::Receiver<BackendAiOutboundMsg>,
}

impl App {
    pub fn try_from_music_gen(
        text_encoder: MusicGenTextEncoder,
        decoder: Box<dyn MusicGenDecoder>,
        audio_encodec: MusicGenAudioEncodec,
    ) -> anyhow::Result<Self> {
        let mut stdout = io::stdout();
        enable_raw_mode().context("failed to enable raw mode")?;
        execute!(stdout, EnterAlternateScreen).context("unable to enter alternate screen")?;
        let terminal =
            Terminal::new(CrosstermBackend::new(stdout)).context("creating terminal failed")?;

        let backend_ai = BackendAi::new(MusicGenJobProcessor {
            text_encoder,
            decoder,
            audio_encodec,
        });
        let backend_audio = BackendAudio::default();
        let (ai_tx, ai_rx) = backend_ai.run();
        let (audio_tx, audio_rx) = backend_audio.run();

        Ok(Self {
            terminal ,
            ai_tx,
            ai_rx,
            audio_tx,
            audio_rx
        })
    }

    pub fn run(mut self) -> anyhow::Result<()> {
        let mut last_tick = Instant::now();
        loop {
            self.terminal.draw(|frame| {})?;

            if let crossterm::event::Event::Key(key) = crossterm::event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Backspace => {}
                        KeyCode::Enter => {}
                        KeyCode::Left => {}
                        KeyCode::Right => {}
                        KeyCode::Up => {}
                        KeyCode::Down => {}
                        KeyCode::Home => {}
                        KeyCode::End => {}
                        KeyCode::PageUp => {}
                        KeyCode::PageDown => {}
                        KeyCode::Tab => {}
                        KeyCode::BackTab => {}
                        KeyCode::Delete => {}
                        KeyCode::Insert => {}
                        KeyCode::F(_) => {}
                        KeyCode::Char(_) => {}
                        KeyCode::Null => {}
                        KeyCode::Esc => {}
                        KeyCode::CapsLock => {}
                        KeyCode::ScrollLock => {}
                        KeyCode::NumLock => {}
                        KeyCode::PrintScreen => {}
                        KeyCode::Pause => {}
                        KeyCode::Menu => {}
                        KeyCode::KeypadBegin => {}
                        KeyCode::Media(_) => {}
                        KeyCode::Modifier(j) => {}
                    }
                }
            }
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        let _ = disable_raw_mode().context("failed to disable raw mode");
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen)
            .context("unable to switch to main screen");
        let _ = self.terminal.show_cursor().context("unable to show cursor");
    }
}
