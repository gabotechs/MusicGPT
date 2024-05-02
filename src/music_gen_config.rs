use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct MusicGenConfig {
    pub audio_encoder: AudioEncoderConfig,
    pub decoder: DecoderConfig,
    pub text_encoder: TextEncoderConfig,
}

#[derive(Serialize, Deserialize)]
pub struct AudioEncoderConfig {
    pub sampling_rate: usize,
}

#[derive(Serialize, Deserialize)]
pub struct DecoderConfig {
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub top_k: usize,
    pub pad_token_id: i64,
}

#[derive(Serialize, Deserialize)]
pub struct TextEncoderConfig {
    pub d_kv: usize,
}
