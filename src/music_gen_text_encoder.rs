use tokenizers::Tokenizer;

use crate::tensor_ops::ones_tensor;

pub struct MusicGenTextEncoder {
    pub tokenizer: Tokenizer,
    pub text_encoder: ort::Session,
}

impl MusicGenTextEncoder {
    pub fn encode(&self, text: &str) -> ort::Result<(ort::DynValue, ort::DynValue)> {
        let tokens = self
            .tokenizer
            .encode(text, true)
            .expect("Error tokenizing text")
            .get_ids()
            .iter()
            .map(|e| *e as i64)
            .collect::<Vec<_>>();

        let tokens_len = tokens.len();
        let input_ids = ort::Tensor::from_array(([1, tokens_len], tokens))?;
        let attention_mask = ones_tensor::<i64>(&[1, tokens_len]);

        let mut output = self
            .text_encoder
            .run(ort::inputs![input_ids, attention_mask]?)?;

        let last_hidden_state = output
            .remove("last_hidden_state")
            .expect("last_hidden_state not found in output");

        Ok((last_hidden_state, ones_tensor::<i64>(&[1, tokens_len]).into_dyn()))
    }
}

