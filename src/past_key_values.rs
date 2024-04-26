use ndarray::{Array, ArrayViewD};

use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct PastKeyValuesConfig {
    pub num_encoder_heads: usize,
    pub num_decoder_heads: usize,
    pub encoder_dim_kv: usize,
    pub decoder_dim_kv: usize,
    pub num_decoder_layers: usize,
}

pub struct PastKeyValues {
    values: Vec<[Tensor<f32>; 4]>,
    is_empty: bool,
}

impl PastKeyValues {
    pub fn empty(config: PastKeyValuesConfig) -> Self {
        let encoder_dims = (1, config.num_decoder_heads, 0, config.encoder_dim_kv);
        let decoder_dims = (1, config.num_decoder_heads, 0, config.decoder_dim_kv);
        let mut values = Vec::with_capacity(config.num_decoder_layers);
        for _ in 0..config.num_decoder_layers {
            values.push([
                // "past_key_values.0.decoder.key"
                // "past_key_values.0.decoder.value"
                // "past_key_values.0.encoder.key"
                // "past_key_values.0.encoder.value"
                Tensor::from_array(Array::<f32, _>::zeros(decoder_dims)),
                Tensor::from_array(Array::<f32, _>::zeros(decoder_dims)),
                Tensor::from_array(Array::<f32, _>::zeros(encoder_dims)),
                Tensor::from_array(Array::<f32, _>::zeros(encoder_dims)),
            ])
        }
        Self {
            values,
            is_empty: true,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.is_empty
    }

    pub fn view_all(&self) -> impl IntoIterator<Item = ArrayViewD<f32>> {
        self.values
            .iter()
            .flat_map(|entries| entries.iter().map(|entry| entry.view()))
    }

    pub fn set(&mut self, index: usize, entry: usize, tensor: Tensor<f32>) {
        self.is_empty = false;
        self.values[index][entry] = tensor
    }
}
