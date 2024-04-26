use crate::tensor::Tensor;
use ndarray::ArrayView1;
use std::collections::VecDeque;

#[derive(Debug)]
pub struct InputIds<const N: usize> {
    entries: VecDeque<[i64; N]>,
}

impl<const N: usize> InputIds<N> {
    pub fn new() -> Self {
        Self {
            entries: VecDeque::new(),
        }
    }

    pub fn push(&mut self, token_ids: impl IntoIterator<Item = i64>) {
        let mut tokens = [0; N];
        let mut i = 0;
        for token_id in token_ids.into_iter() {
            if i >= N {
                panic!("Expected exactly {N} token_ids")
            }
            tokens[i] = token_id;
            i += 1;
        }
        if i != N {
            panic!("Expected exactly {N} token_ids")
        }
        self.entries.push_back(tokens)
    }

    pub fn apply_delay_pattern_mask(&self, pad_token_id: i64) -> Self {
        let mut entries = self.entries.clone();
        for i in 0..N {
            for j in 0..entries.len() {
                if i % N >= j {
                    entries[j][i] = pad_token_id;
                }
            }
        }
        Self { entries }
    }

    pub fn last(&self) -> Tensor<i64> {
        let last = self
            .entries
            .iter()
            .last()
            .expect("There's no entries in input_ids");
        Tensor::from_array(ArrayView1::from(&last).into_dyn().to_owned())
    }
}
