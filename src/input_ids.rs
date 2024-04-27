use ndarray::{Array1, Array2};
use std::ops::Index;

use crate::tensor::Tensor;

#[derive(Debug)]
pub struct InputIds<const N: usize> {
    batches: [Vec<i64>; N],
}

impl<const N: usize> Index<(usize, usize)> for InputIds<N> {
    type Output = i64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.batches[index.0][index.1]
    }
}

impl<const N: usize> Index<usize> for InputIds<N> {
    type Output = i64;

    fn index(&self, index: usize) -> &Self::Output {
        let (_, seq_length) = self.dims();
        let row = index / seq_length;
        let column = index % seq_length;
        &self.batches[row][column]
    }
}

impl<const N: usize> InputIds<N> {
    pub fn new() -> Self {
        assert!(N > 0, "N needs to be greater than 0");
        Self {
            batches: [(); N].map(|()| vec![]),
        }
    }

    pub fn dims(&self) -> (usize, usize) {
        (N, self.batches[0].len())
    }

    pub fn push(&mut self, token_ids: impl IntoIterator<Item = i64>) {
        let mut i = 0;
        for token_id in token_ids.into_iter() {
            assert!(i < N, "Expected exactly {N} token_ids");
            self.batches[i].push(token_id);
            i += 1;
        }
        assert_eq!(i, N, "Expected exactly {N} token_ids");
    }

    pub fn apply_delay_pattern_mask(&self, pad_token_id: i64) -> Self {
        // TODO: copying the whole array each time is not efficient.
        let mut batches = self.batches.clone();
        for i in 0..batches.len() {
            for j in 0..batches[0].len() {
                if i % N >= j {
                    batches[i][j] = pad_token_id;
                }
            }
        }
        Self { batches }
    }

    pub fn last(&self) -> Tensor<i64> {
        let mut last = Vec::with_capacity(N);
        for i in 0..N {
            last.push(*self.batches[i].last().expect("There are no input_ids"))
        }

        Tensor::from_array(Array1::from(last).into_dyn())
    }

    pub fn last_raw(&self) -> [i64; N] {
        let mut result = [0; N];
        for i in 0..N {
            result[i] = *self.batches[i].last().expect("There are no input_ids");
        }

        result
    }

    pub fn last_de_delayed(&self) -> Option<[i64; N]> {
        // We want to trim away the Ps
        //   0 1 2 3 4 5 6 7 8 9
        // 0 P x x x x x x P P P
        // 1 P P x x x x x x P P
        // 2 P P P x x x x x x P
        // 3 P P P P x x x x x x
        if self.batches[0].len() < N {
            return None;
        }
        let mut result = [0; N];
        for i in 0..N {
            result[i] = self.batches[i][self.batches[i].len() - N + i]
        }
        Some(result)
    }

    pub fn tensor(&self) -> Tensor<i64> {
        let mut v = Vec::with_capacity(self.batches[0].len() * N);
        for batch in self.batches.iter() {
            for n in batch {
                v.push(*n)
            }
        }
        let arr = Array2::from_shape_vec((N, self.batches[0].len()), v)
            .expect("Could not build bi-dimensional array from raw array");
        Tensor::from_array(arr.into_dyn())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Ix1, Ix2};

    #[test]
    fn convert_to_tensor() {
        let mut input_ids = InputIds::<4>::new();
        input_ids.push([1, 2, 3, 4]);
        input_ids.push([5, 6, 7, 8]);
        input_ids.push([9, 10, 11, 12]);
        let tensor = input_ids.tensor();
        assert_eq!(tensor.ndim(), 2);
        let arr = tensor.into_inner().into_dimensionality::<Ix2>().unwrap();
        assert_eq!(arr[(0, 0)], 1);
        assert_eq!(arr[(0, 1)], 5);
        assert_eq!(arr[(0, 2)], 9);
        assert_eq!(arr[(1, 0)], 2);
        assert_eq!(arr[(1, 1)], 6);
        assert_eq!(arr[(1, 2)], 10);
        assert_eq!(arr[(2, 0)], 3);
        assert_eq!(arr[(2, 1)], 7);
        assert_eq!(arr[(2, 2)], 11);
        assert_eq!(arr[(3, 0)], 4);
        assert_eq!(arr[(3, 1)], 8);
        assert_eq!(arr[(3, 2)], 12);
    }

    #[test]
    fn last() {
        let mut input_ids = InputIds::<4>::new();
        input_ids.push([1, 2, 3, 4]);
        input_ids.push([5, 6, 7, 8]);
        input_ids.push([9, 10, 11, 12]);
        let last = input_ids.last();
        let arr = last.into_inner().into_dimensionality::<Ix1>().unwrap();
        assert_eq!(arr[0], 9);
        assert_eq!(arr[1], 10);
        assert_eq!(arr[2], 11);
        assert_eq!(arr[3], 12);
    }

    #[test]
    fn index() {
        let mut input_ids = InputIds::<4>::new();
        input_ids.push([1, 2, 3, 4]);
        input_ids.push([5, 6, 7, 8]);
        input_ids.push([9, 10, 11, 12]);
        // 2D indexing
        assert_eq!(input_ids[(1, 2)], 10);

        // 1D indexing
        assert_eq!(input_ids[0], 1);
        assert_eq!(input_ids[1], 5);
        assert_eq!(input_ids[2], 9);
        assert_eq!(input_ids[3], 2);
        assert_eq!(input_ids[4], 6);
        assert_eq!(input_ids[5], 10);
        assert_eq!(input_ids[6], 3);
        assert_eq!(input_ids[7], 7);
        assert_eq!(input_ids[8], 11);
        assert_eq!(input_ids[9], 4);
        assert_eq!(input_ids[10], 8);
        assert_eq!(input_ids[11], 12);
    }
}
