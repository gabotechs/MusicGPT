#[derive(Debug)]
pub struct DelayedPatternMaskIds<const N: usize> {
    batches: [Vec<i64>; N],
}

impl<const N: usize> DelayedPatternMaskIds<N> {
    pub fn new() -> Self {
        assert!(N > 0, "N needs to be greater than 0");
        Self {
            batches: [(); N].map(|()| vec![]),
        }
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

    pub fn last_delayed_masked(&self, pad_token_id: i64) -> [i64; N] {
        // We want to apply the Ps to the last
        //   0 1 2 3 4 5 6 7 8 9 10
        // 0 x x x x x x x x x x ...
        // 1 P x x x x x x x x x ...
        // 2 P P x x x x x x x x ...
        // 3 P P P x x x x x x x ...
        let seq_len = self.batches[0].len();
        let mut result = [0; N];
        for (i, item) in result.iter_mut().enumerate() {
            if (seq_len as i64 - i as i64) <= 0 {
                *item = pad_token_id
            } else {
                *item = *self.batches[i].last().expect("There are no input_ids");
            }
        }
        result
    }

    pub fn last_de_delayed(&self) -> Option<[i64; N]> {
        // We want to gather the last diagonal set of numbers avoiding Ps
        // (e.g. [(0,0), (1,1), (2,2), (3,3)])
        //   0 1 2 3 4 5 6 7 8 9
        // 0 x x x x x x x P P P
        // 1 P x x x x x x x P P
        // 2 P P x x x x x x x P
        // 3 P P P x x x x x x x
        if self.batches[0].len() < N {
            return None;
        }
        let mut result = [0; N];
        for (i, item) in result.iter_mut().enumerate() {
            *item = self.batches[i][self.batches[i].len() - N + i]
        }
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn last_delayed_masked() {
        let mut input_ids = DelayedPatternMaskIds::<4>::new();
        assert_eq!(input_ids.last_delayed_masked(0), [0, 0, 0, 0]);
        input_ids.push([1, 2, 3, 4]);
        assert_eq!(input_ids.last_delayed_masked(0), [1, 0, 0, 0]);
        input_ids.push([5, 6, 7, 8]);
        assert_eq!(input_ids.last_delayed_masked(0), [5, 6, 0, 0]);
        input_ids.push([9, 10, 11, 12]);
        assert_eq!(input_ids.last_delayed_masked(0), [9, 10, 11, 0]);
        input_ids.push([13, 14, 15, 16]);
        assert_eq!(input_ids.last_delayed_masked(0), [13, 14, 15, 16]);
        input_ids.push([17, 18, 19, 20]);
        assert_eq!(input_ids.last_delayed_masked(0), [17, 18, 19, 20]);
    }

    #[test]
    fn last_de_delayed() {
        let mut input_ids = DelayedPatternMaskIds::<4>::new();
        assert_eq!(input_ids.last_de_delayed(), None);
        input_ids.push([1, 2, 3, 4]);
        assert_eq!(input_ids.last_de_delayed(), None);
        input_ids.push([5, 6, 7, 8]);
        assert_eq!(input_ids.last_de_delayed(), None);
        input_ids.push([9, 10, 11, 12]);
        assert_eq!(input_ids.last_de_delayed(), None);
        input_ids.push([13, 14, 15, 16]);
        assert_eq!(input_ids.last_de_delayed(), Some([1, 6, 11, 16]));
        input_ids.push([17, 18, 19, 20]);
        assert_eq!(input_ids.last_de_delayed(), Some([5, 10, 15, 20]));
    }
}
