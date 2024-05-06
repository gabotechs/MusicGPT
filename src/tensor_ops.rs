use std::fmt::Debug;
use ndarray::Array;
use num_traits::{One, Zero};

pub fn zeros_tensor<T: ort::IntoTensorElementType + Debug + Clone + Zero + 'static>(
    shape: &[usize],
) -> ort::Tensor<T> {
    ort::Value::from_array(Array::<T, _>::zeros(shape)).expect("Could not build zeros tensor")
}

pub fn dupe_zeros_along_first_dim<T: ort::IntoTensorElementType + Debug + Zero + Clone + 'static>(
    tensor: ort::Tensor<T>,
) -> ort::Result<ort::Tensor<T>> {
    let (mut shape, data) = tensor.try_extract_raw_tensor()?;
    shape[0] *= 2;
    let data = [data.to_vec(), vec![T::zero(); data.len()]].concat();
    ort::Tensor::from_array((shape, data))
}

pub fn ones_tensor<T: ort::IntoTensorElementType + Debug + Clone + One + 'static>(
    shape: &[usize],
) -> ort::Tensor<T> {
    ort::Value::from_array(Array::<T, _>::ones(shape)).expect("Could not build zeros tensor")
}
