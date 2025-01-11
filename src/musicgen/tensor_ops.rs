use ndarray::Array;
use num_traits::{One, Zero};
use ort::tensor::PrimitiveTensorElementType;
use ort::value::Tensor;
use std::fmt::Debug;

pub fn zeros_tensor<T: PrimitiveTensorElementType + Debug + Clone + Zero + 'static>(
    shape: &[usize],
) -> Tensor<T> {
    ort::value::Value::from_array(Array::<T, _>::zeros(shape))
        .expect("Could not build zeros tensor")
}

pub fn dupe_zeros_along_first_dim<
    T: PrimitiveTensorElementType + Debug + Zero + Clone + 'static,
>(
    tensor: Tensor<T>,
) -> ort::Result<Tensor<T>> {
    let (shape, data) = tensor.try_extract_raw_tensor()?;
    let mut shape = shape.to_vec();
    shape[0] *= 2;
    let data = [data.to_vec(), vec![T::zero(); data.len()]].concat();
    Tensor::from_array((shape, data))
}

pub fn ones_tensor<T: PrimitiveTensorElementType + Debug + Clone + One + 'static>(
    shape: &[usize],
) -> Tensor<T> {
    ort::value::Value::from_array(Array::<T, _>::ones(shape)).expect("Could not build zeros tensor")
}
