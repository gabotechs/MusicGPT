use std::fmt::{Debug, Formatter};
use std::ops::{Deref, DerefMut};

use ndarray::{concatenate, Array, Axis, IxDyn, RemoveAxis};
use num_traits::Zero;

pub trait TensorData: Clone + Debug + ort::IntoTensorElementType {}

impl TensorData for f32 {}
impl TensorData for f64 {}
impl TensorData for u8 {}
impl TensorData for u16 {}
impl TensorData for u32 {}
impl TensorData for u64 {}
impl TensorData for i8 {}
impl TensorData for i16 {}
impl TensorData for i32 {}
impl TensorData for i64 {}
impl TensorData for bool {}

pub struct Tensor<T: TensorData>(Array<T, IxDyn>);

impl<T: TensorData> TryFrom<ort::DynValue> for Tensor<T> {
    type Error = ort::Error;

    fn try_from(value: ort::DynValue) -> Result<Self, Self::Error> {
        Ok(Self(value.try_extract_tensor::<T>()?.into_owned()))
    }
}

fn vec_to_arr<T, const N: usize>(v: Vec<T>) -> [T; N] {
    v.try_into()
        .unwrap_or_else(|v: Vec<T>| panic!("Expected a Vec of length {} but it was {}", N, v.len()))
}

impl<T: TensorData> TryFrom<&ort::DynValue> for Tensor<T> {
    type Error = ort::Error;

    fn try_from(value: &ort::DynValue) -> Result<Self, Self::Error> {
        let mut empty = false;
        let shape = value
            .shape()
            .unwrap_or_default()
            .into_iter()
            .map(|e| e as usize)
            .collect::<Vec<_>>();
        for dim in shape.iter() {
            if dim == &0 {
                empty = true;
                break;
            }
        }
        // https://github.com/pykeio/ort/issues/185
        if empty {
            let arr = match shape.len() {
                0 => Array::from_shape_vec(vec_to_arr::<usize, 0>(shape), vec![])
                    .unwrap()
                    .into_dyn(),
                1 => Array::from_shape_vec(vec_to_arr::<usize, 1>(shape), vec![])
                    .unwrap()
                    .into_dyn(),
                2 => Array::from_shape_vec(vec_to_arr::<usize, 2>(shape), vec![])
                    .unwrap()
                    .into_dyn(),
                3 => Array::from_shape_vec(vec_to_arr::<usize, 3>(shape), vec![])
                    .unwrap()
                    .into_dyn(),
                4 => Array::from_shape_vec(vec_to_arr::<usize, 4>(shape), vec![])
                    .unwrap()
                    .into_dyn(),
                5 => Array::from_shape_vec(vec_to_arr::<usize, 5>(shape), vec![])
                    .unwrap()
                    .into_dyn(),
                6 => Array::from_shape_vec(vec_to_arr::<usize, 6>(shape), vec![])
                    .unwrap()
                    .into_dyn(),
                d => return Err(ort::Error::InvalidDimension(d)),
            };
            Ok(Self(arr))
        } else {
            Ok(Self(value.try_extract_tensor::<T>()?.into_owned()))
        }
    }
}

impl<T: TensorData> Deref for Tensor<T> {
    type Target = Array<T, IxDyn>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: TensorData> DerefMut for Tensor<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: TensorData> Debug for Tensor<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl<T: TensorData> Tensor<T> {
    pub fn from_array<D: RemoveAxis>(value: Array<T, D>) -> Self {
        Self(value.into_dyn())
    }

    pub fn dupe_along_first_dim(&self) -> Self {
        Tensor::from_array(concatenate(Axis(0), &[self.0.view(), self.0.view()]).unwrap())
    }

    pub fn squeeze(self, dim: i64) -> Self {
        let dim = if dim < 0 {
            (self.0.ndim() as i64 + dim) as usize
        } else {
            dim as usize
        };
        Self(self.0.remove_axis(Axis(dim)))
    }

    pub fn unsqueeze(mut self, dim: usize) -> Self {
        self.0.insert_axis_inplace(Axis(dim));
        Self(self.0)
    }

    pub fn into_inner(self) -> Array<T, IxDyn> {
        self.0
    }
}

impl Tensor<bool> {
    pub fn bool(v: bool) -> Self {
        Tensor::from_array(Array::<bool, _>::from_vec(vec![v]))
    }
}

impl<T: TensorData + Zero> Tensor<T> {
    pub fn dupe_zeros_along_first_dim(&self) -> Self {
        Tensor::from_array(
            concatenate(
                Axis(0),
                &[self.0.view(), Array::zeros(self.0.shape()).view()],
            )
            .unwrap(),
        )
    }
}
