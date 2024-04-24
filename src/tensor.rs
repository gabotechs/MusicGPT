use std::fmt::{Debug, Formatter};
use std::ops::{Deref, DerefMut};

use ndarray::{concatenate, Array, Axis, Dimension, IxDyn, RemoveAxis, ShapeBuilder};
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

impl<T: TensorData> TryFrom<&ort::DynValue> for Tensor<T> {
    type Error = ort::Error;

    fn try_from(value: &ort::DynValue) -> Result<Self, Self::Error> {
        Ok(Self(value.try_extract_tensor::<T>()?.into_owned()))
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

    pub fn from_value<D: Dimension, Sh: ShapeBuilder<Dim = D>>(shape: Sh, value: T) -> Self {
        let arr = Array::from_shape_simple_fn(shape, || value.clone());
        Self(arr.into_dyn())
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
