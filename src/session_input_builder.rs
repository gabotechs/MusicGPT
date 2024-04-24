pub struct SessionInputsBuilder(Vec<ort::SessionInputValue<'static>>);

impl SessionInputsBuilder {
    pub fn start() -> Self {
        Self(vec![])
    }

    pub fn add<T>(mut self, value: T) -> Self
    where
        ort::DynValue: TryFrom<T, Error = ort::Error>,
    {
        let dyn_value: ort::DynValue = value
            .try_into()
            .expect("Could not convert value to ort::DynValue");
        self.0.push(dyn_value.into());
        self
    }

    pub fn add_many<T>(mut self, values: impl IntoIterator<Item = T>) -> Self
    where
        ort::DynValue: TryFrom<T, Error = ort::Error>,
    {
        for value in values {
            self = self.add(value)
        }
        self
    }

    pub fn end(&self) -> ort::SessionInputs {
        ort::SessionInputs::from(self.0.as_slice())
    }
}
