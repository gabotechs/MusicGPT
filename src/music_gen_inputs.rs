use std::collections::HashMap;

pub struct MusicGenInputs {
    inputs: HashMap<String, ort::DynValue>,
    pub use_cache_branch: bool,
}

impl MusicGenInputs {
    pub fn new() -> Self {
        Self {
            inputs: HashMap::new(),
            use_cache_branch: false,
        }
    }
    pub fn encoder_attention_mask<T, E>(&mut self, v: T) -> Result<(), E>
    where
        ort::DynValue: TryFrom<T, Error = E>,
    {
        self.inputs
            .insert("encoder_attention_mask".to_string(), v.try_into()?);
        Ok(())
    }

    pub fn input_ids<T, E>(&mut self, v: T) -> Result<(), E>
    where
        ort::DynValue: TryFrom<T, Error = E>,
    {
        self.inputs.insert("input_ids".to_string(), v.try_into()?);
        Ok(())
    }

    pub fn encoder_hidden_states<T, E>(&mut self, v: T) -> Result<(), E>
    where
        ort::DynValue: TryFrom<T, Error = E>,
    {
        self.inputs
            .insert("encoder_hidden_states".to_string(), v.try_into()?);
        Ok(())
    }

    pub fn past_key_value_decoder_key<T, E>(&mut self, i: usize, v: T) -> Result<(), E>
    where
        ort::DynValue: TryFrom<T, Error = E>,
    {
        self.inputs
            .insert(format!("past_key_values.{i}.decoder.key"), v.try_into()?);
        Ok(())
    }

    pub fn past_key_value_decoder_value<T, E>(&mut self, i: usize, v: T) -> Result<(), E>
    where
        ort::DynValue: TryFrom<T, Error = E>,
    {
        self.inputs
            .insert(format!("past_key_values.{i}.decoder.value"), v.try_into()?);
        Ok(())
    }

    pub fn past_key_value_encoder_key<T, E>(&mut self, i: usize, v: T) -> Result<(), E>
    where
        ort::DynValue: TryFrom<T, Error = E>,
    {
        self.inputs
            .insert(format!("past_key_values.{i}.encoder.key"), v.try_into()?);
        Ok(())
    }

    pub fn past_key_value_encoder_value<T, E>(&mut self, i: usize, v: T) -> Result<(), E>
    where
        ort::DynValue: TryFrom<T, Error = E>,
    {
        self.inputs
            .insert(format!("past_key_values.{i}.encoder.value"), v.try_into()?);
        Ok(())
    }

    pub fn use_cache_branch(&mut self, value: bool) {
        self.use_cache_branch = value;
        self.inputs.insert(
            "use_cache_branch".to_string(),
            ort::Tensor::from_array(([1], vec![value]))
                .unwrap()
                .into_dyn(),
        );
    }

    pub fn ort(&self) -> ort::SessionInputs {
        ort::SessionInputs::ValueMap(
            self.inputs
                .iter()
                .map(|e| (e.0.to_string().into(), e.1.view().into()))
                .collect::<Vec<_>>(),
        )
    }
}
