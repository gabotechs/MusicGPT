use crate::musicgen::logits::Logits;
use ort::session::SessionOutputs;
use ort::value::DynValue;

pub struct MusicGenOutputs<'r, 's> {
    outputs: SessionOutputs<'r, 's>,
}

impl<'r, 's> MusicGenOutputs<'r, 's> {
    pub fn new(outputs: SessionOutputs<'r, 's>) -> Self {
        Self { outputs }
    }

    pub fn take_logits(&mut self) -> ort::Result<Logits> {
        Logits::from_3d_dyn_value(&self.outputs.remove("logits").unwrap())
    }

    pub fn take_present_decoder_key(&mut self, i: usize) -> DynValue {
        let key = format!("present.{i}.decoder.key");
        self.outputs
            .remove(key.as_str())
            .unwrap_or_else(|| panic!("{key} was already taken from outputs"))
    }

    pub fn take_present_decoder_value(&mut self, i: usize) -> DynValue {
        let value = format!("present.{i}.decoder.value");
        self.outputs
            .remove(value.as_str())
            .unwrap_or_else(|| panic!("{value} was already taken from outputs"))
    }

    pub fn take_present_encoder_key(&mut self, i: usize) -> DynValue {
        let key = format!("present.{i}.encoder.key");
        self.outputs
            .remove(key.as_str())
            .unwrap_or_else(|| panic!("{key} was already taken from outputs"))
    }

    pub fn take_present_encoder_value(&mut self, i: usize) -> DynValue {
        let value = format!("present.{i}.encoder.value");
        self.outputs
            .remove(value.as_str())
            .unwrap_or_else(|| panic!("{value} was already taken from outputs"))
    }
}
