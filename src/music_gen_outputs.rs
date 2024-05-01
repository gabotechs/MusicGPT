use crate::logits::Logits;

pub struct MusicGenOutputs<'s> {
    outputs: ort::SessionOutputs<'s>,
}

impl<'s> MusicGenOutputs<'s> {
    pub fn new(outputs: ort::SessionOutputs<'s>) -> Self {
        Self { outputs }
    }

    pub fn take_logits(&mut self) -> ort::Result<Logits> {
        Logits::from_3d_dyn_value(&self.outputs.remove("logits").unwrap())
    }

    pub fn take_present_decoder_key(&mut self, i: usize) -> ort::DynValue {
        let key = format!("present.{i}.decoder.key");
        self.outputs
            .remove(key.as_str())
            .expect(&format!("{key} was already taken from outputs"))
    }

    pub fn take_present_decoder_value(&mut self, i: usize) -> ort::DynValue {
        let value = format!("present.{i}.decoder.value");
        self.outputs
            .remove(value.as_str())
            .expect(&format!("{value} was already taken from outputs"))
    }

    pub fn take_present_encoder_key(&mut self, i: usize) -> ort::DynValue {
        let key = format!("present.{i}.encoder.key");
        self.outputs
            .remove(key.as_str())
            .expect(&format!("{key} was already taken from outputs"))
    }

    pub fn take_present_encoder_value(&mut self, i: usize) -> ort::DynValue {
        let value = format!("present.{i}.encoder.value");
        self.outputs
            .remove(value.as_str())
            .expect(&format!("{value} was already taken from outputs"))
    }
}
