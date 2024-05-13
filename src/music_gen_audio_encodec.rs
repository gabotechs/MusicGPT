use std::collections::VecDeque;

use half::f16;
use ndarray::{Array, Axis};

pub struct MusicGenAudioEncodec {
    pub audio_encodec_decode: ort::Session,
}

impl MusicGenAudioEncodec {
    pub fn encode(&self, tokens: impl IntoIterator<Item = [i64; 4]>) -> ort::Result<VecDeque<f32>> {
        let mut data = vec![];
        for ids in tokens {
            for id in ids {
                data.push(id)
            }
        }

        let seq_len = data.len() / 4;
        let arr = Array::from_shape_vec((seq_len, 4), data).expect("Programming error");
        let arr = arr.t().insert_axis(Axis(0)).insert_axis(Axis(0));
        let mut outputs = self.audio_encodec_decode.run(ort::inputs![arr]?)?;
        let audio_values: ort::DynValue = outputs
            .remove("audio_values")
            .expect("audio_values not found in output");

        if let Ok((_, data)) = audio_values.try_extract_raw_tensor::<f32>() {
            return Ok(data.into_iter().map(|e| *e).collect());
        }
        if let Ok((_, data)) = audio_values.try_extract_raw_tensor::<f16>() {
            return Ok(data.into_iter().map(|e| f32::from(*e)).collect());
        }

        return Err(ort::Error::CustomError(
            "Token stream must be either f16 or f32".into(),
        ));
    }
}
