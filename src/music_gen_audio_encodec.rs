use ndarray::{Array, Axis};
use tokio::sync::mpsc::Receiver;

pub struct MusicGenAudioEncodec {
    pub audio_encodec_decode: ort::Session,
}

impl MusicGenAudioEncodec {
    pub async fn encode<Cb: Fn(usize, usize)>(
        &self,
        mut channel: Receiver<ort::Result<[i64; 4]>>,
        max_len: usize, // TODO <- this parameter should not go here, I just put it here because I'm refactoring stuff.
        cb: Cb,
    ) -> ort::Result<Receiver<ort::Result<f32>>> {
        // TODO: The signature of this function suggests that a stream is returned,
        //  and it's true, but it's a "fake" stream. We first need to gather all
        //  the generated token ids, concatenate them, and feed them to the audio_encodec_decode
        //  model, because we cannot just feed it in chunks. It has some LSTMs and stateful
        //  things that will produce weird samples if the tokens are fed in chunks, so we
        //  need to feed them altogether at once.
        //  If we could feed chunks of tokens to the audio_encodec_decode, we could produce
        //  actual music streams.
        let mut data = vec![];
        let mut count = 0;
        while let Some(ids) = channel.recv().await {
            let ids = match ids {
                Ok(ids) => ids,
                Err(err) => return Err(err),
            };
            count += 1;
            cb(count, max_len);
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

        let (tx, rx) = tokio::sync::mpsc::channel(100);
        tokio::spawn(async move {
            let (_, data) = audio_values.try_extract_raw_tensor()?;
            for sample in data {
                let _ = tx.send(Ok(*sample)).await;
            }
            Ok::<(), ort::Error>(())
        });
        Ok(rx)
    }
}
