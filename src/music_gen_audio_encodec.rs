use std::sync::Arc;
use half::f16;
use ndarray::{Array, Axis};
use tokio::sync::mpsc::Receiver;

pub struct MusicGenAudioEncodec {
    pub audio_encodec_decode: Arc<ort::Session>,
}

impl MusicGenAudioEncodec {
    pub fn encode<Cb: Fn(usize, usize) + Sync + Send + 'static>(
        &self,
        mut channel: Receiver<ort::Result<[i64; 4]>>,
        max_len: usize, // TODO <- this parameter should not go here, I just put it here because I'm refactoring stuff.
        cb: Cb,
    ) -> Receiver<ort::Result<f32>> {
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        let tx2 = tx.clone();
        
        let audio_encodec_decode = self.audio_encodec_decode.clone();
        
        let future = async move {
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
            let mut outputs = audio_encodec_decode.run(ort::inputs![arr]?)?;
            let audio_values: ort::DynValue = outputs
                .remove("audio_values")
                .expect("audio_values not found in output");

            if let Ok((_, data)) = audio_values.try_extract_raw_tensor::<f32>() {
                for sample in data {
                    let _ = tx.send(Ok(*sample)).await;
                }
                return Ok(());
            }
            if let Ok((_, data)) = audio_values.try_extract_raw_tensor::<f16>() {
                for sample in data {
                    let _ = tx.send(Ok(f32::from(*sample))).await;
                }
                return Ok(());
            }
            Ok::<(), ort::Error>(())
        };

        tokio::spawn(async move {
            if let Err(err) = future.await {
                let _ = tx2.send(Err(err)).await;
            };
        });

        rx
    }
}
