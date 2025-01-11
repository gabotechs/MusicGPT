use anyhow::anyhow;
use log::{error, info};
use ort::execution_providers::{
    CUDAExecutionProvider, CoreMLExecutionProvider, ExecutionProvider, ExecutionProviderDispatch,
    TensorRTExecutionProvider,
};
use ort::session::Session;

pub fn init_gpu() -> anyhow::Result<(&'static str, ExecutionProviderDispatch)> {
    let mut dummy_builder = Session::builder()?;

    if cfg!(feature = "tensorrt") {
        let provider = TensorRTExecutionProvider::default();
        match provider.register(&mut dummy_builder) {
            Ok(_) => {
                info!("{} detected", provider.as_str());
                return Ok(("TensorRT", provider.build()));
            }
            Err(err) => error!("Could not load {}: {}", provider.as_str(), err),
        }
    }
    if cfg!(feature = "cuda") {
        let provider = CUDAExecutionProvider::default();
        match provider.register(&mut dummy_builder) {
            Ok(_) => {
                info!("{} detected", provider.as_str());
                return Ok(("Cuda", provider.build()));
            }
            Err(err) => error!("Could not load {}: {}", provider.as_str(), err),
        }
    }
    if cfg!(feature = "coreml") {
        let provider = CoreMLExecutionProvider::default().with_ane_only();
        match provider.register(&mut dummy_builder) {
            Ok(_) => {
                info!("{} detected", provider.as_str());
                return Ok(("CoreML", provider.build()));
            }
            Err(err) => error!("Could not load {}: {}", provider.as_str(), err),
        }
    }

    Err(anyhow!(
        "No hardware accelerator was detected, try running the program without the --gpu flag",
    ))
}
