fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_ONNXRUNTIME_FROM_SOURCE");
    build::build()?;
    built::write_built_file()?;
    Ok(())
}

#[cfg(feature = "onnxruntime-from-source")]
mod build {
    #[allow(unused_imports)]
    use build_system::Accelarators;
    use std::env;
    use std::path::PathBuf;

    pub(crate) fn build() -> Result<(), Box<dyn std::error::Error>> {
        println!("cargo:rerun-if-changed=build-system");
        println!("cargo:rerun-if-env-changed=CARGO_FEATURE_COREML");
        println!("cargo:rerun-if-env-changed=CARGO_FEATURE_TENSORRT");
        println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA");
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
        #[allow(unused_mut)]
        let mut accelerators = vec![];
        #[cfg(feature = "cuda")]
        accelerators.push(Accelarators::CUDA);
        #[cfg(feature = "coreml")]
        accelerators.push(Accelarators::COREML);
        #[cfg(feature = "tensorrt")]
        accelerators.push(Accelarators::TENSORRT);

        let info = build_system::build(PathBuf::from(manifest_dir).join("target"), accelerators)?;
        info.to_out_dir();
        Ok(())
    }
}

#[cfg(not(feature = "onnxruntime-from-source"))]
mod build {
    pub(crate) fn build() -> Result<(), Box<dyn std::error::Error>> {
        // nothing.
        Ok(())
    }
}
