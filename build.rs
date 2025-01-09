fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_ONNXRUNTIME_FROM_SOURCE");
    build::build()?;
    built::write_built_file()?;
    Ok(())
}

#[cfg(feature = "onnxruntime-from-source")]
mod build {
    #[allow(unused_imports)]
    use build_system::Accelerators;
    use std::env;
    use std::path::PathBuf;

    pub(crate) fn build() -> Result<(), Box<dyn std::error::Error>> {
        println!("cargo:rerun-if-changed=build-system");
        println!("cargo:rerun-if-env-changed=CARGO_FEATURE_COREML");
        println!("cargo:rerun-if-env-changed=CARGO_FEATURE_TENSORRT");
        println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA");
        println!("cargo:rerun-if-env-changed=ONNXRUNTIME_BUILD_DIR");
        println!("cargo:rerun-if-env-changed=BUILD_HASH_FILE");
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
        #[allow(unused_mut)]
        let mut accelerators = vec![];
        #[cfg(feature = "cuda")]
        accelerators.push(Accelerators::CUDA);
        #[cfg(feature = "coreml")]
        accelerators.push(Accelerators::COREML);
        #[cfg(feature = "tensorrt")]
        accelerators.push(Accelerators::TENSORRT);

        let dir = match env::var("ONNXRUNTIME_BUILD_DIR") {
            Ok(p) => PathBuf::from(p),
            Err(_) => PathBuf::from(manifest_dir).join("target"),
        };

        let info = build_system::build(dir, accelerators);
        info.write_build_info();
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
