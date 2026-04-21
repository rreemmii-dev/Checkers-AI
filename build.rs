extern crate cc;

use cfg_aliases::cfg_aliases;

fn main() {
    cfg_aliases! {
        base: { not(any(feature = "openblas", feature = "cublas")) },
        openblas: { all(feature = "openblas", not(feature = "cublas")) },
        cublas: { feature = "cublas" },
        f64_precision: { feature = "f64_precision" },
        nn_is_sync: { not(cublas) },
    }

    #[cfg(feature = "cublas")]
    {
        const FOLDER: &str = "src/neural_network/types/cuda";
        const F64_PRECISION: bool = cfg!(feature = "f64_precision");
        const FLOAT_FLAG: &str = if F64_PRECISION {
            "-DF64_PRECISION"
        } else {
            "-DF32_PRECISION" // Unused by C, can be removed eventually
        };

        cc::Build::new()
            .cuda(true)
            // .ccbin(false) // To let NVCC choose a compatible C++ compiler
            .file(FOLDER.to_string() + "/kernel.cu")
            .flag(FLOAT_FLAG)
            // .flag("-g") // Debug host
            // .flag("-lineinfo") // Profiling
            .compile("libkernel.a");

        println!("cargo:rustc-link-lib=cublasLt");
        println!("cargo:rerun-if-changed={FOLDER}/kernel.*");
    }
}
