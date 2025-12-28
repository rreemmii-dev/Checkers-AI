extern crate cc;

fn main() {
    #[cfg(feature = "cublas")]
    {
        cc::Build::new()
            .cuda(true)
            .file("src/players/neural_network/neural_networks_types/cuda/kernel.cu")
            .compile("libkernel.a");

        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rerun-if-changed=src/kernel.cu");
    }
}
