use std::{env, path::PathBuf};

fn main() {
    println!("{:?}", env::var("EMBREE_DIR"));
    if let Ok(e) = env::var("EMBREE_DIR") {
        let mut embree_dir = PathBuf::from(e);
        embree_dir.push("lib");
        println!("cargo:rustc-link-search=native={}", embree_dir.display());
        println!("cargo:rustc-link-lib=embree4");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", embree_dir.display());
    }
    else {
        panic!("embree not found at $EMBREE_DIR!");
    }
    
    if let Ok(e) = env::var("POCL_RT_PATH") {
        if let Ok(f) = env::var("VORTEX_DRV_STUB_PATH") {
            let mut pocl_runtime = PathBuf::from(e);
            pocl_runtime.push("lib");
            println!("cargo:rustc-link-search=native={}", pocl_runtime.display());

            let vortex_driver_stub = PathBuf::from(f);
            println!("cargo:rustc-link-search=native={}", vortex_driver_stub.display());
            println!("cargo:rustc-link-lib=vortex");
        }
    }
}