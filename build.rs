use std::{env, path::PathBuf};

fn main() {
    println!("{:?}", env::var("EMBREE_DIR"));
    if let Ok(e) = env::var("DEP_EMBREE4_LIB") {
        let embree_lib_dir = PathBuf::from(e);
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", embree_lib_dir.display());
    }
    else {
        panic!("unable to get metadata from embree4 dependency");
    }
    println!("cargo:rerun-if-env-changed=POCL_RT_PATH");
    println!("cargo:rerun-if-env-changed=VORTEX_DRV_STUB_PATH");
    // ugly hack: if environment variables from env.sh are set, then link against them
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