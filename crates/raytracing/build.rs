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
}