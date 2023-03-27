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
    
}