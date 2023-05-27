extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    if let Ok(e) = env::var("EMBREE_DIR") {
        let mut embree_dir = PathBuf::from(e);
        embree_dir.push("lib");
        eprintln!("linking to {}", embree_dir.display());
        println!("cargo:rerun-if-env-changed=EMBREE_DIR");

        // pass path which is being linked against to dependencies
        println!("cargo:lib={}", embree_dir.display());

        // Tell cargo to tell rustc to link the embree4
        // shared library.
        println!("cargo:rustc-link-search=native={}", embree_dir.display());
        println!("cargo:rustc-link-lib=embree4");
        
        // Tell cargo to invalidate the built crate whenever the wrapper changes
        println!("cargo:rerun-if-changed=wrapper.h");

        // The bindgen::Builder is the main entry point
        // to bindgen, and lets you build up options for
        // the resulting bindings.
        embree_dir.pop();
        embree_dir.push("include");
        let include_path = format!("-I{}", embree_dir.display());
        eprintln!("include path for bindgen: {}", include_path);
        let bindings = bindgen::Builder::default()
            // The input header we would like to generate
            // bindings for.
            .header("wrapper.h")
            // include embree headers
            .clang_arg(include_path)
            // nicer enums for BVH build settings
            .constified_enum_module("RTCBuildQuality")
            .constified_enum_module("RTCBuildFlags")
            // Tell cargo to invalidate the built crate whenever any of the
            // included header files changed.
            .parse_callbacks(Box::new(bindgen::CargoCallbacks))
            // Finish the builder and generate the bindings.
            .generate()
            // Unwrap the Result and panic on failure.
            .expect("Unable to generate bindings");

        // Write the bindings to the $OUT_DIR/bindings.rs file.
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        bindings
            .write_to_file(out_path.join("bindings.rs"))
            .expect("Couldn't write bindings!");
    } else {
        panic!("EMBREE_DIR set incorrectly");
    }
}