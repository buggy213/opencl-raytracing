use std::env;
use std::path::PathBuf;

fn main() {
    if let Ok(e) = env::var("EMBREE_DIR") {
        let embree_dir = PathBuf::from(e);
         
        let embree_lib_dir = embree_dir.join("lib");
        println!("cargo::rerun-if-env-changed=EMBREE_DIR");

        // Tell cargo to tell rustc to link the embree4
        // shared library.
        println!("cargo::rustc-link-search=native={}", embree_lib_dir.display());
        println!("cargo::rustc-link-lib=embree4");

        // Copy DLLs to output directory if on Windows
        if cfg!(target_os = "windows") {
            let dll_path = embree_dir.join("bin");
            let dlls = std::fs::read_dir(dll_path)
                .expect("Failed to open directory containing DLLs")
                .filter_map(|res| res.ok())
                .map(|entry| entry.path())
                .filter_map(|path| {
                    eprintln!("file path: {path:?}");
                    eprintln!("file ext: {:?}", path.extension().and_then(|s| s.to_str()));
                    if path.extension().and_then(|s| s.to_str()) == Some("dll") {
                        Some(path)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            
            let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
            for dll_path in dlls {
                let dest_path = out_path.join(dll_path.file_name().unwrap());
                std::fs::copy(dll_path, dest_path).expect("Failed to copy DLL");
            }

            // this makes cargo add OUT_DIR to PATH so that dynamic linker can find it there
            println!("cargo::rustc-link-search=native={}", out_path.display());
        }
        else if cfg!(target_os = "linux") {
            // unfortunately, theres not really a good way to ensure that 
            // loader will find the correct thing at runtime. I might try to add
            // RPATH $ORIGIN later, but for now just use LD_LIBRARY_PATH

            // see: https://github.com/rust-lang/cargo/issues/5077#issuecomment-1284482987     
        }
        
        // Tell cargo to invalidate the built crate whenever the wrapper changes
        println!("cargo::rerun-if-changed=wrapper.h");

        // The bindgen::Builder is the main entry point
        // to bindgen, and lets you build up options for
        // the resulting bindings.
        let embree_include_dir = embree_dir.join("include");
        let include_path = format!("-I{}", embree_include_dir.display());
        eprintln!("include path for bindgen: {include_path}");
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
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
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