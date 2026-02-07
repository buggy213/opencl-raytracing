use std::path::{Path, PathBuf};

use cmake;

fn bindgen_customization(bindings_builder: bindgen::Builder) -> bindgen::Builder {
    bindings_builder.rustified_enum("CameraTypeKind")
}

fn main() {
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-changed=build.rs");
    
    // Builds the project in the directory located in `csrc`, installing it
    // into $OUT_DIR
    let dst = cmake::build("csrc");
    
    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=dylib=raytracing_optix");

    let include_path = if cfg!(target_os = "windows") {
        format!("-I \"{}\"", dst.display())
    } else {
        format!("-I{}", dst.display())
    };

    let Ok(optix_path) = std::env::var("OPTIX90_PATH") else {
        panic!("OPTIX90_PATH is not set");
    };

    let optix_include_path = PathBuf::from(optix_path).join("include");
    let optix_include_path = if cfg!(target_os = "windows") {
        format!("-I \"{}\"", optix_include_path.display())
    } else {
        format!("-I{}", optix_include_path.display())
    };

    
    let bindings_builder = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(include_path)
        .clang_arg(optix_include_path)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

    let bindings_builder = bindgen_customization(bindings_builder);

    let bindings = bindings_builder.generate().expect("unable to generate bindings");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_path.join("bindings.rs"))
        .expect("unable to write bindings");

    // compile OptiX IR through make
    println!("cargo:rerun-if-changed=csrc/kernels");
    let kernels_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("csrc/kernels");
    let exit_code = std::process::Command::new("make")
        .current_dir(&kernels_dir)
        .arg("all")
        .spawn()
        .expect("failed to run make")
        .wait()
        .expect("failed to run make");

    if !exit_code.success() {
        panic!("make failed with exit code {exit_code}");
    }
}