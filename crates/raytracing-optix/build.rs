use cmake;

fn main() {
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-changed=build.rs");
    
    // Builds the project in the directory located in `csrc`, installing it
    // into $OUT_DIR
    let dst = cmake::build("csrc");

    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=dylib=raytracing_optix");
}