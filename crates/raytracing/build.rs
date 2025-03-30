use std::{env, path::PathBuf};

fn main() {
    println!("{:?}", env::var("EMBREE_DIR"));
}