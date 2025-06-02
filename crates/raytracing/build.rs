use std::env;

fn main() {
    println!("{:?}", env::var("EMBREE_DIR"));
}