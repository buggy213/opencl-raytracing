mod bundle;

use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
#[command(name = "xtask", about = "Build automation tasks")]
enum Xtask {
    /// Build the CLI binary and bundle shared libraries into a self-contained directory
    Bundle {
        /// Build in release mode
        #[arg(long)]
        release: bool,

        /// Include OptiX backend
        #[arg(long)]
        optix: bool,

        /// Output directory
        #[arg(long, default_value = "dist")]
        output_dir: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    match Xtask::parse() {
        Xtask::Bundle {
            release,
            optix,
            output_dir,
        } => bundle::run(release, optix, &output_dir),
    }
}
