use clap::Parser;

#[derive(Parser)]
pub struct Cli {
    pub(crate) output_file: Option<String>,
    pub(crate) tile_x: Option<usize>,
    pub(crate) tile_y: Option<usize>,
    pub(crate) tile_width: Option<usize>,
    pub(crate) tile_height: Option<usize>
}