

use std::{fs::{self, read_to_string}, ptr::{null_mut}, path::Path};

use accel::{bvh2::BVH2, LinearizedBVHNode};
use geometry::{Transform, Vec3, CLMesh};
use lights::Light;
use scene::{Scene, RenderTile};
use cli::Cli;
use clap::Parser;

mod geometry;
mod accel;
mod macros;
mod scene;
mod lights;
mod cli;
mod backends;

fn main() {
    let cli: Cli = Cli::parse();
    let use_tile: bool = cli.tile_x.and(cli.tile_y).and(cli.tile_height).and(cli.tile_width).is_some();
    let render_tile: Option<RenderTile> = if use_tile { 
        Some(RenderTile { x0: cli.tile_x.unwrap(), y0: cli.tile_y.unwrap(), x1: cli.tile_x.unwrap() + cli.tile_width.unwrap(), y1: cli.tile_y.unwrap() + cli.tile_height.unwrap() }) } 
    else { 
        None 
    };
    
}
