

use std::{fs::{self, read_to_string}, ptr::{null_mut}, path::Path};

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

pub use backends::RenderingBackend;