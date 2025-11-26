//! I/O utilities for generated output, shared between command-line driver and test code

use std::{fs::File, path::Path};

use raytracing::{geometry::Vec3, scene::Scene};

pub fn save_png(radiance: &[Vec3], scene: &Scene, output_path: &Path) {
    let width = scene.camera.raster_width;
    let height = scene.camera.raster_height;

    let file = File::create(output_path).expect("failed to create output file");
    let mut encoder = png::Encoder::new(file, width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_source_gamma(png::ScaledFloat::new(1.0));

    let mut writer = encoder
        .write_header()
        .expect("failed to write PNG header");

    let image_data: Vec<u8> = radiance.iter().flat_map(|v| {
        let r = (v.x() / 1000.0 * 255.0).clamp(0.0, 255.0) as u8;
        let g = (v.y() / 1000.0 * 255.0).clamp(0.0, 255.0) as u8;
        let b = (v.z() / 1000.0 * 255.0).clamp(0.0, 255.0) as u8;
        [r, g, b]
    }).collect();

    writer.write_image_data(&image_data).expect("failed to write PNG data");
}

pub fn normals_to_rgb(normals: &mut [Vec3]) {
    for normal in normals {
        *normal += Vec3(1.0, 1.0, 1.0);
        *normal /= 2.0;
    }
}