use std::{fs::File, path::Path};

use raytracing::{geometry::Vec3, scene::Scene};
use raytracing_cpu::render;

fn save_png(radiance: &[Vec3], scene: &Scene, output_path: &Path) {
    let width = scene.camera.raster_width;
    let height = scene.camera.raster_height;

    let file = File::create(output_path).expect("failed to create output file");
    let mut encoder = png::Encoder::new(file, width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);

    let mut writer = encoder
        .write_header()
        .expect("failed to write PNG header");

    let image_data: Vec<u8> = radiance.iter().flat_map(|v| {
        let r = (v.x() * 255.0).clamp(0.0, 255.0) as u8;
        let g = (v.y() * 255.0).clamp(0.0, 255.0) as u8;
        let b = (v.z() * 255.0).clamp(0.0, 255.0) as u8;
        [r, g, b]
    }).collect();

    writer.write_image_data(&image_data).expect("failed to write PNG data");
}

fn main() {
    let path = Path::new("scenes/test.glb");
    let mut scene = Scene::from_file(path, None).expect("failed to load scene");
    let output = render(&mut scene, 1);

    let output_path = Path::new("scenes/test.png");
    save_png(&output, &scene, output_path);
}