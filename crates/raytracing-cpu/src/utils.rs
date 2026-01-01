//! I/O utilities for generated output, shared between command-line driver and test code

pub mod png {
    use std::{fs::File, path::Path};
    use raytracing::geometry::{Vec2, Vec3};

    pub fn save_png(rgb_data: &[Vec3], exposure: f32, width: u32, height: u32, output_path: &Path) {
        let file = File::create(output_path).expect("failed to create output file");
        let mut encoder = png::Encoder::new(file, width as u32, height as u32);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);

        // rgb values we put in are linear; png encoder will do gamma correction for us
        encoder.set_source_gamma(png::ScaledFloat::new(1.0));

        let mut writer = encoder
            .write_header()
            .expect("failed to write PNG header");

        let image_data: Vec<u8> = rgb_data.iter().flat_map(|v| {
            let r = (v.x() / exposure * 255.0).clamp(0.0, 255.0) as u8;
            let g = (v.y() / exposure * 255.0).clamp(0.0, 255.0) as u8;
            let b = (v.z() / exposure * 255.0).clamp(0.0, 255.0) as u8;
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

    // converts uvs to RGB by encoding in only R and G channels
    // assumes uvs are between 0.0 and 1.0 (i.e. no wrapping is occurring). if this is
    // a problem, use exr instead
    pub fn uvs_to_rgb(uvs: &[Vec2]) -> Vec<Vec3> {
        let mut uv_rgb = Vec::with_capacity(uvs.len());
        for uv in uvs {
            uv_rgb.push(Vec3(uv.0, uv.1, 0.0));
        }

        uv_rgb
    }
}

pub mod exr {
    use std::path::Path;

    use exr::prelude::*;
    use num::PrimInt;
    use raytracing::geometry::Vec3;

    pub fn channels_from_vec3(
        channels: &mut Vec<AnyChannel<FlatSamples>>, 
        names: &[&str; 3],
        data: &[Vec3],
    ) {
        let mut channel0_data = Vec::with_capacity(data.len());
        let mut channel1_data = Vec::with_capacity(data.len());
        let mut channel2_data = Vec::with_capacity(data.len());

        for v in data {
            channel0_data.push(v.0);
            channel1_data.push(v.1);
            channel2_data.push(v.2);
        }

        let channel0 = AnyChannel::new(names[0], FlatSamples::F32(channel0_data));
        let channel1 = AnyChannel::new(names[1], FlatSamples::F32(channel1_data));
        let channel2 = AnyChannel::new(names[2], FlatSamples::F32(channel2_data));

        channels.push(channel0);
        channels.push(channel1);
        channels.push(channel2);
    }

    pub fn channel_from_f32_array(
        channels: &mut Vec<AnyChannel<FlatSamples>>,
        name: &str,
        data: &[f32]
    ) {
        let channel = AnyChannel::new(name, FlatSamples::F32(data.to_vec()));
        channels.push(channel);
    }

    pub fn channel_from_int_array<Value: PrimInt>(
        channels: &mut Vec<AnyChannel<FlatSamples>>,
        name: &str,
        data: &[Value]
    ) {
        const { assert!(size_of::<Value>() <= 4, "exr can only support u32-sized data"); }
        let mut channel_data = Vec::with_capacity(data.len());
        for v in data {
            let v_u32 = v.to_u32().unwrap();
            channel_data.push(v_u32);
        }

        let channel = AnyChannel::new(name, FlatSamples::U32(channel_data));
        channels.push(channel);
    }

    pub fn save_openexr(
        channels: Vec<AnyChannel<FlatSamples>>,
        width: u32,
        height: u32,
        output_path: &Path
    ) {
        let width = width as usize;
        let height = height as usize;
        
        let layer = Layer::new(
            (width, height),
            LayerAttributes::named("main"),
            Encoding::FAST_LOSSLESS,
            AnyChannels::sort(SmallVec::from_vec(channels))
        );

        let img: Image<_> = Image::new(
            ImageAttributes::with_size((width, height)),
            layer
        );

        img.write()
            .to_file(output_path)
            .expect("writing exr failed");
    }
}