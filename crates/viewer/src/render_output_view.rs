// 

use std::borrow::Cow;

use raytracing::geometry::Vec3;
use wgpu::{util::DeviceExt, ColorTargetState, SurfaceTexture};

use crate::{RenderGui, WgpuHandles};

// TODO: is it possible to just blit directly to the render target?
pub(crate) struct RenderOutputView {
    shader: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,
    pipeline: wgpu::RenderPipeline,

    triangle_buffer: wgpu::Buffer,
    texture: wgpu::Texture,
    texture_sampler: wgpu::Sampler,

    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,

    render_output_size: (u32, u32),
    render_output: Vec<Vec3>
}

impl RenderGui for RenderOutputView {
    fn render_imgui(&mut self, ui: &mut imgui::Ui) {
        let window = ui.window("Viewer")
            .size([300.0, 300.0], imgui::Condition::Always)
            .position([0.0, 0.0], imgui::Condition::Always);
        window.build(|| {
            ui.text("asdf");
        });
    }
}

impl RenderOutputView {
    pub(crate) fn init(
        wgpu_handles: &WgpuHandles, 
        targets: &[Option<ColorTargetState>],
        size: (u32, u32),
    ) -> RenderOutputView {
        let WgpuHandles { instance, surface, adapter, device, queue, shader, pipeline_layout, pipeline, swapchain_format } = wgpu_handles;

        let shader_module_descriptor = wgpu::ShaderModuleDescriptor {
            label: Some("Render Output Shaders"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("render_output.wgsl"))),
        };

        let shader = device.create_shader_module(shader_module_descriptor);

        let bind_group_layout_desc = wgpu::BindGroupLayoutDescriptor {
            label: Some("Render Output Bind Group Layout Descriptor"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { 
                        sample_type: wgpu::TextureSampleType::Float { filterable: true }, 
                        view_dimension: wgpu::TextureViewDimension::D2, 
                        multisampled: false
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                }
            ],
        };
        
        let bind_group_layout = device.create_bind_group_layout(&bind_group_layout_desc);
        let pipeline_layout_descriptor = wgpu::PipelineLayoutDescriptor {
            label: Some("Render Output Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        };

        let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_descriptor);

        let render_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Render Output Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vert_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("frag_main"),
                compilation_options: Default::default(),
                targets,
            }), 
            multiview: None,
            cache: None,
        };
        
        let render_pipeline = device.create_render_pipeline(&render_pipeline_descriptor);
        let tris = &[2u32, 1u32, 0u32, 2u32, 3u32, 1u32];

        let triangle_buffer_desc = wgpu::util::BufferInitDescriptor { 
            label: Some("Render Output Quad"), 
            contents: bytemuck::cast_slice(tris), 
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST 
        };
        let triangle_buffer = device.create_buffer_init(&triangle_buffer_desc);
        
        let extent = wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1
        };
        let usage = wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING;
        let texture_desc = wgpu::TextureDescriptor { 
            label: Some("Render Output Texture"), 
            size: extent, 
            mip_level_count: 1, 
            sample_count: 1, 
            dimension: wgpu::TextureDimension::D2, 
            format: wgpu::TextureFormat::Rgba32Float, 
            usage, 
            view_formats: &[wgpu::TextureFormat::Rgba32Float] 
        };

        let texture = device.create_texture(&texture_desc);

        let texture_view_desc = wgpu::TextureViewDescriptor {
            label: Some("Render View Texture View Descriptor"),
            ..Default::default()
        };
        let texture_view = texture.create_view(&texture_view_desc);

        let sampler_desc = wgpu::SamplerDescriptor {
            label: Some("Render Output Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        };
        let sampler = device.create_sampler(&sampler_desc);

        let bind_group_desc = wgpu::BindGroupDescriptor {
            label: Some("Render Output Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },

                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                }
            ],
        };

        let bind_group = device.create_bind_group(&bind_group_desc);

        let me = RenderOutputView { 
            shader,
            pipeline_layout,
            pipeline: render_pipeline,
            triangle_buffer,
            texture,
            texture_sampler: sampler,

            bind_group_layout,
            bind_group,

            render_output_size: size, 
            render_output: Vec::new(),
        };

        // generate test texture
        me.generate_test_texture(wgpu_handles);
        
        me
    }

    // For now just piggybacks off of imgui's IO abstraction
    pub(crate) fn update(&mut self, io: &imgui::Io) {
        
    }

    // Renders texture onto 2 triangles covering the screen.
    pub(crate) fn render(&self, wgpu_handles: &WgpuHandles, render_target: &SurfaceTexture) {
        let WgpuHandles { instance, surface, adapter, device, queue, shader, pipeline_layout, pipeline, swapchain_format } = wgpu_handles;

        let frame_view_descriptor = wgpu::TextureViewDescriptor::default();
        let view = render_target.texture.create_view(&&frame_view_descriptor);

        let encoder_descriptor = wgpu::CommandEncoderDescriptor {
            label: Some("Render Output Command Encoder"),
        };
        let mut encoder = device.create_command_encoder(&encoder_descriptor);
        {
            let color_attachments = [
                Some(wgpu::RenderPassColorAttachment { 
                    view: &view, 
                    resolve_target: None, 
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } 
                })
            ];
            let rpass_descriptor = wgpu::RenderPassDescriptor {
                label: Some("Render Output Render Pass"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            };
            let mut rpass = encoder.begin_render_pass(&rpass_descriptor);
            rpass.set_pipeline(&self.pipeline);

            let indices = self.triangle_buffer.slice(..);
            rpass.set_index_buffer(indices, wgpu::IndexFormat::Uint32);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.draw_indexed(0..6, 0, 0..1);
        }

        queue.submit(Some(encoder.finish()));
    }

    pub(crate) fn resize(&self, width: u32, height: u32) {
        // no-op, since texture sampling allows for arbitrary size
    }

    fn generate_test_texture(&self, wgpu_handles: &WgpuHandles) {
        let elts = self.render_output_size.0 * self.render_output_size.1 * 4;
        let mut test_texture: Vec<f32> = Vec::with_capacity(elts as usize);

        for j in 0..self.render_output_size.1 {
            for i in 0..self.render_output_size.0 {
                let r = (i as f32) / (self.render_output_size.0 as f32);
                let b = (j as f32) / (self.render_output_size.1 as f32);
                test_texture.push(r);
                test_texture.push(0.0);
                test_texture.push(b);
                test_texture.push(1.0);
            }
        }

        let WgpuHandles { instance, surface, adapter, device, queue, shader, pipeline_layout, pipeline, swapchain_format } = wgpu_handles;
        
        let texture = wgpu::TexelCopyTextureInfo {
            texture: &self.texture,
            mip_level: 0,
            origin: Default::default(),
            aspect: wgpu::TextureAspect::All,
        };

        let data_layout = wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(self.render_output_size.0 * 4 * 4),
            rows_per_image: None,
        };
        let data: &[u8] = bytemuck::cast_slice(test_texture.as_slice());
        let size = wgpu::Extent3d {
            width: self.render_output_size.0,
            height: self.render_output_size.1,
            depth_or_array_layers: 1,
        };

        queue.write_texture(texture, data, data_layout, size);
    }
}