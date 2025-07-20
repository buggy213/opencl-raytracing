use std::borrow::Cow;

use encase::{impl_vector, vector::{AsMutVectorParts, AsRefVectorParts, FromVectorParts}, ShaderSize, ShaderType, StorageBuffer};
use raytracing::geometry::{Matrix4x4};
use wgpu::util::DeviceExt;

use crate::{RenderView, WindowRequests};

pub(crate) struct SceneView {
    // scene: Scene,

    mvp: Matrix4x4,
    square_vertices: wgpu::Buffer,
    square_triangles: wgpu::Buffer,

    bind_group: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline
}

struct Vec3([f32; 3]);

impl AsRefVectorParts<f32, 3> for Vec3 {
    fn as_ref_parts(&self) -> &[f32; 3] {
        &self.0
    }
}

impl AsMutVectorParts<f32, 3> for Vec3 {
    fn as_mut_parts(&mut self) -> &mut [f32; 3] {
        &mut self.0
    }
}

impl FromVectorParts<f32, 3> for Vec3 {
    fn from_parts(parts: [f32; 3]) -> Self {
        Self(parts)
    }
}

impl_vector!(3, Vec3, f32);

#[derive(ShaderType)]
struct VertexData {
    position: Vec3,
}

struct PushConstants {
    // note: must be in column major order
    mvp: [f32; 16],
    screen_size: [f32; 2]
}

impl RenderView for SceneView {
    fn init(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,

        render_target_format: wgpu::TextureFormat,

        // imgui has its own texture management system, which we need to work with
        _imgui_renderer: &mut imgui_wgpu::Renderer
    ) -> (Self, crate::WindowRequests) 
        where Self: Sized {
        
        dbg!(VertexData::SHADER_SIZE);
        
        let square_vertices: [VertexData; 4] = [
            VertexData { position: Vec3([-0.5, -0.5, 0.0]) },
            VertexData { position: Vec3([0.5, -0.5, 0.0]) },
            VertexData { position: Vec3([-0.5, 0.5, 0.0]) },
            VertexData { position: Vec3([0.5, 0.5, 0.0]) }
        ];
        let mut square_vertices_buffer = StorageBuffer::new(Vec::<u8>::new());
        square_vertices_buffer.write(&square_vertices).unwrap();

        let square_vertices = wgpu::util::BufferInitDescriptor {
            label: Some("Square Vertices"),
            contents: &square_vertices_buffer.into_inner(),
            usage: wgpu::BufferUsages::STORAGE,
        };

        let square_vertices = device.create_buffer_init(&square_vertices);

        let square_triangles = [
            0, 2, 1,
            2, 3, 1
        ];

        let square_triangles = wgpu::util::BufferInitDescriptor {
            label: Some("Square Triangles"),
            contents: bytemuck::cast_slice(square_triangles.as_slice()),
            usage: wgpu::BufferUsages::STORAGE,
        };

        let square_triangles = device.create_buffer_init(&square_triangles);

        let mvp = Matrix4x4::identity();

        let bind_group_layout_desc = wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },

                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None
                }
            ],
        };

        let bind_group_layout = device.create_bind_group_layout(&bind_group_layout_desc);
        let render_pipeline_layout_desc = wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[
                wgpu::PushConstantRange { stages: wgpu::ShaderStages::VERTEX, range: 0..size_of::<PushConstants>() as u32 }
            ],
        };

        let render_pipeline_layout = device.create_pipeline_layout(&render_pipeline_layout_desc);

        let shader_desc = wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/wireframe.wgsl"))),
        };

        let shader = device.create_shader_module(shader_desc);

        let targets = &[Some(render_target_format.into())];
        let render_pipeline_desc = wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState { 
                module: &shader, 
                entry_point: None, 
                compilation_options: Default::default(), 
                buffers: &[] 
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: None,
                compilation_options: Default::default(),
                targets,
            }),
            
            primitive: Default::default(),
            depth_stencil: Default::default(),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        };

        let render_pipeline = device.create_render_pipeline(&render_pipeline_desc);

        let bind_group_desc = wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &square_vertices,
                        offset: 0,
                        size: None
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &square_triangles,
                        offset: 0,
                        size: None
                    }),
                }
            ],
        };

        let bind_group = device.create_bind_group(&bind_group_desc);

        let me = Self {
            mvp,
            square_vertices,
            square_triangles,

            bind_group,
            render_pipeline
        };

        (me, WindowRequests::default())
    }

    fn render(
        &mut self, 
        command_encoder: &mut wgpu::CommandEncoder, 
        render_texture: &wgpu::TextureView,

        _imgui_textures: &imgui::Textures<imgui_wgpu::Texture>,

        // in case view requires creating bind group or writing buffer / texture data
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
        let rpass_desc = wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: render_texture,
                resolve_target: None,
                ops: Default::default()
            })],
            ..Default::default()
        };
        let mut rpass = command_encoder.begin_render_pass(&rpass_desc);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.set_pipeline(&self.render_pipeline);

        let transposed_mvp = &self.mvp.transposed();
        let transposed_mvp: &[f32] = transposed_mvp.into();
        let screen_size: &[f32] = &[500.0, 500.0];

        rpass.set_push_constants(wgpu::ShaderStages::VERTEX, 0, bytemuck::cast_slice(transposed_mvp));
        rpass.set_push_constants(wgpu::ShaderStages::VERTEX, 64, bytemuck::cast_slice(screen_size));
        rpass.draw(0..6, 0..1);
    }

    fn render_imgui(&mut self, ui: &mut imgui::Ui) {
        let mut opened = true;
        ui.show_demo_window(&mut opened);
    }
}