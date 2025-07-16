use std::{borrow::Cow, time::{Duration, Instant}};

use crate::{RenderView, WindowRequests};

pub(crate) struct DemoApplicationView {
    pipeline: wgpu::RenderPipeline,

    demo_open: bool,
    
    last_update: Instant,
    delta_s: Duration
}

impl RenderView for DemoApplicationView {
    fn init(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,

        render_target_format: wgpu::TextureFormat,
        _imgui_textures: &mut imgui_wgpu::Renderer,
    ) -> (Self, WindowRequests)
        where Self: Sized {
        let shader_module_descriptor = wgpu::ShaderModuleDescriptor {
            label: Some("Shaders"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/shader.wgsl"))),
        };

        let shader = device.create_shader_module(shader_module_descriptor);

        let pipeline_layout_descriptor = wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        };
        
        let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_descriptor);
        
        let targets = [Some(render_target_format.into())];
        let render_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &targets,
            }), // what does it mean to have no fragment stage (??)
            multiview: None, // what is a multiview render pass :skull:
            cache: None,
        };

        let render_pipeline = device.create_render_pipeline(&render_pipeline_descriptor);

        let me = Self {
            pipeline: render_pipeline,
            demo_open: true,
            delta_s: Duration::new(0, 0),
            last_update: Instant::now(),
        };

        (me, WindowRequests::default())
    }
    
    fn render(
        &mut self, 
        command_encoder: &mut wgpu::CommandEncoder, 
        render_texture: &wgpu::TextureView,
    
        _imgui_textures: &imgui::Textures<imgui_wgpu::Texture>,
    
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
        let color_attachments = [
            Some(wgpu::RenderPassColorAttachment { 
                view: &render_texture, 
                resolve_target: None, 
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::GREEN), store: wgpu::StoreOp::Store } 
            })
        ];
        let rpass_descriptor = wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        };
        let mut rpass = command_encoder.begin_render_pass(&rpass_descriptor);
        rpass.set_pipeline(&self.pipeline);
        rpass.draw(0..3, 0..1);
    }

    fn render_imgui(&mut self, ui: &mut imgui::Ui) {
        let window = ui.window("Hello world");
        window
            .size([300.0, 100.0], imgui::Condition::FirstUseEver)
            .build(|| {
                ui.text("Hello world!");
                ui.text("This...is...imgui-rs on WGPU!");
                ui.separator();
                let mouse_pos = ui.io().mouse_pos;
                ui.text(format!(
                    "Mouse Position: ({:.1},{:.1})",
                    mouse_pos[0], mouse_pos[1]
                ));
            });

        self.delta_s = Instant::now() - self.last_update;
        self.last_update = Instant::now();

        let delta_s = self.delta_s;
        let window = ui.window("Hello too");
        window
            .size([400.0, 200.0], imgui::Condition::FirstUseEver)
            .position([400.0, 200.0], imgui::Condition::FirstUseEver)
            .build(|| {
                ui.text(format!("Frametime: {delta_s:?}"));
            });

        ui.show_demo_window(&mut self.demo_open);
    }
}
