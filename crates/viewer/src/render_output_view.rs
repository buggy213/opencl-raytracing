// 

use std::{borrow::Cow, collections::HashMap, rc::Rc};

use bytemuck::NoUninit;
use raytracing::geometry::Vec3;
use wgpu::{util::DeviceExt, ColorTargetState};

use crate::{ImguiInternalState, RenderGui, WgpuHandles};

// TODO: is it possible to just blit directly to the render target?
pub(crate) struct RenderOutputView {
    render: RenderResources,
    compute: ComputeResources,
    
    // opaque handle for imgui to render debug texture
    debug_texture_id: Option<imgui::TextureId>,
    imgui_blitter: Option<wgpu::util::TextureBlitter>,

    gui_state: GuiState,

    render_output_size: (u32, u32),
    render_output: Vec<Vec3>
}

struct GuiState {
    gamma: f32,
    exposure: f32,

    mouse_pos: [u32; 2],
}

struct RenderResources {
    shader: wgpu::ShaderModule,

    pipeline_layout: wgpu::PipelineLayout,
    pipeline: wgpu::RenderPipeline,

    triangle_buffer: wgpu::Buffer,

    texture: Rc<wgpu::Texture>,
    texture_sampler: wgpu::Sampler,

    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

#[derive(NoUninit, Clone, Copy)]
#[repr(C)]
struct ComputePushConstants {
    gamma: f32,
    exposure: f32,

    mouse_pos: [u32; 2]
}
struct ComputeResources {
    shader: wgpu::ShaderModule,

    compute_pipeline_layout: wgpu::PipelineLayout,
    main_compute_pipeline: wgpu::ComputePipeline,
    debug_compute_pipeline: wgpu::ComputePipeline,

    radiance_buffer: wgpu::Buffer,

    texture: Rc<wgpu::Texture>,
    debug_texture: wgpu::Texture,

    compute_bind_group_layout: wgpu::BindGroupLayout,
    compute_bind_group: wgpu::BindGroup,
}

impl RenderGui for RenderOutputView {
    fn render_imgui(&mut self, ui: &mut imgui::Ui) {
        let viewer_window = ui.window("Viewer")
            .size([300.0, 300.0], imgui::Condition::Always)
            .position([0.0, 0.0], imgui::Condition::Always)
            .resizable(false);
        viewer_window.build(|| {
            ui.text("Pixel peeper");
            let tex_id = self.debug_texture_id.expect("imgui texture not initialized");
            imgui::Image::new(tex_id, [200.0, 200.0]).build(&ui);

            ui.separator();
            ui.text("Render Output Settings");
            ui.input_float("Exposure", &mut self.gui_state.exposure).build();
            ui.input_float("Gamma", &mut self.gui_state.gamma).build();
            
            ui.text(format!("Mouse position: ({}, {})", 
                self.gui_state.mouse_pos[0], 
                self.gui_state.mouse_pos[1]));
        });
    }
}

impl RenderOutputView {
    pub(crate) fn init(
        wgpu_handles: &WgpuHandles, 
        targets: &[Option<ColorTargetState>],
        size: (u32, u32),
    ) -> RenderOutputView {
        let WgpuHandles { device, .. } = wgpu_handles;
        
        let extent = wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1
        };
        let usage = wgpu::TextureUsages::COPY_DST 
            | wgpu::TextureUsages::STORAGE_BINDING 
            | wgpu::TextureUsages::TEXTURE_BINDING;
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
        let texture = Rc::new(texture);

        let render_resources = Self::init_render_resources(wgpu_handles, targets, size, Rc::clone(&texture));
        let compute_resources = Self::init_compute_resources(wgpu_handles, size, Rc::clone(&texture));

        let me = RenderOutputView {
            render: render_resources,
            compute: compute_resources,

            debug_texture_id: None,
            imgui_blitter: None,

            gui_state: GuiState { 
                gamma: 2.2, 
                exposure: 1.0, 
                mouse_pos: [0, 0] 
            },
            

            render_output_size: size, 
            render_output: Vec::new(),
        };

        // generate test texture
        me.generate_test_pattern(wgpu_handles);
        me
    }

    fn init_render_resources(
        wgpu_handles: &WgpuHandles,
        targets: &[Option<ColorTargetState>],
        size: (u32, u32),

        texture: Rc<wgpu::Texture>,
    ) -> RenderResources {
        let WgpuHandles { device, .. } = wgpu_handles;

        let shader_module_descriptor = wgpu::ShaderModuleDescriptor {
            label: Some("Render Output Shaders"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("render_output_render.wgsl"))),
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

        let pipeline_constants = HashMap::from([
            ("SIZE_X".to_string(), size.0 as f64),
            ("SIZE_Y".to_string(), size.1 as f64)
        ]);
        let pipeline_compilation_options = wgpu::PipelineCompilationOptions {
            constants: &pipeline_constants,
            zero_initialize_workgroup_memory: false,
        };
        let render_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Render Output Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vert_main"),
                compilation_options: pipeline_compilation_options.clone(), // cheap clone
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
                compilation_options: pipeline_compilation_options.clone(), // cheap clone
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

        RenderResources {
            shader,
            pipeline_layout,
            pipeline: render_pipeline,
            triangle_buffer,
            texture,
            texture_sampler: sampler,
            bind_group_layout,
            bind_group,
        }
    }

    fn init_compute_resources(
        wgpu_handles: &WgpuHandles,
        size: (u32, u32),

        texture: Rc<wgpu::Texture>,
    ) -> ComputeResources {
        let WgpuHandles { device, .. } = wgpu_handles;

        let debug_texture_descriptor = wgpu::TextureDescriptor {
            label: Some("Render Output Debug Texture"),
            size: wgpu::Extent3d {
                width: 200,
                height: 200,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[wgpu::TextureFormat::Rgba32Float],
        };
        let debug_texture = device.create_texture(&debug_texture_descriptor);

        let shader_module_descriptor = wgpu::ShaderModuleDescriptor {
            label: Some("Render Output Compute Shaders"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("render_output_compute.wgsl"))),
        };

        let shader = device.create_shader_module(shader_module_descriptor);

        let bind_group_layout_desc = wgpu::BindGroupLayoutDescriptor {
            label: Some("Render Output Compute Bind Group Layout Descriptor"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { 
                        ty: wgpu::BufferBindingType::Storage { read_only: true }, 
                        has_dynamic_offset: false, 
                        min_binding_size: None 
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture { 
                        access: wgpu::StorageTextureAccess::WriteOnly, 
                        format: wgpu::TextureFormat::Rgba32Float, 
                        view_dimension: wgpu::TextureViewDimension::D2,
                    } ,
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture { 
                        access: wgpu::StorageTextureAccess::WriteOnly, 
                        format: wgpu::TextureFormat::Rgba32Float, 
                        view_dimension: wgpu::TextureViewDimension::D2,
                    } ,
                    count: None,
                }
            ],
        };

        let bind_group_layout = device.create_bind_group_layout(&bind_group_layout_desc);
        let pipeline_layout_descriptor = wgpu::PipelineLayoutDescriptor {
            label: Some("Render Output Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..size_of::<ComputePushConstants>() as u32
            }],
        };

        let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_descriptor);
        let pipeline_constants = HashMap::from([
            ("SIZE_X".to_string(), size.0 as f64),
            ("SIZE_Y".to_string(), size.1 as f64)
        ]);
        let pipeline_compilation_options = wgpu::PipelineCompilationOptions {
            constants: &pipeline_constants,
            zero_initialize_workgroup_memory: false,
        };
        let main_compute_pipeline_descriptor = wgpu::ComputePipelineDescriptor {
            label: Some("Render Output Compute Pipeline (main texture)"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main_texture"),
            compilation_options: pipeline_compilation_options.clone(), // cheap clone
            cache: None,
        };
        let debug_compute_pipeline_descriptor = wgpu::ComputePipelineDescriptor {
            label: Some("Render Output Compute Pipeline (debug texture)"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("debug_texture"),
            compilation_options: pipeline_compilation_options.clone(), // cheap clone
            cache: None,
        };

        let main_compute_pipeline = device.create_compute_pipeline(&main_compute_pipeline_descriptor);
        let debug_compute_pipeline = device.create_compute_pipeline(&debug_compute_pipeline_descriptor);

        let radiance_buffer_desc = wgpu::BufferDescriptor {
            label: Some("Render Output Radiance Buffer"),
            size: size.0 as u64 * size.1 as u64 * 3 * 4, // 3 channels, float32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        };

        let radiance_buffer = device.create_buffer(&radiance_buffer_desc);

        let texture_view_desc = wgpu::TextureViewDescriptor::default();
        let texture_view = texture.create_view(&texture_view_desc);
        let debug_texture_view = debug_texture.create_view(&texture_view_desc);

        let bind_group_desc = wgpu::BindGroupDescriptor {
            label: Some("Render Output Compute Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &radiance_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&debug_texture_view),
                }
            ],
        };
        let bind_group = device.create_bind_group(&bind_group_desc);

        ComputeResources { 
            shader, 
            compute_pipeline_layout: pipeline_layout, 
            main_compute_pipeline, 
            debug_compute_pipeline, 
            radiance_buffer, 
            texture, 
            debug_texture, 
            compute_bind_group_layout: bind_group_layout, 
            compute_bind_group: bind_group
        }
    }

    pub(crate) fn init_imgui(&mut self, wgpu_handles: &WgpuHandles, imgui_state: &mut ImguiInternalState) {
        let texture_config = imgui_wgpu::TextureConfig {
            size: wgpu::Extent3d { width: 200, height: 200, depth_or_array_layers: 1 },
            label: Some("Render Output Debug Texture"),
            // RENDER_ATTACHMENT needed for blit
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            ..Default::default()
        };
          
        let imgui_debug_texture = imgui_wgpu::Texture::new(
            &wgpu_handles.device,
            &imgui_state.renderer,
            texture_config
        );
        let id = imgui_state.renderer.textures.insert(imgui_debug_texture);
        self.debug_texture_id = Some(id);

        let dst_format = imgui_state.renderer.textures.get(id).unwrap().texture().format();

        self.imgui_blitter = Some(wgpu::util::TextureBlitter::new(
            &wgpu_handles.device,
            dst_format
        ));
    }

    // For now just piggybacks off of imgui's IO abstraction
    pub(crate) fn update(&mut self, io: &imgui::Io) {
        if io.mouse_pos != [f32::MAX, f32::MAX] {
            self.gui_state.mouse_pos = [
                io.mouse_pos[0] as u32,
                io.mouse_pos[1] as u32,
            ];
        }
    }

    // Renders texture onto 2 triangles covering the screen.
    pub(crate) fn render(
        &self, 
        imgui_textures: &imgui::Textures<imgui_wgpu::Texture>,

        wgpu_handles: &WgpuHandles, 
        render_target: &wgpu::Texture,
    ) {
        let WgpuHandles { device, queue, .. } = wgpu_handles;

        let view_descriptor = wgpu::TextureViewDescriptor::default();
        let view = render_target.create_view(&view_descriptor);

        let encoder_descriptor = wgpu::CommandEncoderDescriptor {
            label: Some("Render Output Command Encoder"),
        };
        let mut encoder = device.create_command_encoder(&encoder_descriptor);
        self.do_compute(&mut encoder);
        self.do_graphics(&mut encoder, view);
        self.prepare_render_imgui(imgui_textures, device, &mut encoder);

        queue.submit(Some(encoder.finish()));
    }

    fn do_compute(&self, encoder: &mut wgpu::CommandEncoder) {
        let cpass_descriptor = wgpu::ComputePassDescriptor {
            label: Some("Render Output Compute Pass"),
            timestamp_writes: None,
        };
        let mut cpass = encoder.begin_compute_pass(&cpass_descriptor);
        
        let push_constants = ComputePushConstants {
            gamma: self.gui_state.gamma,
            exposure: self.gui_state.exposure,
            mouse_pos: self.gui_state.mouse_pos,
        };
        
        cpass.set_pipeline(&self.compute.main_compute_pipeline);
        cpass.set_push_constants(0, bytemuck::bytes_of(&push_constants));
        
        cpass.set_bind_group(0, &self.compute.compute_bind_group, &[]);
        cpass.dispatch_workgroups(
            u32::div_ceil(self.render_output_size.0, 8), 
            u32::div_ceil(self.render_output_size.1, 8), 
            1
        );

        cpass.set_pipeline(&self.compute.debug_compute_pipeline);
        cpass.set_push_constants(0, bytemuck::bytes_of(&push_constants));
        
        cpass.dispatch_workgroups(
            u32::div_ceil(200, 8), 
            u32::div_ceil(200, 8), 
            1
        );
    }

    fn do_graphics(&self, encoder: &mut wgpu::CommandEncoder, view: wgpu::TextureView) {
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
        rpass.set_pipeline(&self.render.pipeline);

        let indices = self.render.triangle_buffer.slice(..);
        rpass.set_index_buffer(indices, wgpu::IndexFormat::Uint32);
        rpass.set_bind_group(0, &self.render.bind_group, &[]);
        rpass.draw_indexed(0..6, 0, 0..1);
    }

    pub(crate) fn prepare_render_imgui(
        &self, 
        imgui_textures: &imgui::Textures<imgui_wgpu::Texture>,

        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder
    ) {
        let src_view = self.compute.debug_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let dst_view = imgui_textures.get(self.debug_texture_id.expect("imgui texture not put in textures map"))
            .expect("imgui texture not put in textures map")
            .texture()
            .create_view(&wgpu::TextureViewDescriptor::default());
        
        
        let blitter = self.imgui_blitter.as_ref().expect("imgui blitter not initialized"); 
        blitter.copy(
            device, 
            encoder,
            &src_view,
            &dst_view,
        );
    }

    pub(crate) fn resize(&self, width: u32, height: u32) {
        // no-op, since texture sampling allows for arbitrary size
    }

    fn generate_test_pattern(&self, wgpu_handles: &WgpuHandles) {
        let elts = self.render_output_size.0 * self.render_output_size.1 * 4;
        let mut test_texture: Vec<f32> = Vec::with_capacity(elts as usize);

        for j in 0..self.render_output_size.1 {
            for i in 0..self.render_output_size.0 {
                let r = (i as f32) / (self.render_output_size.0 as f32);
                let b = (j as f32) / (self.render_output_size.1 as f32);
                test_texture.push(r);
                test_texture.push(0.0);
                test_texture.push(b);
            }
        }

        let WgpuHandles { queue, .. } = wgpu_handles;
        
        // Write the test texture to the radiance buffer
        let data: &[u8] = bytemuck::cast_slice(test_texture.as_slice());
        queue.write_buffer(&self.compute.radiance_buffer, 0, data);
    }
}