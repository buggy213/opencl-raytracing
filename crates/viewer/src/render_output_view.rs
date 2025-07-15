use std::{borrow::Cow, collections::HashMap, path::{Path, PathBuf}};

use bytemuck::NoUninit;
use raytracing::geometry::Vec3;
use wgpu::{util::DeviceExt, TextureFormat};

use crate::{RenderView, WgpuHandles};

// TODO: is it possible to just blit directly to the render target?
pub(crate) struct RenderOutputView {
    graphics: GraphicsResources,
    compute: ComputeResources,
    
    // opaque handle for imgui to render debug texture
    debug_texture_id: imgui::TextureId,
    imgui_blitter: wgpu::util::TextureBlitter,

    gui_state: GuiState,

    render_output_size: (u32, u32),
    render_output: Vec<Vec3>
}

struct GuiState {
    gamma: f32,
    exposure: f32,

    mouse_pos: [u32; 2],

    scenes: Vec<PathBuf>,
    selected_scene: usize,
    render_scene_requested: bool,

    spp: u32,
    light_samples: u32
}

struct GraphicsResources {
    pipeline: wgpu::RenderPipeline,

    triangle_buffer: wgpu::Buffer,
    
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
    main_compute_pipeline: wgpu::ComputePipeline,
    debug_compute_pipeline: wgpu::ComputePipeline,

    radiance_buffer: wgpu::Buffer,

    debug_texture: wgpu::Texture,
    
    compute_bind_group: wgpu::BindGroup,
}

#[derive(Clone, Copy)]
struct RaytraceParams {
    spp: u32,
    light_samples: u32,
}

struct RaytracerResult {
    radiance: Vec<Vec3>,
    raster_size: (u32, u32)
}

fn raytrace_scene(path: &Path, params: RaytraceParams) -> RaytracerResult {
    let mut scene = raytracing::scene::Scene::from_file(path, None).expect("failed to load scene");
    let output = raytracing_cpu::render(
        &mut scene, 
        params.spp,
        params.light_samples
    );
    
    RaytracerResult { 
        radiance: output, 
        raster_size: (scene.camera.raster_width as u32, scene.camera.raster_height as u32) 
    }
}

impl RenderView for RenderOutputView {
    fn init(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    
        render_target_format: wgpu::TextureFormat,
    
        // imgui has its own texture management system, which we need to work with
        imgui_renderer: &mut imgui_wgpu::Renderer
    ) -> Self 
        where Self: Sized {
        RenderOutputView::init(device, queue, render_target_format, imgui_renderer, (800, 600))
    }
    
    fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    
        io: &imgui::Io,
    ) {
        self.update(io, device, queue);
    }
    
    fn render(
        &mut self, 
        command_encoder: &mut wgpu::CommandEncoder, 
        render_texture: &wgpu::TextureView,
    
        imgui_textures: &imgui::Textures<imgui_wgpu::Texture>,
    
        // in case view requires creating bind group or writing buffer / texture data
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        RenderOutputView::render(self, command_encoder, imgui_textures, device, queue, render_texture);
    }
    
    fn render_imgui(&mut self, ui: &mut imgui::Ui) {
        let viewer_window = ui.window("Viewer")
            .size([300.0, 300.0], imgui::Condition::Always)
            .position([0.0, 0.0], imgui::Condition::Always)
            .resizable(false);
        viewer_window.build(|| {
            ui.text("Pixel peeper");
            let tex_id = self.debug_texture_id;
            imgui::Image::new(tex_id, [200.0, 200.0]).build(ui);

            let index = self.gui_state.mouse_pos[1] as usize * self.render_output_size.0 as usize 
                    + self.gui_state.mouse_pos[0] as usize;
            let radiance_value = self.render_output[index];

            ui.text(format!("Radiance value: {radiance_value:?}"));
            ui.text(format!("Mouse position: ({}, {})", 
                self.gui_state.mouse_pos[0], 
                self.gui_state.mouse_pos[1]));

            ui.separator();
            ui.text("Render Output Settings");
            ui.input_float("Exposure", &mut self.gui_state.exposure).build();
            ui.input_float("Gamma", &mut self.gui_state.gamma).build();
            
            ui.separator();
            ui.text("Render");

            fn path_to_filename(path: &Path) -> &str {
                path.file_name().expect("should not be ..").to_str().expect("contains invalid utf8")
            }
            
            let selected_filename = path_to_filename(&self.gui_state.scenes[self.gui_state.selected_scene]);
            let combo = ui.begin_combo(
                "GLTF", 
                selected_filename
            );

            if let Some(combo) = combo {
                for (i, scene) in self.gui_state.scenes.iter().enumerate() {
                    if ui.selectable(path_to_filename(scene)) {
                        self.gui_state.selected_scene = i;
                    }
                }

                combo.end();
            }
            
            ui.input_scalar("spp", &mut self.gui_state.spp).build();
            self.gui_state.spp = u32::max(self.gui_state.spp, 1);

            ui.input_scalar("light samples", &mut self.gui_state.light_samples).build();
            self.gui_state.light_samples = u32::max(self.gui_state.light_samples, 1);

            self.gui_state.render_scene_requested = ui.button("Render");
        });
    }
}

fn enumerate_scenes() -> Vec<PathBuf> {
    // kind of a hack, but ok for now
    let viewer_crate_path = env!("CARGO_MANIFEST_DIR"); 
    let scenes_path = std::path::Path::new(viewer_crate_path)
        .parent().unwrap()
        .parent().unwrap()
        .join("scenes");

    scenes_path.read_dir()
        .expect("Failed to read scenes directory")
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "gltf" || ext == "glb"))
        .collect()
}

impl RenderOutputView {
    fn init(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        
        target_format: TextureFormat,

        imgui_renderer: &mut imgui_wgpu::Renderer,
        size: (u32, u32),
    ) -> RenderOutputView {
        
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

        let render_resources = Self::init_graphics_resources(device, target_format, size, &texture);
        let compute_resources = Self::init_compute_resources(device, size, &texture);

        // create debug texture
        let texture_config = imgui_wgpu::TextureConfig {
            size: wgpu::Extent3d { width: 200, height: 200, depth_or_array_layers: 1 },
            label: Some("Render Output Debug Texture"),
            // RENDER_ATTACHMENT needed for blit
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            ..Default::default()
        };
          
        let imgui_debug_texture = imgui_wgpu::Texture::new(
            &device,
            &imgui_renderer,
            texture_config
        );

        let dst_format = imgui_debug_texture.texture().format();
        let debug_texture_id = imgui_renderer.textures.insert(imgui_debug_texture);
        
        let imgui_blitter = wgpu::util::TextureBlitter::new(
            device,
            dst_format
        );

        let me = RenderOutputView {
            graphics: render_resources,
            compute: compute_resources,

            debug_texture_id,
            imgui_blitter,

            gui_state: GuiState { 
                gamma: 2.2, 
                exposure: 1.0, 
                mouse_pos: [0, 0],

                scenes: enumerate_scenes(),
                selected_scene: 0,
                render_scene_requested: false,

                spp: 1,
                light_samples: 1,
            },
            

            render_output_size: size, 
            render_output: Self::generate_test_pattern(size),
        };

        me.upload_radiance_buffer(queue);
        me
    }

    fn init_graphics_resources(
        device: &wgpu::Device,
        target_format: TextureFormat,
        size: (u32, u32),

        texture: &wgpu::Texture,
    ) -> GraphicsResources {
        let shader_module_descriptor = wgpu::ShaderModuleDescriptor {
            label: Some("Render Output Shaders"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/render_output_render.wgsl"))),
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

        let targets = &[Some(target_format.into())];
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

        GraphicsResources {
            pipeline: render_pipeline,
            triangle_buffer,
            bind_group,
        }
    }

    fn init_compute_resources(
        device: &wgpu::Device,
        size: (u32, u32),

        texture: &wgpu::Texture,
    ) -> ComputeResources {

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
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/render_output_compute.wgsl"))),
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
            main_compute_pipeline, 
            debug_compute_pipeline, 
            radiance_buffer,
            debug_texture, 
            compute_bind_group: bind_group
        }
    }

    fn update(&mut self, io: &imgui::Io, _device: &wgpu::Device, queue: &wgpu::Queue) {
        if io.mouse_pos != [f32::MAX, f32::MAX] {
            self.gui_state.mouse_pos = [
                io.mouse_pos[0] as u32,
                io.mouse_pos[1] as u32,
            ];
        }

        // If the user requested a render, we need to update the radiance buffer
        // For now, just blocks the entire viewer until raytracing is done
        if self.gui_state.render_scene_requested {
            let params = RaytraceParams {
                spp: self.gui_state.spp,
                light_samples: self.gui_state.light_samples,
            };
            let RaytracerResult { radiance, .. } = 
                raytrace_scene(&self.gui_state.scenes[self.gui_state.selected_scene], params);
            self.render_output = radiance;
            self.upload_radiance_buffer(queue);
            self.gui_state.render_scene_requested = false;    
        }
    }

    // Renders texture onto 2 triangles covering the screen.
    fn render(
        &self, 
        command_encoder: &mut wgpu::CommandEncoder,
        imgui_textures: &imgui::Textures<imgui_wgpu::Texture>,

        device: &wgpu::Device,
        _queue: &wgpu::Queue, 
        render_target: &wgpu::TextureView,
    ) {       
        self.do_compute(command_encoder);
        self.do_graphics(command_encoder, render_target);
        self.prepare_render_imgui(imgui_textures, device, command_encoder);
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

    fn do_graphics(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        let color_attachments = [
            Some(wgpu::RenderPassColorAttachment { 
                view, 
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
        rpass.set_pipeline(&self.graphics.pipeline);

        let indices = self.graphics.triangle_buffer.slice(..);
        rpass.set_index_buffer(indices, wgpu::IndexFormat::Uint32);
        rpass.set_bind_group(0, &self.graphics.bind_group, &[]);
        rpass.draw_indexed(0..6, 0, 0..1);
    }

    fn prepare_render_imgui(
        &self, 
        imgui_textures: &imgui::Textures<imgui_wgpu::Texture>,

        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder
    ) {
        let src_view = self.compute.debug_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let dst_view = imgui_textures.get(self.debug_texture_id)
            .expect("imgui texture not in textures map")
            .texture()
            .create_view(&wgpu::TextureViewDescriptor::default());
        
        let blitter = &self.imgui_blitter; 
        blitter.copy(
            device, 
            encoder,
            &src_view,
            &dst_view,
        );
    }

    fn resize(&mut self, width: u32, height: u32, wgpu_handles: &WgpuHandles) {
        todo!()    
    }

    fn generate_test_pattern(size: (u32, u32)) -> Vec<Vec3> {
        let elts = size.0 * size.1;
        let mut test_pattern: Vec<Vec3> = Vec::with_capacity(elts as usize);

        for j in 0..size.1 {
            for i in 0..size.0 {
                let r = (i as f32) / (size.0 as f32);
                let b = (j as f32) / (size.1 as f32);
                test_pattern.push(Vec3(r, 0.0, b));
            }
        }

        test_pattern
    }

    fn upload_radiance_buffer(&self, queue: &wgpu::Queue) {
        // Write radiance buffer to gpu
        let data: &[u8] = Self::cast_vec3_slice(&self.render_output);
        queue.write_buffer(&self.compute.radiance_buffer, 0, data);
    }

    fn cast_vec3_slice(slice: &[Vec3]) -> &[u8] {
        let data = slice.as_ptr() as *const u8;
        let len = std::mem::size_of_val(slice);

        // SAFETY: Vec3 is a POD type, so we can safely cast the slice to a byte slice
        unsafe {
            std::slice::from_raw_parts(data, len)
        }
    }
}
