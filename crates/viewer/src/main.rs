use std::{borrow::Cow, sync::Arc, time::{Duration, Instant}};

use imgui::{Condition, MouseCursor};
use imgui_wgpu::{Renderer, RendererConfig};
use imgui_winit_support::WinitPlatform;
use pollster::FutureExt;
use render_output_view::RenderOutputView;
use winit::{application::ApplicationHandler, dpi::PhysicalSize, event::{Event, WindowEvent}, event_loop::{ActiveEventLoop, EventLoop}, window::Window};

mod render_output_view;

struct Application {
    window: Option<Arc<Window>>,
    width: u32,
    height: u32,

    wgpu_handles: Option<WgpuHandles<'static>>,
    
    imgui_state: Option<ImguiInternalState>,
    demo_state: DemoApplicationState,

    render_output_view: Option<RenderOutputView>
}

struct RequestWindowUpdate {
    resize: Option<(u32, u32)>
}

struct WgpuHandles<'window> {
    instance: wgpu::Instance,
    surface: wgpu::Surface<'window>,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,

    blitter: wgpu::util::TextureBlitter,
    draw_texture: wgpu::Texture,

    // specific to current rendering pipeline
    shader: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,
    pipeline: wgpu::RenderPipeline,
}

struct ImguiInternalState {
    context: imgui::Context,
    platform: WinitPlatform,
    renderer: Renderer,
    last_cursor: Option<MouseCursor>,
    last_frame: Instant,
}

trait RenderGui {
    fn render_imgui(&mut self, ui: &mut imgui::Ui);
}

struct DemoApplicationState {
    demo_open: bool,
    delta_s: Duration
}

impl RenderGui for DemoApplicationState {
    fn render_imgui(&mut self, ui: &mut imgui::Ui) {
    
        let window = ui.window("Hello world");
        window
            .size([300.0, 100.0], Condition::FirstUseEver)
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

        let delta_s = self.delta_s;
        let window = ui.window("Hello too");
        window
            .size([400.0, 200.0], Condition::FirstUseEver)
            .position([400.0, 200.0], Condition::FirstUseEver)
            .build(|| {
                ui.text(format!("Frametime: {delta_s:?}"));
            });

        ui.show_demo_window(&mut self.demo_open);
    }
}

impl Application {
    fn new() -> Self {
        Self {
            window: None,
            width: 800,
            height: 600,

            wgpu_handles: None,

            imgui_state: None,
            demo_state: DemoApplicationState { demo_open: true, delta_s: Duration::ZERO },

            render_output_view: None,
        }
    }
}

impl Application {
    fn init_wgpu(&self) -> WgpuHandles<'static> {
        let instance_descriptor = wgpu::InstanceDescriptor::from_env_or_default();
        let instance = wgpu::Instance::new(&instance_descriptor);

        let window_clone = Arc::clone(self.window.as_ref().unwrap());
        let surface = instance.create_surface(window_clone)
            .expect("Unable to create surface");

        let request_adapter_options = wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: Some(&surface)
        };

        let adapter = instance.request_adapter(&request_adapter_options)
            .block_on()
            .expect("Unable to create adapter (physical device)");

        let required_features = wgpu::Features::FLOAT32_FILTERABLE
            | wgpu::Features::PUSH_CONSTANTS;

        let limits = wgpu::Limits {
            max_push_constant_size: 128, // big enough for our use case
            ..wgpu::Limits::default()
        };

        let device_descriptor = wgpu::DeviceDescriptor {
            label: Some("Main Device"),
            required_features,
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::default(),
        };

        let (device, queue) = adapter.request_device(&device_descriptor, None)
            .block_on()
            .expect("Unable to get device (logical device)");

        
        let shader_module_descriptor = wgpu::ShaderModuleDescriptor {
            label: Some("Shaders"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        };

        let shader = device.create_shader_module(shader_module_descriptor);

        let pipeline_layout_descriptor = wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        };
        
        let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_descriptor);
        
        // arbitrarily choose 1st "allowed" swapchain format
        let swapchain_capabilities = surface.get_capabilities(&adapter);
        let swapchain_format = swapchain_capabilities.formats[0];
        println!("Using swapchain format {swapchain_format:?}");
        
        let render_targets = [Some(swapchain_format.into())];

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
                targets: &render_targets,
            }), // what does it mean to have no fragment stage (??)
            multiview: None, // what is a multiview render pass :skull:
            cache: None,
        };

        let render_pipeline = device.create_render_pipeline(&render_pipeline_descriptor);

        let config = surface
            .get_default_config(&adapter, self.width, self.height)
            .expect("Unable to get surface configuration");

        surface.configure(&device, &config);

        // don't use swapchain directly, draw to texture then blit to swapchain
        let blitter = wgpu::util::TextureBlitter::new(&device, swapchain_format);
        let draw_texture = Self::make_draw_texture(self.width, self.height, &device);

        WgpuHandles {
            instance,
            surface,
            adapter,
            device,
            queue,
            blitter,
            draw_texture,
            shader,
            pipeline_layout,
            pipeline: render_pipeline,
        }
    }

    fn make_draw_texture(
        width: u32,
        height: u32,
        device: &wgpu::Device,
    ) -> wgpu::Texture {
        let texture_desc = wgpu::TextureDescriptor {
            label: Some("Draw Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            // need TEXTURE_BINDING for blitting to swapchain
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
        };
        device.create_texture(&texture_desc)
    }

    fn init_imgui(&self) -> ImguiInternalState {
        let mut context = imgui::Context::create();
        let mut platform = imgui_winit_support::WinitPlatform::new(&mut context);
        let window = self.window.as_ref().expect("window not created yet");
        let handles = self.wgpu_handles.as_ref().expect("wgpu not initialized");
        platform.attach_window(
            context.io_mut(), 
            window, 
            // ensure mouse coordinates are in physical pixels
            // TODO: maybe think about using our own input handling so this doesn't matter
            imgui_winit_support::HiDpiMode::Locked(1.0)
        );

        context.set_ini_filename(None);

        let texture_format = handles.draw_texture.format();

        let renderer_config = RendererConfig {
            texture_format,
            ..Default::default()
        };

        let renderer = Renderer::new(&mut context, &handles.device, &handles.queue, renderer_config);
        
        let last_frame = Instant::now();
        let last_cursor = None;
        
        ImguiInternalState { 
            context, 
            platform, 
            renderer,  
            last_cursor, 
            last_frame
        }
        
    }

    fn resize(&mut self, height: u32, width: u32) {
        let wgpu_handles = self.wgpu_handles.as_mut().expect("wgpu not initialized");
        let config = wgpu_handles.surface.get_default_config(&wgpu_handles.adapter, width, height).expect("surface not supported");
        
        // surface width / height must be nonzero
        if height != 0 && width != 0 {
            wgpu_handles.surface.configure(&wgpu_handles.device, &config);
        }

        // recreate the draw texture
        let new_draw_texture = Self::make_draw_texture(width, height, &wgpu_handles.device);
        wgpu_handles.draw_texture = new_draw_texture;
    }

    fn render_imgui(
        imgui_state: &mut ImguiInternalState, 
        render_gui: &mut dyn RenderGui,
        wgpu_handles: &WgpuHandles, 
        render_target: &wgpu::Texture, 
        window: &Window
    ) {
        let now = Instant::now();
        imgui_state
            .context
            .io_mut()
            .update_delta_time(now - imgui_state.last_frame);
        imgui_state.last_frame = now;

        imgui_state
            .platform
            .prepare_frame(imgui_state.context.io_mut(), window)
            .expect("Failed to prepare frame");
        
        let ui = imgui_state.context.new_frame();
        render_gui.render_imgui(ui);

        let WgpuHandles { 
            device, queue, ..
        } = wgpu_handles;

        let mut encoder: wgpu::CommandEncoder = device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        if imgui_state.last_cursor != ui.mouse_cursor() {
            imgui_state.last_cursor = ui.mouse_cursor();
            imgui_state.platform.prepare_render(ui, window);
        }

        let view = render_target
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        imgui_state.renderer
            .render(
                imgui_state.context.render(),
                queue,
                device,
                &mut rpass,
            )
            .expect("Rendering failed");

        drop(rpass);

        queue.submit(Some(encoder.finish()));
    }

    fn render(wgpu_handles: &WgpuHandles, render_target: &wgpu::Texture) {
        let WgpuHandles { 
            device, 
            queue, 
            pipeline,
            ..
        } = wgpu_handles;

        let frame_view_descriptor = wgpu::TextureViewDescriptor::default();
        let view = render_target.create_view(&frame_view_descriptor);

        let encoder_descriptor = wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        };
        let mut encoder = device.create_command_encoder(&encoder_descriptor);
        {
            let color_attachments = [
                Some(wgpu::RenderPassColorAttachment { 
                    view: &view, 
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
            let mut rpass = encoder.begin_render_pass(&rpass_descriptor);
            rpass.set_pipeline(pipeline);
            rpass.draw(0..3, 0..1)
            
        }

        queue.submit(Some(encoder.finish()));
    }
}

impl ApplicationHandler for Application {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window_attributes = Window::default_attributes()
                .with_title("Viewer")
                .with_inner_size(PhysicalSize::new(self.width, self.height));
            let window = event_loop.create_window(window_attributes).expect("Unable to create window");
            self.window = Some(Arc::new(window));

            self.wgpu_handles = Some(self.init_wgpu());
            self.imgui_state = Some(self.init_imgui());

            let wgpu_handles = self.wgpu_handles.as_ref().unwrap();
            let targets: &[Option<wgpu::ColorTargetState>] = &[Some(wgpu_handles.draw_texture.format().into())];
            let size = (self.width, self.height);

            let mut render_output_view = RenderOutputView::init(wgpu_handles, targets, size);
            render_output_view.init_imgui(wgpu_handles, self.imgui_state.as_mut().unwrap());

            self.render_output_view = Some(render_output_view)
            
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                println!("Window close requested");
                
                // prevent segfault by tearing down wgpu first, then closing window
                let wgpu_handles = std::mem::take(&mut self.wgpu_handles);
                drop(wgpu_handles);

                event_loop.exit();
            }
            
            WindowEvent::RedrawRequested if !event_loop.exiting() => {
                let wgpu_handles = self.wgpu_handles.as_ref().unwrap();
                let imgui_state = self.imgui_state.as_mut().unwrap();
                let frame = wgpu_handles.surface.get_current_texture()
                    .expect("Unable to get next swapchain image");
                let window = self.window.as_ref().unwrap();
                
                // Self::render(wgpu_handles, &frame);

                let render_output_view = self.render_output_view.as_mut().unwrap();
                render_output_view.update(imgui_state.context.io(), wgpu_handles);
                render_output_view.render(
                    &imgui_state.renderer.textures,
                    wgpu_handles, 
                    &wgpu_handles.draw_texture
                );

                let now = Instant::now();
                self.demo_state.delta_s = now - imgui_state.last_frame;

                Self::render_imgui(
                    imgui_state, 
                    render_output_view,
                    wgpu_handles, 
                    &wgpu_handles.draw_texture, 
                    window
                );

                // blit the rendered output to the swapchain image
                let mut blit_encoder = wgpu_handles.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("Blit Encoder") }
                );
                let source_view = wgpu_handles.draw_texture.create_view(&wgpu::TextureViewDescriptor::default());
                let dest_view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                wgpu_handles.blitter.copy(
                    &wgpu_handles.device, 
                    &mut blit_encoder, 
                    &source_view, 
                    &dest_view
                );
                wgpu_handles.queue.submit(Some(blit_encoder.finish()));

                frame.present();

                // request redraw for the next frame
                self.window.as_ref().unwrap().request_redraw();
            }
            
            WindowEvent::Resized(new_size) => {
                println!("Window resized to: {new_size:?}");
                self.height = new_size.height;
                self.width = new_size.width;
                self.resize(self.height, self.width);
            }

            _ => ()
        }

        let imgui_state = self.imgui_state.as_mut().unwrap();
        let window = self.window.as_ref().unwrap();
        
        imgui_state.platform.handle_event::<()>(
            imgui_state.context.io_mut(),
            window,
            &Event::WindowEvent { window_id, event: event.clone() },
        );
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("Unable to create event loop");
    let mut app = Application::new();
    
    event_loop.run_app(&mut app).expect("Unable to run application");
}