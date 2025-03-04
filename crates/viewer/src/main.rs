use std::{borrow::Cow, rc::Rc, sync::Arc};

use pollster::FutureExt;
use winit::{application::ApplicationHandler, dpi::PhysicalSize, event::WindowEvent, event_loop::{ActiveEventLoop, ControlFlow, EventLoop}, window::Window};

struct Application {
    window: Option<Arc<Window>>,
    width: u32,
    height: u32,

    wgpu_handles: Option<WgpuHandles<'static>>
}

struct WgpuHandles<'window> {
    instance: wgpu::Instance,
    surface: wgpu::Surface<'window>,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,

    // specific to current rendering pipeline
    shader: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,
    pipeline: wgpu::RenderPipeline,
}

impl Application {
    fn new() -> Self {
        Self {
            window: None,
            width: 800,
            height: 600,

            wgpu_handles: None
        }
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

            let device_descriptor = wgpu::DeviceDescriptor {
                label: Some("Main Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
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

            self.wgpu_handles = Some(WgpuHandles {
                instance,
                surface,
                adapter,
                device,
                queue,
                shader,
                pipeline_layout,
                pipeline: render_pipeline,
            })
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
                
                // prevent segfault by tearing down wgpu first
                std::mem::take(&mut self.wgpu_handles);

                event_loop.exit();
            }
            
            WindowEvent::RedrawRequested if !event_loop.exiting() => {
                let wgpu_handles = self.wgpu_handles.as_ref().unwrap();
                let WgpuHandles { instance, surface, adapter, device, queue, shader, pipeline_layout, pipeline } = wgpu_handles;

                let frame = surface.get_current_texture().expect("Unable to get next swapchain image");

                let frame_view_descriptor = wgpu::TextureViewDescriptor::default();
                let view = frame.texture.create_view(&&frame_view_descriptor);

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
                frame.present();

                // request redraw for the next frame
                self.window.as_ref().unwrap().request_redraw();
            }

            _ => ()
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("Unable to create event loop");
    let mut app = Application::new();
    
    event_loop.run_app(&mut app).expect("Unable to run application");
}