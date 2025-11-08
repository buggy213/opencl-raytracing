use std::{borrow::Cow, sync::Arc, time::{Duration, Instant}};

use imgui::MouseCursor;
use imgui_wgpu::{Renderer, RendererConfig};
use imgui_winit_support::WinitPlatform;
use pollster::FutureExt;
use wgpu::TextureFormat;
use winit::{application::ApplicationHandler, dpi::{LogicalSize, PhysicalSize}, event::{Event, WindowEvent}, event_loop::{ActiveEventLoop, EventLoop}, window::Window};

use crate::{demo_view::DemoApplicationView, render_output_view::RenderOutputView, scene_view::SceneView};

mod demo_view;
mod scene_view;
mod render_output_view;

// Wrapper to handle first `resumed` event and initialize Application with created window
struct ApplicationWrapper {
    inner: Option<Application>
}

impl ApplicationWrapper {
    fn new() -> Self {
        Self { 
            inner: None 
        }
    }
}

struct Application {
    window: Arc<Window>,
    logical_size: LogicalSize<f64>,

    wgpu_handles: WgpuHandles<'static>,
    
    imgui_state: ImguiInternalState,
    last_frame: Instant,
    
    current_view: Box<dyn RenderView>,
}

struct WgpuHandles<'window> {
    surface: wgpu::Surface<'window>,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,

    // intermediate draw texture which gets blitted to swapchain image as final step
    // in rendering a frame
    draw_texture: wgpu::Texture,
    blitter: wgpu::util::TextureBlitter,
}

struct ImguiInternalState {
    context: imgui::Context,
    platform: WinitPlatform,
    renderer: Renderer,
    last_cursor: Option<MouseCursor>,
    last_frame: Instant,
}

#[derive(Default)]
struct WindowRequests {
    resize: Option<LogicalSize<f64>>,
    rename: Option<Cow<'static, str>>,
    resizable: Option<bool>,
}

impl WindowRequests {
    fn apply(&self, window: &Window) {
        if let Some(new_size) = self.resize {
            if new_size != window.inner_size().to_logical(window.scale_factor()) {
                _ = window.request_inner_size(new_size);
            }
        }

        if let Some(ref new_name) = self.rename {
            if *new_name != window.title() {
                window.set_title(new_name);
            }
        }

        if let Some(resizable) = self.resizable {
            if resizable != window.is_resizable() {
                window.set_resizable(resizable);
            }
        }
    }
}

trait RenderView {
    fn init(
        device: &wgpu::Device,
        queue: &wgpu::Queue,

        render_target_format: TextureFormat,

        // imgui has its own texture management system, which we need to work with
        imgui_renderer: &mut imgui_wgpu::Renderer
    ) -> (Self, WindowRequests) 
        where Self: Sized;

    // For now just piggybacks off of imgui's IO abstraction
    fn update(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,

        _io: &imgui::Io,
    ) -> WindowRequests {
        Default::default()
    }

    fn render(
        &mut self, 
        command_encoder: &mut wgpu::CommandEncoder, 
        render_texture: &wgpu::TextureView,

        imgui_textures: &imgui::Textures<imgui_wgpu::Texture>,

        // in case view requires creating bind group or writing buffer / texture data
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    );
    
    fn render_imgui(&mut self, ui: &mut imgui::Ui);
}


impl Application {
    fn new(window: Window) -> Self {
        let window = Arc::new(window);
        let wgpu_handles = Self::init_wgpu(Arc::clone(&window));
        let mut imgui_state = Self::init_imgui(&window, &wgpu_handles);
        
        let target_format = wgpu_handles.draw_texture.format();
        
        let (demo_view, initial_requests) = DemoApplicationView::init(
            &wgpu_handles.device,
            &wgpu_handles.queue,
            target_format,
            &mut imgui_state.renderer
        );

        let (render_output_view, initial_requests) = RenderOutputView::init(
            &wgpu_handles.device,
            &wgpu_handles.queue,
            target_format,
            &mut imgui_state.renderer
        );

        let (scene_view, initial_requests) = SceneView::init(&wgpu_handles.device,
            &wgpu_handles.queue,
            target_format,
            &mut imgui_state.renderer
        );
        
        let logical_size = window.inner_size().to_logical(window.scale_factor());
        initial_requests.apply(&window);

        Self {
            window,
            logical_size,

            wgpu_handles,
            last_frame: Instant::now(),

            imgui_state,
            
            current_view: Box::new(render_output_view)
        }
    }

    fn init_wgpu(window: Arc<Window>) -> WgpuHandles<'static> {
        let instance_descriptor = wgpu::InstanceDescriptor::from_env_or_default();
        let instance = wgpu::Instance::new(&instance_descriptor);

        let window_clone = Arc::clone(&window);
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

        
        // arbitrarily choose 1st "allowed" swapchain format
        let swapchain_capabilities = surface.get_capabilities(&adapter);
        let swapchain_format = swapchain_capabilities.formats[0];
        println!("Using swapchain format {swapchain_format:?}");
        
        let physical_size = window.inner_size();
        let config = surface
            .get_default_config(&adapter, physical_size.width, physical_size.height)
            .expect("Unable to get surface configuration");

        surface.configure(&device, &config);

        // don't use swapchain directly, draw to texture then blit to swapchain
        let blitter = wgpu::util::TextureBlitter::new(&device, swapchain_format);
        let draw_texture = Self::make_draw_texture(physical_size, &device);

        WgpuHandles {
            surface,
            adapter,
            device,
            queue,
            blitter,
            draw_texture,
        }
    }

    fn make_draw_texture(
        physical_size: PhysicalSize<u32>,
        device: &wgpu::Device,
    ) -> wgpu::Texture {
        let texture_desc = wgpu::TextureDescriptor {
            label: Some("Draw Texture"),
            size: wgpu::Extent3d {
                width: physical_size.width,
                height: physical_size.height,
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

    fn init_imgui(window: &Window, wgpu_handles: &WgpuHandles) -> ImguiInternalState {
        let mut context = imgui::Context::create();
        let mut platform = imgui_winit_support::WinitPlatform::new(&mut context);
        
        platform.attach_window(
            context.io_mut(), 
            window, 
            imgui_winit_support::HiDpiMode::Default
        );

        context.set_ini_filename(None);

        let texture_format = wgpu_handles.draw_texture.format();

        let renderer_config = RendererConfig {
            texture_format,
            ..Default::default()
        };

        let renderer = Renderer::new(&mut context, &wgpu_handles.device, &wgpu_handles.queue, renderer_config);
        
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
}

impl Application {
    fn resize(&mut self, physical_size: PhysicalSize<u32>) {
        let wgpu_handles = &mut self.wgpu_handles;
        let config = wgpu_handles.surface.get_default_config(&wgpu_handles.adapter, physical_size.width, physical_size.height).expect("surface not supported");
        
        // surface width / height must be nonzero
        if physical_size.width != 0 && physical_size.height != 0 {
            wgpu_handles.surface.configure(&wgpu_handles.device, &config);
        }

        // recreate the draw texture
        let new_draw_texture = Self::make_draw_texture(physical_size, &wgpu_handles.device);
        wgpu_handles.draw_texture = new_draw_texture;
    }

    fn render_imgui(
        &mut self,
    ) {
        let now = Instant::now();
        let imgui_state = &mut self.imgui_state;
        imgui_state
            .context
            .io_mut()
            .update_delta_time(now - imgui_state.last_frame);
        imgui_state.last_frame = now;

        imgui_state
            .platform
            .prepare_frame(imgui_state.context.io_mut(), &self.window)
            .expect("Failed to prepare frame");
        
        let ui = imgui_state.context.new_frame();
        self.current_view.render_imgui(ui);

        let WgpuHandles { 
            device, queue, ..
        } = &self.wgpu_handles;

        let mut encoder: wgpu::CommandEncoder = device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        if imgui_state.last_cursor != ui.mouse_cursor() {
            imgui_state.last_cursor = ui.mouse_cursor();
            imgui_state.platform.prepare_render(ui, &self.window);
        }

        let view = self.wgpu_handles.draw_texture
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

    fn render(
        &mut self,
    ) {
        let view_descriptor = wgpu::TextureViewDescriptor::default();
        let view = self.wgpu_handles.draw_texture.create_view(&view_descriptor);

        let WgpuHandles { device, queue, .. } = &self.wgpu_handles;

        let encoder_descriptor = wgpu::CommandEncoderDescriptor {
            label: Some("Render Output Command Encoder"),
        };

        let mut encoder = device.create_command_encoder(&encoder_descriptor);
        
        self.current_view.render(
            &mut encoder, 
            &view, 
            &self.imgui_state.renderer.textures, 
            device, 
            queue);
        
        queue.submit(Some(encoder.finish()));
    }

    fn update(&mut self) -> WindowRequests {
        self.current_view.update(
            &self.wgpu_handles.device, 
            &self.wgpu_handles.queue, 
            self.imgui_state.context.io()
        )
    }

    fn draw_frame(&mut self) {
        let frame = self.wgpu_handles.surface.get_current_texture()
            .expect("Unable to get next swapchain image");
        
        let window_requests = self.update();
        self.handle_window_requests(&window_requests);
        self.render();
        self.render_imgui();
        
        // blit the rendered output to the swapchain image
        let mut blit_encoder = self.wgpu_handles.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Blit Encoder") }
        );
        let source_view = self.wgpu_handles.draw_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let dest_view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.wgpu_handles.blitter.copy(
            &self.wgpu_handles.device, 
            &mut blit_encoder, 
            &source_view, 
            &dest_view
        );
        self.wgpu_handles.queue.submit(Some(blit_encoder.finish()));

        frame.present();
    }

    fn handle_window_requests(&self, window_requests: &WindowRequests) {
        window_requests.apply(&self.window);
    }
}

impl ApplicationHandler for ApplicationWrapper {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.inner.is_none() {
            let default_height = 500;
            let default_width = 500;

            let window_attributes = Window::default_attributes()
                .with_title("Viewer")
                .with_inner_size(LogicalSize::new(default_width, default_height));


            let window = event_loop.create_window(window_attributes)
                .expect("failed to create window");

            let application = Application::new(window);

            self.inner = Some(application);
        }

        self.inner.as_mut().unwrap().resumed(event_loop);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        self.inner.as_mut().unwrap().window_event(event_loop, window_id, event);
    }
}

impl ApplicationHandler for Application {
    fn resumed(&mut self, _event_loop: &ActiveEventLoop) {
        
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                println!("Window close requested");
                
                event_loop.exit();
            }
            
            WindowEvent::RedrawRequested if !event_loop.exiting() => {
                let now = Instant::now();
                if now.duration_since(self.last_frame) >= Duration::from_secs_f32(1.0 / 60.0) {
                    self.draw_frame();
                    self.last_frame = now;
                    self.window.request_redraw();
                }
                else {
                    self.window.request_redraw();
                }
            }
            
            WindowEvent::Resized(new_size) => {
                let logical_size: LogicalSize<f64> = new_size.to_logical(self.window.scale_factor());
                println!("Window resized to: {logical_size:?}, (physical: {new_size:?})");
                self.logical_size = logical_size;
                self.resize(new_size);
            }

            _ => ()
        }

        let imgui_state = &mut self.imgui_state;
        
        imgui_state.platform.handle_event::<()>(
            imgui_state.context.io_mut(),
            &self.window,
            &Event::WindowEvent { window_id, event: event.clone() },
        );
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("Unable to create event loop");
    let mut app = ApplicationWrapper::new();
    
    event_loop.run_app(&mut app).expect("Unable to run application");
}