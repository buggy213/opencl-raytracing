use winit::{application::ApplicationHandler, event_loop::{ActiveEventLoop, EventLoop}, window::{Window, WindowAttributes}};

struct Application {
    window: Option<Window>
}

impl Application {
    fn new() -> Self {
        Self {
            window: None
        }
    }
}

impl ApplicationHandler for Application {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window_attributes = Window::default_attributes().with_title("Viewer");
            let window = event_loop.create_window(window_attributes).expect("Unable to create window");
            self.window = Some(window);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        println!("Received event");
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("Unable to create event loop");
    let mut app = Application::new();
    event_loop.run_app(&mut app).expect("Unable to run application");
}