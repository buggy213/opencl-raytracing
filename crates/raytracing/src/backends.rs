use crate::scene::Scene;

// very basic interface - pass in Scene object and get linear radiance values out
pub trait RenderingBackend {
    fn render(&self, scene: &Scene);
}