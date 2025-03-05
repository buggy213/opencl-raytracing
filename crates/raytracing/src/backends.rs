use crate::scene::Scene;

// very basic interface - pass in Scene object, call render and get linear radiance values out
trait RenderingBackend {
    fn load_scene(&self, scene: &Scene);
    fn render(&self);
}