use raytracing::scene;
use raytracing_optix::render;

fn main() {
    let sphere_scene = scene::test_scenes::sphere_scene();
    render(&sphere_scene);
}