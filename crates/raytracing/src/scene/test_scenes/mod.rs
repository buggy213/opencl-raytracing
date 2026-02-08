//! Basic test scenes which act as a "smoke test" for the renderer through `rttest` in visual-testing 
//! 
//! Assets for these test scenes are included inline within source directory under `assets` for convenience

use crate::{
    geometry::{Mesh, Shape, Vec2, Vec3, Vec3u, Vec4},
    lights::{EnvironmentLight, Light},
    materials::{Image, Material, Texture, TextureSampler},
    renderer::{AOVFlags, RaytracerSettings},
    sampling::Sampler,
    scene::{Camera, Scene, SceneBuilder},
};

mod assets {
    pub(super) const LAKE_PIER_ENVIRONMENT: &[u8] = include_bytes!("assets/lake_pier_1k.exr");
    pub(super) const BUNNY_PLY: &[u8] = include_bytes!("assets/bunny.ply");
}

// helpers for basic meshes
fn make_mesh(verts: &[Vec3], tris: &[Vec3u], normals: &[Vec3]) -> Mesh {
    Mesh {
        vertices: verts.to_vec(),
        tris: tris.to_vec(),
        normals: normals.to_vec(),
        uvs: Vec::new(),
    }
}

fn make_plane(a: Vec3, b: Vec3, c: Vec3, d: Vec3, normal: Vec3) -> Mesh {
    // ensure a, b, c, d are coplanar, counterclockwise
    let ab = b - a;
    let ac = c - a;
    let x = Vec3::cross(ab, ac).unit();

    let cd = d - c;
    let ca = -ac;
    let y = Vec3::cross(cd, ca).unit();

    assert!(
        (x - normal).near_zero(),
        "points not in plane defined by normal"
    );
    assert!(
        (y - normal).near_zero(),
        "points not in plane defined by normal"
    );

    make_mesh(
        &[a, b, c, d],
        &[Vec3u(0, 1, 2), Vec3u(2, 3, 0)],
        &[normal, normal, normal, normal],
    )
}

#[rustfmt::skip]
fn make_cube(side_length: f32) -> Mesh {
    let h = side_length / 2.0;

    // 6 faces, each with 4 vertices and 2 triangles
    // Vertices are duplicated per face for correct flat-shaded normals
    let mut vertices = Vec::with_capacity(24);
    let mut normals = Vec::with_capacity(24);
    let mut tris = Vec::with_capacity(12);

    // Face order: +X, -X, +Y, -Y, +Z, -Z
    // Each face defined CCW when looking at the face from outside

    // +X face (normal = +X)
    let base = vertices.len() as u32;

    vertices.extend_from_slice(&[
        Vec3(h, -h, -h),
        Vec3(h,  h, -h),
        Vec3(h,  h,  h),
        Vec3(h, -h,  h),
    ]);
    normals.extend_from_slice(&[Vec3(1.0, 0.0, 0.0); 4]);
    tris.push(Vec3u(base, base + 1, base + 2));
    tris.push(Vec3u(base, base + 2, base + 3));

    // -X face (normal = -X)
    let base = vertices.len() as u32;
    vertices.extend_from_slice(&[
        Vec3(-h,  h, -h),
        Vec3(-h, -h, -h),
        Vec3(-h, -h,  h),
        Vec3(-h,  h,  h),
    ]);
    normals.extend_from_slice(&[Vec3(-1.0, 0.0, 0.0); 4]);
    tris.push(Vec3u(base, base + 1, base + 2));
    tris.push(Vec3u(base, base + 2, base + 3));

    // +Y face (normal = +Y)
    let base = vertices.len() as u32;
    vertices.extend_from_slice(&[
        Vec3( h, h, -h),
        Vec3(-h, h, -h),
        Vec3(-h, h,  h),
        Vec3( h, h,  h),
    ]);
    normals.extend_from_slice(&[Vec3(0.0, 1.0, 0.0); 4]);
    tris.push(Vec3u(base, base + 1, base + 2));
    tris.push(Vec3u(base, base + 2, base + 3));

    // -Y face (normal = -Y)
    let base = vertices.len() as u32;
    vertices.extend_from_slice(&[
        Vec3(-h, -h, -h),
        Vec3( h, -h, -h),
        Vec3( h, -h,  h),
        Vec3(-h, -h,  h),
    ]);
    normals.extend_from_slice(&[Vec3(0.0, -1.0, 0.0); 4]);
    tris.push(Vec3u(base, base + 1, base + 2));
    tris.push(Vec3u(base, base + 2, base + 3));

    // +Z face (normal = +Z)
    let base = vertices.len() as u32;
    vertices.extend_from_slice(&[
        Vec3(-h, -h, h),
        Vec3( h, -h, h),
        Vec3( h,  h, h),
        Vec3(-h,  h, h),
    ]);
    normals.extend_from_slice(&[Vec3(0.0, 0.0, 1.0); 4]);
    tris.push(Vec3u(base, base + 1, base + 2));
    tris.push(Vec3u(base, base + 2, base + 3));

    // -Z face (normal = -Z)
    let base = vertices.len() as u32;
    vertices.extend_from_slice(&[
        Vec3( h, -h, -h),
        Vec3(-h, -h, -h),
        Vec3(-h,  h, -h),
        Vec3( h,  h, -h),
    ]);
    normals.extend_from_slice(&[Vec3(0.0, 0.0, -1.0); 4]);
    tris.push(Vec3u(base, base + 1, base + 2));
    tris.push(Vec3u(base, base + 2, base + 3));

    Mesh {
        vertices,
        tris,
        normals,
        uvs: Vec::new(),
    }
}

// super simple test scene with one sphere, no light
pub fn sphere_scene() -> Scene {
    let mut scene_builder = SceneBuilder::new();

    let sphere = Shape::Sphere {
        center: Vec3::zero(),
        radius: 1.0,
    };

    let white = scene_builder.add_constant_texture(Vec4(1.0, 1.0, 1.0, 1.0));

    let white_diffuse = scene_builder.add_material(Material::Diffuse { albedo: white });

    scene_builder.add_shape_at_position(sphere, white_diffuse, Vec3(0.0, 0.0, -3.0));

    let camera = Camera::lookat_camera_perspective(
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.0, 0.0, -3.0),
        Vec3(0.0, 1.0, 0.0),
        false,
        (45.0_f32).to_radians(),
        400,
        400,
    );
    scene_builder.add_camera(camera);

    scene_builder.build()
}

pub fn cube_scene() -> Scene {
    let mut scene_builder = SceneBuilder::new();

    let cube_mesh = make_cube(1.0);
    let cube = Shape::TriangleMesh(cube_mesh);

    let white = scene_builder.add_constant_texture(Vec4(1.0, 1.0, 1.0, 1.0));

    let white_diffuse = scene_builder.add_material(Material::Diffuse { albedo: white });

    scene_builder.add_shape_at_position(cube, white_diffuse, Vec3(0.0, 0.0, -3.0));

    let camera = Camera::lookat_camera_perspective(
        Vec3(1.0, 0.75, -1.0),
        Vec3(0.0, 0.0, -3.0),
        Vec3(0.0, 1.0, 0.0),
        false,
        (45.0_f32).to_radians(),
        400,
        400,
    );
    scene_builder.add_camera(camera);

    scene_builder.build()
}

pub fn cube_orthographic_scene() -> Scene {
    let mut scene_builder = SceneBuilder::new();

    let cube_mesh = make_cube(1.0);
    let cube = Shape::TriangleMesh(cube_mesh);

    let white = scene_builder.add_constant_texture(Vec4(1.0, 1.0, 1.0, 1.0));

    let white_diffuse = scene_builder.add_material(Material::Diffuse { albedo: white });

    scene_builder.add_shape_at_position(cube, white_diffuse, Vec3(0.0, 0.0, -3.0));

    let camera = Camera::lookat_camera_orthographic(
        Vec3(1.0, 0.75, -1.0),
        Vec3(0.0, 0.0, -3.0),
        Vec3(0.0, 1.0, 0.0),
        false,
        400,
        400,
        2.5 / 400.0,
    );

    scene_builder.add_camera(camera);

    scene_builder.build()
}

pub fn checkered_plane_scene() -> Scene {
    let mut scene_builder = SceneBuilder::new();

    let plane = {
        let mut plane = make_plane(
            Vec3(-100.0, -100.0, 0.1),
            Vec3(100.0, -100.0, 0.1),
            Vec3(100.0, 100.0, 0.1),
            Vec3(-100.0, 100.0, 0.1),
            Vec3(0.0, 0.0, 1.0),
        );

        plane.uvs = vec![
            Vec2(-500.0, -500.0),
            Vec2(500.0, -500.0),
            Vec2(500.0, 500.0),
            Vec2(-500.0, 500.0),
        ];

        plane
    };
    let plane_shape = Shape::TriangleMesh(plane);

    let checker_tex = Texture::CheckerTexture {
        color1: Vec4(0.0, 0.0, 0.0, 1.0),
        color2: Vec4(1.0, 1.0, 1.0, 1.0),
    };
    let checker_tex_id = scene_builder.add_texture(checker_tex);
    let checker_material = Material::Diffuse {
        albedo: checker_tex_id,
    };

    let checker_material_id = scene_builder.add_material(checker_material);

    scene_builder.add_shape_at_position(plane_shape, checker_material_id, Vec3::zero());

    // light from straight up
    let sun = Light::DirectionLight {
        direction: Vec3(0.0, 0.0, -1.0),
        radiance: Vec3(1000.0, 1000.0, 1000.0),
    };
    scene_builder.add_light(sun);

    // angle below +y axis
    let y_angle = (10.0_f32).to_radians();
    let lookat_dist = 1.0;

    let camera = Camera::lookat_camera_perspective(
        Vec3(0.0, 0.0, 0.22),
        Vec3(
            0.0,
            f32::cos(y_angle) * lookat_dist,
            0.22 - f32::sin(y_angle) * lookat_dist,
        ),
        Vec3(0.0, 0.0, 1.0),
        false,
        (40.0_f32).to_radians(),
        480,
        270,
    );
    scene_builder.add_camera(camera);

    scene_builder.build()
}

// template for cornell box, returns a SceneBuilder so other functions can add on top
// note: this isn't the same dimensions as the real cornell box
#[rustfmt::skip]
fn cornell_box() -> SceneBuilder {
    let mut scene_builder = SceneBuilder::new();

    // Dimensions: width=2, height=1.5, depth=2.0, y-up
    let w = 2.0;
    let h = 1.5;
    let d = 2.0;

    // Box corners
    let left   = w / 2.0;
    let right  =  -w / 2.0;
    let bottom = 0.0;
    let top    = h;
    let back   = -d / 2.0;
    let front  =  d / 2.0;

    // Plane normals
    let up    = Vec3(0.0, 0.0, 1.0);
    let down  = Vec3(0.0, 0.0, -1.0);
    let leftn = Vec3(-1.0, 0.0, 0.0);
    let rightn= Vec3(1.0, 0.0, 0.0);
    let backn = Vec3(0.0, 1.0, 0.0);

    // Floor
    let floor = make_plane(
        Vec3(right, front, bottom),
        Vec3(right, back, bottom),
        Vec3(left, back, bottom),
        Vec3(left, front, bottom),
        up,
    );
    // Ceiling
    let ceiling = make_plane(
        Vec3(left, front, top),
        Vec3(left, back, top),
        Vec3(right, back, top),
        Vec3(right, front, top),
        down,
    );
    // Left wall
    let left_wall = make_plane(
        Vec3(left, front, bottom),
        Vec3(left, back, bottom),
        Vec3(left, back, top),
        Vec3(left, front, top),
        leftn,
    );
    // Right wall
    let right_wall = make_plane(
        Vec3(right, front, top),
        Vec3(right, back, top),
        Vec3(right, back, bottom),
        Vec3(right, front, bottom),
        rightn,
    );
    // Back wall
    let back_wall = make_plane(
        Vec3(right, back, top),
        Vec3(left, back, top),
        Vec3(left, back, bottom),
        Vec3(right, back, bottom),
        backn,
    );

    // Add planes to scene builder with colored walls
    let white = scene_builder.add_constant_texture(Vec4(0.6, 0.6, 0.6, 1.0));
    let red   = scene_builder.add_constant_texture(Vec4(0.6, 0.2, 0.2, 1.0));
    let blue  = scene_builder.add_constant_texture(Vec4(0.2, 0.2, 0.6, 1.0));

    let white_diffuse = scene_builder.add_material(Material::Diffuse { albedo: white });
    let red_diffuse   = scene_builder.add_material(Material::Diffuse { albedo: red });
    let blue_diffuse  = scene_builder.add_material(Material::Diffuse { albedo: blue });

    scene_builder.add_shape_at_position(Shape::TriangleMesh(floor), white_diffuse, Vec3(0.0, 0.0, 0.0));
    scene_builder.add_shape_at_position(Shape::TriangleMesh(ceiling), white_diffuse, Vec3(0.0, 0.0, 0.0));
    scene_builder.add_shape_at_position(Shape::TriangleMesh(left_wall), red_diffuse, Vec3(0.0, 0.0, 0.0));
    scene_builder.add_shape_at_position(Shape::TriangleMesh(right_wall), blue_diffuse, Vec3(0.0, 0.0, 0.0));
    scene_builder.add_shape_at_position(Shape::TriangleMesh(back_wall), white_diffuse, Vec3(0.0, 0.0, 0.0));

    // Camera looking into the box from the front
    let camera = Camera::lookat_camera_perspective(
        Vec3(0.0, front + 3.4, 0.4),
        Vec3(0.0, 0.0, h / 2.0),
        Vec3(0.0, 0.0, 1.0),
        false,
        (37.8_f32).to_radians(),
        500,
        500,
    );
    scene_builder.add_camera(camera);

    // Add a point light near the top center
    scene_builder.add_point_light(
        Vec3(0.0, 0.0, top - 0.1),    // slightly below the ceiling, centered
        Vec3(1000.0, 1000.0, 1000.0), // bright white light
    );

    scene_builder
}

// single dielectric sphere (ior = 1.5) in cornell box
pub fn dielectric_scene() -> Scene {
    let mut cornell_box = cornell_box();

    let ior_texture = cornell_box.add_constant_texture(Vec4(1.5, 0.0, 0.0, 0.0));
    let dielectric_material =
        cornell_box.add_material(Material::SmoothDielectric { eta: ior_texture });
    cornell_box.add_shape_at_position(
        Shape::Sphere {
            center: Vec3::zero(),
            radius: 0.5,
        },
        dielectric_material,
        Vec3(0.0, 0.0, 0.75),
    );

    cornell_box.build()
}

// single "gold" sphere in cornell box
// ior at red wavelengths is 0.13 + 4.10i
// ior at green wavelengths is 0.43 + 2.46i
// ior at blue wavelengths is 1.38 + 1.91i
pub fn metal_scene() -> Scene {
    let mut cornell_box = cornell_box();

    let ior_texture = cornell_box.add_constant_texture(Vec4(0.13, 0.43, 1.38, 0.0));
    let kappa_texture = cornell_box.add_constant_texture(Vec4(4.10, 2.46, 1.91, 0.0));
    let metal_material = cornell_box.add_material(Material::SmoothConductor {
        eta: ior_texture,
        kappa: kappa_texture,
    });
    cornell_box.add_shape_at_position(
        Shape::Sphere {
            center: Vec3::zero(),
            radius: 0.5,
        },
        metal_material,
        Vec3(0.0, 0.0, 0.75),
    );

    cornell_box.build()
}

pub fn rough_metal_scene() -> Scene {
    let mut cornell_box = cornell_box();

    let ior_texture = cornell_box.add_constant_texture(Vec4(0.13, 0.43, 1.38, 0.0));
    let kappa_texture = cornell_box.add_constant_texture(Vec4(4.10, 2.46, 1.91, 0.0));
    let roughness_texture = cornell_box.add_constant_texture(Vec4(0.5, 0.5, 0.0, 0.0));
    let rough_conductor_material = cornell_box.add_material(Material::RoughConductor {
        eta: ior_texture,
        kappa: kappa_texture,
        remap_roughness: true,
        roughness: roughness_texture,
    });

    cornell_box.add_shape_at_position(
        Shape::Sphere {
            center: Vec3::zero(),
            radius: 0.5,
        },
        rough_conductor_material,
        Vec3(0.0, 0.0, 0.75),
    );

    cornell_box.build()
}

pub fn rough_dielectric_scene() -> Scene {
    let mut cornell_box = cornell_box();

    let ior_texture = cornell_box.add_constant_texture(Vec4(1.5, 0.0, 0.0, 0.0));
    let roughness_texture = cornell_box.add_constant_texture(Vec4(0.5, 0.5, 0.0, 0.0));
    let rough_dielectric_material = cornell_box.add_material(Material::RoughDielectric {
        eta: ior_texture,
        remap_roughness: true,
        roughness: roughness_texture,
    });
    cornell_box.add_shape_at_position(
        Shape::Sphere {
            center: Vec3::zero(),
            radius: 0.5,
        },
        rough_dielectric_material,
        Vec3(0.0, 0.0, 0.75),
    );

    cornell_box.build()
}

// Thin-lens camera test scene: sphere is out of focus
pub fn out_of_focus_sphere_scene() -> Scene {
    let mut scene_builder = SceneBuilder::new();

    let sphere = Shape::Sphere {
        center: Vec3::zero(),
        radius: 1.0,
    };

    let white = scene_builder.add_constant_texture(Vec4(1.0, 1.0, 1.0, 1.0));
    let white_diffuse = scene_builder.add_material(Material::Diffuse { albedo: white });

    scene_builder.add_shape_at_position(sphere, white_diffuse, Vec3(0.0, 0.0, -5.0));

    let sun = Light::DirectionLight {
        direction: Vec3(0.0, 0.0, -1.0),
        radiance: Vec3(1.0, 1.0, 1.0),
    };
    scene_builder.add_light(sun);

    // Thin-lens camera: focused at z=-3 (in front of sphere), sphere will be blurred
    let camera = Camera::lookat_camera_thin_lens_perspective(
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.0, 0.0, -5.0),
        Vec3(0.0, 1.0, 0.0),
        false,
        (45.0_f32).to_radians(),
        400,
        400,
        0.1,
        3.0,
    );
    scene_builder.add_camera(camera);

    scene_builder.build()
}

pub fn coated_diffuse_bunny_scene() -> Scene {
    let mut cornell_box = cornell_box();

    let bunny_mesh = Mesh::from_ply_reader(std::io::Cursor::new(assets::BUNNY_PLY), false)
        .expect("failed to load bunny.ply");
    let bunny = Shape::TriangleMesh(bunny_mesh);

    // CoatedDiffuse: red diffuse base with clear coat
    let diffuse_albedo = cornell_box.add_constant_texture(Vec4(0.8, 0.2, 0.2, 1.0));
    let dielectric_eta = cornell_box.add_constant_texture(Vec4(1.5, 0.0, 0.0, 0.0));
    let roughness = cornell_box.add_constant_texture(Vec4(0.1, 0.1, 0.0, 0.0));
    let thickness = cornell_box.add_constant_texture(Vec4(0.5, 0.0, 0.0, 0.0));
    let coat_albedo = cornell_box.add_constant_texture(Vec4(1.0, 1.0, 1.0, 1.0));

    let coated_diffuse = cornell_box.add_material(Material::CoatedDiffuse {
        diffuse_albedo,
        dielectric_eta,
        dielectric_remap_roughness: true,
        dielectric_roughness: Some(roughness),
        thickness,
        coat_albedo,
    });

    cornell_box.add_shape_at_position(bunny, coated_diffuse, Vec3(0.0, 0.0, 0.25));

    cornell_box.build()
}

pub fn environment_lighting_scene() -> Scene {
    let mut scene_builder = SceneBuilder::new();

    
    let environment_map = Image::load_from_bytes(assets::LAKE_PIER_ENVIRONMENT)
        .expect("unable to load environment map asset");

    let environment_map_id = scene_builder.add_image(environment_map);
    let environment_map_tex = Texture::ImageTexture { 
        image: environment_map_id, 
        sampler: TextureSampler {
            filter: crate::materials::FilterMode::Nearest,
            wrap: crate::materials::WrapMode::Repeat,
        }
    };

    let environment_map_tex_id = scene_builder.add_texture(environment_map_tex);

    let environment_light = EnvironmentLight {
        mapping: crate::lights::TextureMapping::Spherical,
        radiance: environment_map_tex_id,
    };
    
    scene_builder.add_environment_light(environment_light);

    let cube_mesh = make_cube(1.0);
    let cube = Shape::TriangleMesh(cube_mesh);

    let white = scene_builder.add_constant_texture(Vec4(1.0, 1.0, 1.0, 1.0));

    let white_diffuse = scene_builder.add_material(Material::Diffuse { albedo: white });

    scene_builder.add_shape_at_position(cube, white_diffuse, Vec3(0.0, 15.0, 0.0));

    // camera look along +y
    let camera = Camera::lookat_camera_perspective(
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
        false,
        (37.8_f32).to_radians(),
        500,
        500,
    );
    
    scene_builder.add_camera(camera);

    scene_builder.build()
}

fn debug_normals_settings() -> RaytracerSettings {
    RaytracerSettings {
        outputs: AOVFlags::NORMALS,
        ..Default::default()
    }
}

pub struct TestScene {
    pub name: &'static str,
    pub scene_func: fn() -> Scene,
    pub settings_func: fn() -> RaytracerSettings,
}

pub const fn all_test_scenes() -> &'static [TestScene] {
    &[
        TestScene {
            name: "sphere",
            scene_func: sphere_scene,
            settings_func: debug_normals_settings,
        },
        TestScene {
            name: "cube",
            scene_func: cube_scene,
            settings_func: debug_normals_settings,
        },
        TestScene {
            name: "cube_orthographic",
            scene_func: cube_orthographic_scene,
            settings_func: debug_normals_settings,
        },
        TestScene {
            name: "checkered_plane",
            scene_func: checkered_plane_scene,
            // deliberately only 1 spp to exhibit aliasing
            settings_func: || RaytracerSettings {
                samples_per_pixel: 1,
                ..Default::default()
            },
        },
        TestScene {
            name: "dielectric",
            scene_func: dielectric_scene,
            settings_func: RaytracerSettings::default,
        },
        TestScene {
            name: "metal",
            scene_func: metal_scene,
            settings_func: RaytracerSettings::default,
        },
        TestScene {
            name: "rough_metal",
            scene_func: rough_metal_scene,
            settings_func: RaytracerSettings::default,
        },
        TestScene {
            name: "rough_dielectric",
            scene_func: rough_dielectric_scene,
            settings_func: RaytracerSettings::default,
        },
        
        // this scene is meant to clearly demonstrate how stratified sampling
        // of lens position is able to significantly reduce variance
        TestScene {
            name: "out_of_focus_sphere",
            scene_func: out_of_focus_sphere_scene,
            settings_func: || RaytracerSettings {
                sampler: Sampler::Stratified {
                    jitter: true,
                    x_strata: 6,
                    y_strata: 6,
                },
                samples_per_pixel: 36,
                ..Default::default()
            },
        },

        TestScene {
            name: "environment_light",
            scene_func: environment_lighting_scene,
            settings_func: RaytracerSettings::default
        },
        TestScene {
            name: "coated_diffuse_bunny",
            scene_func: coated_diffuse_bunny_scene,
            settings_func: RaytracerSettings::default,
        },
    ]
}