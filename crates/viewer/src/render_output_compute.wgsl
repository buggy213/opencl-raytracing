// Compute shader `main_texture` converts raw buffer of radiance values into texture data (and sets alpha of 1)
// Compute shader `debug_texture` populates debug texture window

// 5×5 window by default
override DEBUG_WINDOW_SIZE: u32 = 200u;
override DEBUG_PIXEL_SIZE: u32 = 8u;

override SIZE_X: u32;
override SIZE_Y: u32;

struct PushConstants {
    gamma: f32,
    exposure: f32,

    mouse_pos: vec2<u32>,
}

var<push_constant> push_constants: PushConstants;

@group(0) @binding(0)
var<storage, read> radiance_buffer: array<f32>;  

@group(0) @binding(1)
var main_tex: texture_storage_2d<rgba32float, write>;

@group(0) @binding(2)
var debug_tex: texture_storage_2d<rgba32float, write>;

fn radiance_to_rgba(radiance: vec3<f32>) -> vec4<f32> {
    // apply exposure (assume "sensor response" is linear)
    let rgba = vec4f(radiance / push_constants.exposure, 1.0);
    // apply gamma correction
    return pow(rgba, vec4f(push_constants.gamma));
}

@compute
@workgroup_size(8, 8, 1)
fn main_texture(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    // bounds check
    if (id.x >= SIZE_X || id.y >= SIZE_Y) {
        return;
    }

    let pos = vec2u(id.x, id.y);
    let idx = (pos.y * SIZE_X + pos.x) * 3;

    let radiance = vec3f(radiance_buffer[idx], radiance_buffer[idx+1], radiance_buffer[idx+2]);
    let rgba = radiance_to_rgba(radiance);
    textureStore(main_tex, pos, rgba);
}

@compute
@workgroup_size(8, 8, 1)
fn debug_texture(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    if (id.x >= DEBUG_WINDOW_SIZE || id.y >= DEBUG_WINDOW_SIZE) {
        return;
    }

    let pos = vec2u(id.x, id.y);
    
    let offset = i32(DEBUG_WINDOW_SIZE / DEBUG_PIXEL_SIZE) / 2;
    let debug_pixel_pos_local = vec2i(
        i32(pos.x / DEBUG_PIXEL_SIZE) - offset, 
        i32(pos.y / DEBUG_PIXEL_SIZE) - offset
    );
    
    let debug_pixel_pos = debug_pixel_pos_local + vec2i(push_constants.mouse_pos);

    var radiance: vec3<f32>;
    if (any(debug_pixel_pos < vec2i(0)) || any(debug_pixel_pos >= vec2i(i32(SIZE_X), i32(SIZE_Y)))) {
        radiance = vec3(0.0);
    } 
    else {
        let idx = (u32(debug_pixel_pos.y) * SIZE_X + u32(debug_pixel_pos.x)) * 3;
        radiance = vec3f(radiance_buffer[idx], radiance_buffer[idx+1], radiance_buffer[idx+2]);
    }

    let debug_pixel = radiance_to_rgba(radiance);
    textureStore(debug_tex, pos, debug_pixel);
}