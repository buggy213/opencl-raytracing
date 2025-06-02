// Full screen textured quad shader (essentially blit)
struct VSOut {
    @builtin(position) pos: vec4f,
    @location(0) texcoord: vec2f
}

struct FSIn {
    @location(0) texcoord: vec2f
}

@vertex
fn vert_main(@builtin(vertex_index) in_vertex_index: u32) -> VSOut {
    // 0 1
    // 2 3
    let u = f32(i32(in_vertex_index) % 2);
    let v = f32(i32(in_vertex_index) / 2);

    let x = f32(i32(in_vertex_index % 2) * 2 - 1);
    let y = f32(i32(in_vertex_index / 2) * (-2) + 1);
    
    let output = VSOut(
        vec4f(x, y, 0.0, 1.0),
        vec2f(u, v)
    );
    return output;
}

@group(0) @binding(0)
var tex: texture_2d<f32>;

@group(0) @binding(1)
var tex_sampler: sampler;

@fragment
fn frag_main(input: FSIn) -> @location(0) vec4<f32> {
    // return vec4f(input.texcoord.x, input.texcoord.y, 0.0, 1.0);
    return textureSample(tex, tex_sampler, input.texcoord);
}