// distance from vertex to each edge of triangle
struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(linear) bary_dist: vec3<f32>
}

struct VertexData {
    position: vec3f,
}

@group(0) @binding(0)
var<storage, read> vertex_buffer: array<VertexData>;

@group(0) @binding(1)
var<storage, read> vertex_indices: array<u32>;

struct PushConstants {
    mvp: mat4x4f,
    screen_size: vec2f
}

var<push_constant> push_constants: PushConstants;

fn transformed_position(idx: u32) -> vec2f {
    let pos = vec4f(vertex_buffer[idx].position, 1.0);
    let transformed = push_constants.mvp * pos;
    return vec2f(transformed.xy / transformed.w) * push_constants.screen_size;
}

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VSOut {
    let base = (in_vertex_index / 3) * 3;
    let offset = in_vertex_index - base;

    let a_idx = vertex_indices[base+0];
    let b_idx = vertex_indices[base+1];
    let c_idx = vertex_indices[base+2];
    let a_pos = transformed_position(a_idx);
    let b_pos = transformed_position(b_idx);
    let c_pos = transformed_position(c_idx);

    var me: vec2f;
    var p: vec2f;
    var q: vec2f;

    if (offset == 0) {
        me = a_pos;
        p = b_pos;
        q = c_pos;
    }
    else if (offset == 1) {
        me = b_pos;
        p = c_pos;
        q = a_pos;
    }
    else {
        me = c_pos;
        p = a_pos;
        q = b_pos;
    }

    let line = p - q;
    let line_normal = normalize(vec2f(-line.y, line.x));
    let me_minus_p = me - p;
    let distance = abs(dot(me_minus_p, line_normal));

    var bary_dist: vec3f;
    if (offset == 0) {
        bary_dist = vec3f(distance, 0.0, 0.0);
    }
    else if (offset == 1) {
        bary_dist = vec3f(0.0, distance, 0.0);
    }
    else {
        bary_dist = vec3f(0.0, 0.0, distance);
    }

    let vertex_index = vertex_indices[in_vertex_index];
    let vertex_data = vertex_buffer[vertex_index];

    let vertex_position = vec4f(vertex_data.position, 1.0);
    let transformed = push_constants.mvp * vertex_position;
    
    let output = VSOut(
        transformed,
        bary_dist
    );

    return output;
}

@fragment
fn fs_main(vs_out: VSOut) -> @location(0) vec4<f32> {
    if (any(vs_out.bary_dist < vec3f(1.5))) {
        return vec4f(1.0, 1.0, 1.0, 1.0);
    }
    
    discard;
}