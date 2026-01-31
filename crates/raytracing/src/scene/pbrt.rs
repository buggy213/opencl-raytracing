//! PBRT Scene File Importer
//!
//! Parses a subset of PBRT v3/v4 scene files to produce a `Scene`. This is not a
//! feature complete system at all, since much of the functionality is still
//! missing from the renderer.
//!
//! # Supported Directives
//!
//! ## Scene-wide
//! - `Film` - Extract `xresolution` and `yresolution` for image dimensions
//! - `Camera` - `perspective` (with `fov`) and `orthographic`
//! - `WorldBegin`
//!
//! ## Transform
//! - Unnamed transforms are supported, but not named coordinate systems
//!
//! ## Attribute State
//! - `AttributeBegin` / `AttributeEnd` - Push/pop transform and material state
//!
//! ## Materials
//! - `Material` - `diffuse`, `matte`, `conductor`, `metal`, `dielectric`, `glass`
//! - `MakeNamedMaterial` / `NamedMaterial` - Named material definitions
//! - `Texture` - `constant` and `imagemap` textures
//!
//! ## Shapes
//! - `Shape "sphere"` - With `radius` parameter
//! - `Shape "trianglemesh"` - With `P` (positions), `indices`, optional `N` (normals), `uv`
//! - `Shape "plymesh"` - PLY file loading
//!
//! ## Lights
//! - `LightSource "point"` - With `I` (intensity) parameter
//! - `LightSource "distant"` - With `L` (radiance) and `from`/`to` parameters
//! - `AreaLightSource "diffuse"` - Applied to subsequent Shape
//!
//! ## Other
//! - `Include` - Include other PBRT files
//! - `#` comments are stripped
//!
//! # Unsupported Features
//! - Subsurface materials
//! - Spectrum/blackbody color specifications
//! - Procedural textures (other than constant)
//! - Environment/infinite lights
//! - Instancing (ObjectBegin/End)
//! - Media/volumes
//! - Most sampler, integrator, and film parameters (ignored)

use std::path::Path;

use tracing::warn;

use crate::{
    geometry::{Matrix4x4, Mesh, Shape, Transform, Vec2, Vec3, Vec3u, Vec4},
    lights::Light,
    materials::{FilterMode, Image, Material, Texture, TextureId, TextureSampler, WrapMode},
    scene::{
        primitive::MaterialIndex,
        Camera, Scene,
    },
};

use super::scene::SceneBuilder;

#[derive(Debug)]
pub enum ParseError {
    UnknownDirective(String),
    BadFloat(String),
    BadInteger(String),
    BadString,
    BadBool(String),
    BadParameter(String),
    UnexpectedEOF,
    UnexpectedToken(String),
    MissingParameter(String),
    FileError(String),
    UnsupportedFeature(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::UnknownDirective(d) => write!(f, "unknown directive: {}", d),
            ParseError::BadFloat(s) => write!(f, "bad float: {}", s),
            ParseError::BadInteger(s) => write!(f, "bad integer: {}", s),
            ParseError::BadString => write!(f, "bad string"),
            ParseError::BadBool(s) => write!(f, "bad bool: {}", s),
            ParseError::BadParameter(s) => write!(f, "bad parameter: {}", s),
            ParseError::UnexpectedEOF => write!(f, "unexpected end of file"),
            ParseError::UnexpectedToken(s) => write!(f, "unexpected token: {}", s),
            ParseError::MissingParameter(s) => write!(f, "missing parameter: {}", s),
            ParseError::FileError(s) => write!(f, "file error: {}", s),
            ParseError::UnsupportedFeature(s) => write!(f, "unsupported feature: {}", s),
        }
    }
}

impl std::error::Error for ParseError {}

#[derive(Debug, Clone)]
enum ParameterValue {
    Integer(i32),
    Integers(Vec<i32>),
    Float(f32),
    Floats(Vec<f32>),
    Point2([f32; 2]),
    Point2s(Vec<[f32; 2]>),
    Point3([f32; 3]),
    Point3s(Vec<[f32; 3]>),
    Vector3([f32; 3]),
    Vector3s(Vec<[f32; 3]>),
    Normal3([f32; 3]),
    Normal3s(Vec<[f32; 3]>),
    Rgb([f32; 3]),
    Bool(bool),
    String(String),
    Texture(String),
}

#[derive(Debug, Clone, Default)]
struct ParameterList {
    params: Vec<(String, ParameterValue)>,
}

impl ParameterList {
    fn get(&self, name: &str) -> Option<&ParameterValue> {
        self.params.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    }

    fn get_float(&self, name: &str) -> Option<f32> {
        match self.get(name)? {
            ParameterValue::Float(v) => Some(*v),
            ParameterValue::Floats(v) if !v.is_empty() => Some(v[0]),
            ParameterValue::Integer(v) => Some(*v as f32),
            ParameterValue::Integers(v) if !v.is_empty() => Some(v[0] as f32),
            _ => None,
        }
    }

    fn get_float_or(&self, name: &str, default: f32) -> f32 {
        self.get_float(name).unwrap_or(default)
    }

    fn get_integer(&self, name: &str) -> Option<i32> {
        match self.get(name)? {
            ParameterValue::Integer(v) => Some(*v),
            ParameterValue::Integers(v) if !v.is_empty() => Some(v[0]),
            _ => None,
        }
    }

    fn get_integer_or(&self, name: &str, default: i32) -> i32 {
        self.get_integer(name).unwrap_or(default)
    }

    fn get_integers(&self, name: &str) -> Option<&[i32]> {
        match self.get(name)? {
            ParameterValue::Integers(v) => Some(v),
            _ => None,
        }
    }

    fn get_point3(&self, name: &str) -> Option<[f32; 3]> {
        match self.get(name)? {
            ParameterValue::Point3(v) => Some(*v),
            ParameterValue::Point3s(v) if !v.is_empty() => Some(v[0]),
            _ => None,
        }
    }

    fn get_point3s(&self, name: &str) -> Option<&[[f32; 3]]> {
        match self.get(name)? {
            ParameterValue::Point3s(v) => Some(v),
            ParameterValue::Point3(v) => Some(std::slice::from_ref(v)),
            _ => None,
        }
    }

    fn get_normal3s(&self, name: &str) -> Option<&[[f32; 3]]> {
        match self.get(name)? {
            ParameterValue::Normal3s(v) => Some(v),
            ParameterValue::Normal3(v) => Some(std::slice::from_ref(v)),
            _ => None,
        }
    }

    fn get_point2s(&self, name: &str) -> Option<&[[f32; 2]]> {
        match self.get(name)? {
            ParameterValue::Point2s(v) => Some(v),
            ParameterValue::Point2(v) => Some(std::slice::from_ref(v)),
            _ => None,
        }
    }

    fn get_floats(&self, name: &str) -> Option<&[f32]> {
        match self.get(name)? {
            ParameterValue::Floats(v) => Some(v),
            ParameterValue::Float(v) => Some(std::slice::from_ref(v)),
            _ => None,
        }
    }

    fn get_rgb(&self, name: &str) -> Option<Vec3> {
        match self.get(name)? {
            ParameterValue::Rgb(v) => Some(Vec3(v[0], v[1], v[2])),
            ParameterValue::Floats(v) if v.len() >= 3 => Some(Vec3(v[0], v[1], v[2])),
            _ => None,
        }
    }

    fn get_rgb_or(&self, name: &str, default: Vec3) -> Vec3 {
        self.get_rgb(name).unwrap_or(default)
    }

    fn get_string(&self, name: &str) -> Option<&str> {
        match self.get(name)? {
            ParameterValue::String(v) => Some(v),
            _ => None,
        }
    }

    fn get_texture(&self, name: &str) -> Option<&str> {
        match self.get(name)? {
            ParameterValue::Texture(v) => Some(v),
            _ => None,
        }
    }
    
    fn get_bool(&self, name: &str) -> Option<bool> {
        match self.get(name)? {
            ParameterValue::Bool(b) => Some(*b),
            _ => None
        }
    }
}

#[derive(Clone)]
struct AttributeState {
    transform: Transform,
    material: Option<MaterialIndex>,
    area_light_radiance: Option<Vec3>,
}

struct ParserState {
    current_transform: Transform,
    attribute_stack: Vec<AttributeState>,
    film_width: usize,
    film_height: usize,
    named_materials: Vec<(String, MaterialIndex)>,
    named_textures: Vec<(String, TextureId)>,
    current_material: Option<MaterialIndex>,
    area_light_radiance: Option<Vec3>,
    in_world_block: bool,
    has_camera: bool,
    has_lights: bool,
}

impl ParserState {
    fn new() -> Self {
        ParserState {
            current_transform: Transform::identity(),
            attribute_stack: Vec::new(),
            film_width: 640,
            film_height: 480,
            named_materials: Vec::new(),
            named_textures: Vec::new(),
            current_material: None,
            area_light_radiance: None,
            in_world_block: false,
            has_camera: false,
            has_lights: false,
        }
    }

    fn push_attributes(&mut self) {
        self.attribute_stack.push(AttributeState {
            transform: self.current_transform.clone(),
            material: self.current_material,
            area_light_radiance: self.area_light_radiance,
        });
    }

    fn pop_attributes(&mut self) {
        if let Some(state) = self.attribute_stack.pop() {
            self.current_transform = state.transform;
            self.current_material = state.material;
            self.area_light_radiance = state.area_light_radiance;
        } else {
            warn!("AttributeEnd without matching AttributeBegin");
        }
    }

    fn get_named_material(&self, name: &str) -> Option<MaterialIndex> {
        self.named_materials.iter().find(|(n, _)| n == name).map(|(_, v)| *v)
    }

    fn get_named_texture(&self, name: &str) -> Option<TextureId> {
        self.named_textures.iter().find(|(n, _)| n == name).map(|(_, v)| *v)
    }
}

struct TokenStream<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> TokenStream<'a> {
    fn new(input: &'a str) -> Self {
        TokenStream { input, pos: 0 }
    }

    fn skip_whitespace_and_comments(&mut self) {
        let bytes = self.input.as_bytes();
        while self.pos < bytes.len() {
            let c = bytes[self.pos];
            if c == b'#' {
                while self.pos < bytes.len() && bytes[self.pos] != b'\n' {
                    self.pos += 1;
                }
            } else if c.is_ascii_whitespace() {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn peek(&mut self) -> Option<&'a str> {
        self.skip_whitespace_and_comments();
        if self.pos >= self.input.len() {
            return None;
        }

        let bytes = self.input.as_bytes();
        let start = self.pos;

        if bytes[start] == b'"' {
            let mut end = start + 1;
            while end < bytes.len() && bytes[end] != b'"' {
                end += 1;
            }
            if end < bytes.len() {
                end += 1; // include closing quote
            }
            Some(&self.input[start..end])
        } else if bytes[start] == b'[' {
            Some(&self.input[start..start + 1])
        } else if bytes[start] == b']' {
            Some(&self.input[start..start + 1])
        } else {
            let mut end = start;
            while end < bytes.len() {
                let c = bytes[end];
                if c.is_ascii_whitespace() || c == b'[' || c == b']' || c == b'"' || c == b'#' {
                    break;
                }
                end += 1;
            }
            Some(&self.input[start..end])
        }
    }

    fn next(&mut self) -> Option<&'a str> {
        let tok = self.peek()?;
        self.pos += tok.len();
        Some(tok)
    }

    fn expect(&mut self, expected: &str) -> Result<(), ParseError> {
        match self.next() {
            Some(tok) if tok == expected => Ok(()),
            Some(tok) => Err(ParseError::UnexpectedToken(format!(
                "expected '{}', got '{}'",
                expected, tok
            ))),
            None => Err(ParseError::UnexpectedEOF),
        }
    }
}

fn parse_quoted_string(tok: &str) -> Result<&str, ParseError> {
    if tok.starts_with('"') && tok.ends_with('"') && tok.len() >= 2 {
        Ok(&tok[1..tok.len() - 1])
    } else {
        Err(ParseError::BadString)
    }
}

fn parse_bool(tok: &str) -> Result<bool, ParseError> {
    tok.parse::<bool>().map_err(|_| ParseError::BadBool(tok.to_string()))
}

fn parse_float(tok: &str) -> Result<f32, ParseError> {
    tok.parse::<f32>()
        .map_err(|_| ParseError::BadFloat(tok.to_string()))
}

fn parse_integer(tok: &str) -> Result<i32, ParseError> {
    tok.parse::<i32>()
        .map_err(|_| ParseError::BadInteger(tok.to_string()))
}

fn parse_type_string<'a>(toks: &mut TokenStream<'a>) -> Result<&'a str, ParseError> {
    let tok = toks.next().ok_or(ParseError::UnexpectedEOF)?;
    parse_quoted_string(tok)
}

fn parse_single_float(toks: &mut TokenStream) -> Result<f32, ParseError> {
    let tok = toks.next().ok_or(ParseError::UnexpectedEOF)?;
    parse_float(tok)
}

fn parse_parameter_list(toks: &mut TokenStream) -> Result<ParameterList, ParseError> {
    let mut params = ParameterList::default();

    while let Some(tok) = toks.peek() {
        if !tok.starts_with('"') {
            break;
        }

        let type_name = parse_quoted_string(toks.next().unwrap())?;
        let parts: Vec<&str> = type_name.split_whitespace().collect();
        if parts.len() != 2 {
            return Err(ParseError::BadParameter(type_name.to_string()));
        }

        let param_type = parts[0];
        let param_name = parts[1].to_string();

        let value = parse_parameter_value(toks, param_type)?;
        params.params.push((param_name, value));
    }

    Ok(params)
}

fn parse_parameter_value(toks: &mut TokenStream, param_type: &str) -> Result<ParameterValue, ParseError> {
    let has_brackets = toks.peek() == Some("[");
    if has_brackets {
        toks.next();
    }

    let value = match param_type {
        "integer" => {
            let mut values = Vec::new();
            while let Some(tok) = toks.peek() {
                if tok == "]" || tok.starts_with('"') {
                    break;
                }
                values.push(parse_integer(toks.next().unwrap())?);
                if !has_brackets {
                    break;
                }
            }
            if values.len() == 1 {
                ParameterValue::Integer(values[0])
            } else {
                ParameterValue::Integers(values)
            }
        }
        "float" => {
            let mut values = Vec::new();
            while let Some(tok) = toks.peek() {
                if tok == "]" || tok.starts_with('"') {
                    break;
                }
                values.push(parse_float(toks.next().unwrap())?);
                if !has_brackets {
                    break;
                }
            }
            if values.len() == 1 {
                ParameterValue::Float(values[0])
            } else {
                ParameterValue::Floats(values)
            }
        }
        "point2" => {
            let mut values = Vec::new();
            while let Some(tok) = toks.peek() {
                if tok == "]" || tok.starts_with('"') {
                    break;
                }
                let x = parse_float(toks.next().unwrap())?;
                let y = parse_float(toks.next().ok_or(ParseError::UnexpectedEOF)?)?;
                values.push([x, y]);
                if !has_brackets {
                    break;
                }
            }
            if values.len() == 1 {
                ParameterValue::Point2(values[0])
            } else {
                ParameterValue::Point2s(values)
            }
        }
        "point3" | "point" => {
            let mut values = Vec::new();
            while let Some(tok) = toks.peek() {
                if tok == "]" || tok.starts_with('"') {
                    break;
                }
                let x = parse_float(toks.next().unwrap())?;
                let y = parse_float(toks.next().ok_or(ParseError::UnexpectedEOF)?)?;
                let z = parse_float(toks.next().ok_or(ParseError::UnexpectedEOF)?)?;
                values.push([x, y, z]);
                if !has_brackets {
                    break;
                }
            }
            if values.len() == 1 {
                ParameterValue::Point3(values[0])
            } else {
                ParameterValue::Point3s(values)
            }
        }
        "vector3" | "vector" => {
            let mut values = Vec::new();
            while let Some(tok) = toks.peek() {
                if tok == "]" || tok.starts_with('"') {
                    break;
                }
                let x = parse_float(toks.next().unwrap())?;
                let y = parse_float(toks.next().ok_or(ParseError::UnexpectedEOF)?)?;
                let z = parse_float(toks.next().ok_or(ParseError::UnexpectedEOF)?)?;
                values.push([x, y, z]);
                if !has_brackets {
                    break;
                }
            }
            if values.len() == 1 {
                ParameterValue::Vector3(values[0])
            } else {
                ParameterValue::Vector3s(values)
            }
        }
        "normal3" | "normal" => {
            let mut values = Vec::new();
            while let Some(tok) = toks.peek() {
                if tok == "]" || tok.starts_with('"') {
                    break;
                }
                let x = parse_float(toks.next().unwrap())?;
                let y = parse_float(toks.next().ok_or(ParseError::UnexpectedEOF)?)?;
                let z = parse_float(toks.next().ok_or(ParseError::UnexpectedEOF)?)?;
                values.push([x, y, z]);
                if !has_brackets {
                    break;
                }
            }
            if values.len() == 1 {
                ParameterValue::Normal3(values[0])
            } else {
                ParameterValue::Normal3s(values)
            }
        }
        "rgb" | "color" => {
            let r = parse_float(toks.next().ok_or(ParseError::UnexpectedEOF)?)?;
            let g = parse_float(toks.next().ok_or(ParseError::UnexpectedEOF)?)?;
            let b = parse_float(toks.next().ok_or(ParseError::UnexpectedEOF)?)?;
            ParameterValue::Rgb([r, g, b])
        }
        "bool" => {
            let tok = toks.next().ok_or(ParseError::UnexpectedEOF)?;
            let tok_clean = tok.trim_matches('"');
            let v = match tok_clean {
                "true" => true,
                "false" => false,
                _ => return Err(ParseError::BadBool(tok.to_string())),
            };
            ParameterValue::Bool(v)
        }
        "string" => {
            let tok = toks.next().ok_or(ParseError::UnexpectedEOF)?;
            let s = parse_quoted_string(tok)?;
            ParameterValue::String(s.to_string())
        }
        "texture" => {
            let tok = toks.next().ok_or(ParseError::UnexpectedEOF)?;
            let s = parse_quoted_string(tok)?;
            ParameterValue::Texture(s.to_string())
        }
        "spectrum" => {
            warn!("spectrum parameters not fully supported, treating as RGB");
            let r = parse_float(toks.next().ok_or(ParseError::UnexpectedEOF)?)?;
            let g = parse_float(toks.next().ok_or(ParseError::UnexpectedEOF)?)?;
            let b = parse_float(toks.next().ok_or(ParseError::UnexpectedEOF)?)?;
            ParameterValue::Rgb([r, g, b])
        }
        _ => {
            warn!("unknown parameter type '{}', skipping", param_type);
            while let Some(tok) = toks.peek() {
                if tok == "]" || tok.starts_with('"') {
                    break;
                }
                toks.next();
                if !has_brackets {
                    break;
                }
            }
            ParameterValue::Float(0.0)
        }
    };

    if has_brackets {
        toks.expect("]")?;
    }

    Ok(value)
}

fn parse_identity_directive(state: &mut ParserState) {
    state.current_transform = Transform::identity();
}

fn parse_lookat_directive(toks: &mut TokenStream, state: &mut ParserState) -> Result<(), ParseError> {
    let eye_x = parse_single_float(toks)?;
    let eye_y = parse_single_float(toks)?;
    let eye_z = parse_single_float(toks)?;

    let look_x = parse_single_float(toks)?;
    let look_y = parse_single_float(toks)?;
    let look_z = parse_single_float(toks)?;

    let up_x = parse_single_float(toks)?;
    let up_y = parse_single_float(toks)?;
    let up_z = parse_single_float(toks)?;

    let eye = Vec3(eye_x, eye_y, eye_z);
    let look = Vec3(look_x, look_y, look_z);
    let up = Vec3(up_x, up_y, up_z);

    // pbrt uses a left-handed coordinate system, so we apply a handedness-swap within camera2world transform
    // to make outputs look the same
    let look_at_transform = Transform::look_at(eye, look, up, true);
    state.current_transform = state.current_transform.compose(look_at_transform.invert());

    Ok(())
}

fn parse_translate_directive(toks: &mut TokenStream, state: &mut ParserState) -> Result<(), ParseError> {
    let x = parse_single_float(toks)?;
    let y = parse_single_float(toks)?;
    let z = parse_single_float(toks)?;

    let translate_transform = Transform::translate(Vec3(x, y, z));
    state.current_transform = state.current_transform.compose(translate_transform);

    Ok(())
}

fn parse_scale_directive(toks: &mut TokenStream, state: &mut ParserState) -> Result<(), ParseError> {
    let x = parse_single_float(toks)?;
    let y = parse_single_float(toks)?;
    let z = parse_single_float(toks)?;

    let scale_transform = Transform::scale(Vec3(x, y, z));
    state.current_transform = state.current_transform.compose(scale_transform);

    Ok(())
}

fn parse_rotate_directive(toks: &mut TokenStream, state: &mut ParserState) -> Result<(), ParseError> {
    let angle = parse_single_float(toks)?;
    let x = parse_single_float(toks)?;
    let y = parse_single_float(toks)?;
    let z = parse_single_float(toks)?;

    let rotate_transform = Transform::rotate(angle.to_radians(), Vec3(x, y, z));
    state.current_transform = state.current_transform.compose(rotate_transform);

    Ok(())
}

fn parse_transform_directive(toks: &mut TokenStream, state: &mut ParserState) -> Result<(), ParseError> {
    toks.expect("[")?;

    let mut m = [0.0f32; 16];
    for v in &mut m {
        *v = parse_single_float(toks)?;
    }

    toks.expect("]")?;

    // PBRT uses column-major matrix format
    let mat = Matrix4x4::from(m).transposed();
    let transform = Transform::from(mat);
    state.current_transform = transform;

    Ok(())
}

fn parse_concat_transform_directive(toks: &mut TokenStream, state: &mut ParserState) -> Result<(), ParseError> {
    toks.expect("[")?;

    let mut m = [0.0f32; 16];
    for v in &mut m {
        *v = parse_single_float(toks)?;
    }

    toks.expect("]")?;

    // PBRT uses column-major matrix format
    let mat = Matrix4x4::from(m).transposed();
    let transform = Transform::from(mat);
    state.current_transform = state.current_transform.compose(transform);

    Ok(())
}

fn parse_film_directive(toks: &mut TokenStream, state: &mut ParserState) -> Result<(), ParseError> {
    let _film_type = parse_type_string(toks)?;
    let params = parse_parameter_list(toks)?;

    state.film_width = params.get_integer_or("xresolution", 640) as usize;
    state.film_height = params.get_integer_or("yresolution", 480) as usize;

    Ok(())
}

fn parse_camera_directive(
    toks: &mut TokenStream,
    state: &mut ParserState,
    builder: &mut SceneBuilder,
) -> Result<(), ParseError> {
    let camera_type = parse_type_string(toks)?;
    let params = parse_parameter_list(toks)?;

    let camera = match camera_type {
        "perspective" => {
            let fov = params.get_float_or("fov", 90.0);
            let fov_rad = fov.to_radians();

            let camera_to_world = state.current_transform.invert();
            let camera_position = camera_to_world.apply_point(Vec3(0.0, 0.0, 0.0));
            let target = camera_to_world.apply_point(Vec3(0.0, 0.0, 1.0));
            let up = camera_to_world.apply_vector(Vec3(0.0, 1.0, 0.0));
            
            Camera::lookat_camera_perspective(
                camera_position,
                target,
                up,
                false,
                fov_rad,
                state.film_width,
                state.film_height,
            )
        }
        "orthographic" => {
            let camera_to_world = state.current_transform.invert();
            let camera_position = camera_to_world.apply_point(Vec3(0.0, 0.0, 0.0));
            let target = camera_to_world.apply_point(Vec3(0.0, 0.0, 1.0));
            let up = camera_to_world.apply_vector(Vec3(0.0, 1.0, 0.0));

            Camera::lookat_camera_orthographic(
                camera_position,
                target,
                up,
                false,
                state.film_width,
                state.film_height,
                1.0 / state.film_width.min(state.film_height) as f32,
            )
        }
        other => {
            warn!("unsupported camera type '{}', defaulting to perspective", other);
            let camera_to_world = state.current_transform.invert();
            let camera_position = camera_to_world.apply_point(Vec3(0.0, 0.0, 0.0));
            let target = camera_to_world.apply_point(Vec3(0.0, 0.0, 1.0));
            let up = camera_to_world.apply_vector(Vec3(0.0, 1.0, 0.0));

            Camera::lookat_camera_perspective(
                camera_position,
                target,
                up,
                false,
                (90.0_f32).to_radians(),
                state.film_width,
                state.film_height,
            )
        }
    };

    builder.add_camera(camera);
    state.has_camera = true;
    Ok(())
}

fn resolve_texture(
    state: &ParserState,
    builder: &mut SceneBuilder,
    params: &ParameterList,
    param_name: &str,
    default: Vec3,
) -> TextureId {
    if let Some(tex_name) = params.get_texture(param_name) {
        if let Some(tex_id) = state.get_named_texture(tex_name) {
            return tex_id;
        }
    }

    let color = params.get_rgb_or(param_name, default);
    builder.add_constant_texture(Vec4(color.0, color.1, color.2, 1.0))
}

fn resolve_float_texture(
    state: &ParserState,
    builder: &mut SceneBuilder,
    params: &ParameterList,
    param_name: &str,
    default: f32,
) -> TextureId {
    if let Some(tex_name) = params.get_texture(param_name) {
        if let Some(tex_id) = state.get_named_texture(tex_name) {
            return tex_id;
        }
    }

    let value = params.get_float_or(param_name, default);
    builder.add_constant_texture(Vec4(value, value, value, 1.0))
}

fn parse_material_directive(
    toks: &mut TokenStream,
    state: &mut ParserState,
    builder: &mut SceneBuilder,
) -> Result<(), ParseError> {
    let material_type = parse_type_string(toks)?;
    let params = parse_parameter_list(toks)?;

    let material = create_material(material_type, &params, state, builder)?;
    state.current_material = Some(builder.add_material(material));

    Ok(())
}

fn extract_roughness(params: &ParameterList, builder: &mut SceneBuilder, state: &ParserState) -> Option<TextureId> {
    let roughness = params.get("roughness").is_some();
    let anisotropic_roughness = {
        let has_u_roughness = params.get("uroughness").is_some();
        let has_v_roughness = params.get("vroughness").is_some();

        if has_u_roughness != has_v_roughness {
            warn!("bad anisotropic roughness description; both u and v components are required. falling back to smooth");
            return None;
        }
        has_u_roughness && has_v_roughness
    };

    if roughness && anisotropic_roughness {
        warn!("bad roughness description; both `roughness` and `uroughness/vroughness` descriptions provided. falling back to smooth");
        return None;
    }

    if roughness {
        Some(resolve_float_texture(state, builder, params, "roughness", 0.0))
    }
    else if anisotropic_roughness {
        let uroughness = params.get_float("uroughness");
        let vroughness = params.get_float("vroughness");
        
        match (uroughness, vroughness) {
            (Some(alpha_x), Some(alpha_y)) => {
                Some(builder.add_constant_texture(Vec4(alpha_x, alpha_y, 0.0, 0.0)))
            }
            _ => {
                todo!("texture values for uroughness / vroughness")
            }
        }
    }
    else {
        None
    }
}

fn create_material(
    material_type: &str,
    params: &ParameterList,
    state: &ParserState,
    builder: &mut SceneBuilder,
) -> Result<Material, ParseError> {
    let material = match material_type {
        "diffuse" => {
            let albedo = resolve_texture(state, builder, params, "reflectance", Vec3(0.5, 0.5, 0.5));
            Material::Diffuse { albedo }
        }

        "conductor" => {
            let eta = resolve_texture(state, builder, params, "eta", Vec3(0.2, 0.2, 0.2));
            let k = resolve_texture(state, builder, params, "k", Vec3(3.0, 3.0, 3.0));

            let roughness_tex = extract_roughness(params, builder, state);

            if let Some(roughness_tex) = roughness_tex {
                let remap_roughness = params.get_bool("remaproughness").unwrap_or(true);
                Material::RoughConductor {
                    eta,
                    kappa: k,
                    remap_roughness,
                    roughness: roughness_tex,
                }
            } else {
                Material::SmoothConductor { eta, kappa: k }
            }
        }
        "dielectric" => {
            let ior = params.get_float_or("eta", 1.5);
            let eta = builder.add_constant_texture(Vec4(ior, 0.0, 0.0, 0.0));

            let roughness_tex = extract_roughness(params, builder, state);

            if let Some(roughness_tex) = roughness_tex {
                let remap_roughness = params.get_bool("remaproughness").unwrap_or(true);
                Material::RoughDielectric {
                    eta,
                    remap_roughness,
                    roughness: roughness_tex,
                }
            } else {
                Material::SmoothDielectric{ eta }
            }
        }
        "coateddiffuse" => {
            let diffuse_albedo = resolve_texture(state, builder, params, "reflectance", Vec3(0.5, 0.5, 0.5));
            let coat_eta = params.get_float_or("eta", 1.5);
            let dielectric_eta = builder.add_constant_texture(Vec4(coat_eta, 0.0, 0.0, 0.0));

            let dielectric_roughness = extract_roughness(params, builder, state);
            let dielectric_remap_roughness = params.get_bool("remaproughness").unwrap_or(true);

            let thickness_val = params.get_float_or("thickness", 0.01);
            let thickness = builder.add_constant_texture(Vec4(thickness_val, 0.0, 0.0, 0.0));

            let coat_albedo_color = params.get_rgb_or("albedo", Vec3(1.0, 1.0, 1.0));
            let coat_albedo = builder.add_constant_texture(Vec4(coat_albedo_color.0, coat_albedo_color.1, coat_albedo_color.2, 1.0));

            Material::CoatedDiffuse {
                diffuse_albedo,
                dielectric_eta,
                dielectric_remap_roughness,
                dielectric_roughness,
                thickness,
                coat_albedo,
            }
        }

        other => {
            warn!("unsupported material type '{}', defaulting to diffuse gray", other);
            let albedo = builder.add_constant_texture(Vec4(0.5, 0.5, 0.5, 1.0));
            Material::Diffuse { albedo }
        }
    };

    Ok(material)
}

fn parse_make_named_material_directive(
    toks: &mut TokenStream,
    state: &mut ParserState,
    builder: &mut SceneBuilder,
) -> Result<(), ParseError> {
    let name = parse_type_string(toks)?;
    let params = parse_parameter_list(toks)?;

    let material_type = params.get_string("type").unwrap_or("diffuse");
    let material = create_material(material_type, &params, state, builder)?;
    let material_id = builder.add_material(material);
    state.named_materials.push((name.to_string(), material_id));

    Ok(())
}

fn parse_named_material_directive(
    toks: &mut TokenStream,
    state: &mut ParserState,
) -> Result<(), ParseError> {
    let name = parse_type_string(toks)?;

    if let Some(material_id) = state.get_named_material(name) {
        state.current_material = Some(material_id);
    } else {
        warn!("unknown named material '{}', using current material", name);
    }

    Ok(())
}

fn parse_texture_directive(
    toks: &mut TokenStream,
    state: &mut ParserState,
    builder: &mut SceneBuilder,
    base_path: &Path,
) -> Result<(), ParseError> {
    let name = parse_type_string(toks)?;
    let _color_type = parse_type_string(toks)?; // "spectrum", "float", etc.
    let tex_type = parse_type_string(toks)?;
    let params = parse_parameter_list(toks)?;

    let texture = match tex_type {
        "constant" => {
            let value = params.get_rgb_or("value", Vec3(1.0, 1.0, 1.0));
            Texture::ConstantTexture {
                value: Vec4(value.0, value.1, value.2, 1.0),
            }
        }
        "imagemap" => {
            if let Some(filename) = params.get_string("filename") {
                let image_path = base_path.join(filename);
                match Image::load_from_path(&image_path) {
                    Ok(image) => {
                        let image_id = builder.add_image(image);
                        Texture::ImageTexture {
                            image: image_id,
                            sampler: TextureSampler {
                                filter: FilterMode::Bilinear,
                                wrap: WrapMode::Repeat,
                            },
                        }
                    }
                    Err(e) => {
                        warn!("failed to load texture '{}': {}", filename, e);
                        Texture::ConstantTexture {
                            value: Vec4(1.0, 0.0, 1.0, 1.0), // magenta error color
                        }
                    }
                }
            } else {
                warn!("imagemap texture missing filename");
                Texture::ConstantTexture {
                    value: Vec4(1.0, 0.0, 1.0, 1.0),
                }
            }
        }
        "scale" => {
            let scale_val = params.get_float_or("scale", 1.0);
            Texture::ConstantTexture {
                value: Vec4(scale_val, scale_val, scale_val, 1.0),
            }
        }
        "checkerboard" => {
            let tex1 = params.get_rgb_or("tex1", Vec3(0.0, 0.0, 0.0));
            let tex2 = params.get_rgb_or("tex2", Vec3(1.0, 1.0, 1.0));
            Texture::CheckerTexture {
                color1: Vec4(tex1.0, tex1.1, tex1.2, 1.0),
                color2: Vec4(tex2.0, tex2.1, tex2.2, 1.0),
            }
        }
        other => {
            warn!("unsupported texture type '{}', using constant white", other);
            Texture::ConstantTexture {
                value: Vec4(1.0, 1.0, 1.0, 1.0),
            }
        }
    };

    let tex_id = builder.add_texture(texture);
    state.named_textures.push((name.to_string(), tex_id));

    Ok(())
}

fn parse_shape_directive(
    toks: &mut TokenStream,
    state: &mut ParserState,
    builder: &mut SceneBuilder,
    base_path: &Path,
) -> Result<(), ParseError> {
    let shape_type = parse_type_string(toks)?;
    let params = parse_parameter_list(toks)?;

    let material_id = state.current_material.unwrap_or_else(|| {
        let albedo = builder.add_constant_texture(Vec4(0.5, 0.5, 0.5, 1.0));
        builder.add_material(Material::Diffuse { albedo })
    });

    let shape = match shape_type {
        "sphere" => {
            let radius = params.get_float_or("radius", 1.0);
            Shape::Sphere {
                center: Vec3(0.0, 0.0, 0.0),
                radius,
            }
        }
        "trianglemesh" => {
            let positions = params
                .get_point3s("P")
                .ok_or_else(|| ParseError::MissingParameter("P".to_string()))?;

            let indices = params.get_integers("indices");

            let vertices: Vec<Vec3> = positions.iter().map(|p| Vec3(p[0], p[1], p[2])).collect();

            let tris: Vec<Vec3u> = if let Some(indices) = indices {
                indices
                    .chunks(3)
                    .map(|chunk| Vec3u(chunk[0] as u32, chunk[1] as u32, chunk[2] as u32))
                    .collect()
            } else {
                (0..vertices.len() / 3)
                    .map(|i| Vec3u((i * 3) as u32, (i * 3 + 1) as u32, (i * 3 + 2) as u32))
                    .collect()
            };

            let normals: Vec<Vec3> = if let Some(n) = params.get_normal3s("N") {
                n.iter().map(|n| Vec3(n[0], n[1], n[2])).collect()
            } else {
                Vec::new()
            };

            let uvs: Vec<Vec2> = if let Some(uv) = params.get_point2s("uv") {
                uv.iter().map(|uv| Vec2(uv[0], uv[1])).collect()
            } else if let Some(uv) = params.get_floats("uv") {
                uv.chunks(2)
                    .map(|chunk| Vec2(chunk[0], chunk.get(1).copied().unwrap_or(0.0)))
                    .collect()
            } else {
                Vec::new()
            };

            let mesh = Mesh {
                vertices,
                tris,
                normals,
                uvs,
            };

            Shape::TriangleMesh(mesh)
        }
        "plymesh" => {
            let filename = params.get_string("filename")
                .ok_or_else(|| ParseError::MissingParameter("filename".to_string()))?;
            let ply_path = base_path.join(filename);

            // pbrt meshes appear to have CW winding order
            match Mesh::from_ply(&ply_path, true) {
                Ok(mesh) => Shape::TriangleMesh(mesh),
                Err(e) => {
                    warn!("failed to load PLY file '{}': {}", filename, e);
                    return Ok(());
                }
            }
        }
        "disk" => {
            warn!("disk shape not supported, creating placeholder sphere");
            let radius = params.get_float_or("radius", 1.0);
            Shape::Sphere {
                center: Vec3(0.0, 0.0, 0.0),
                radius,
            }
        }
        other => {
            warn!("unsupported shape type '{}', skipping", other);
            return Ok(());
        }
    };

    if state.area_light_radiance.is_some() {
        state.has_lights = true;
    }
    builder.add_shape_with_transform(shape, material_id, &state.current_transform, state.area_light_radiance);
    state.area_light_radiance = None;

    Ok(())
}

fn parse_light_source_directive(
    toks: &mut TokenStream,
    state: &mut ParserState,
    builder: &mut SceneBuilder,
) -> Result<(), ParseError> {
    let light_type = parse_type_string(toks)?;
    let params = parse_parameter_list(toks)?;

    match light_type {
        "point" => {
            let intensity = params.get_rgb_or("I", Vec3(1.0, 1.0, 1.0));
            let scale = params.get_float_or("scale", 1.0);
            let from = params
                .get_point3("from")
                .map(|p| Vec3(p[0], p[1], p[2]))
                .unwrap_or(Vec3(0.0, 0.0, 0.0));

            let position = state.current_transform.apply_point(from);
            let scaled_intensity = intensity * scale;

            builder.add_light(Light::PointLight {
                position,
                intensity: scaled_intensity,
            });
            state.has_lights = true;
        }
        "distant" => {
            let radiance = params.get_rgb_or("L", Vec3(1.0, 1.0, 1.0));
            let scale = params.get_float_or("scale", 1.0);

            let from = params
                .get_point3("from")
                .map(|p| Vec3(p[0], p[1], p[2]))
                .unwrap_or(Vec3(0.0, 0.0, 1.0));
            let to = params
                .get_point3("to")
                .map(|p| Vec3(p[0], p[1], p[2]))
                .unwrap_or(Vec3(0.0, 0.0, 0.0));

            let direction = (to - from).unit();
            let world_direction = state.current_transform.apply_vector(direction);
            let scaled_radiance = radiance * scale;

            builder.add_light(Light::DirectionLight {
                direction: world_direction,
                radiance: scaled_radiance,
            });
            state.has_lights = true;
        }
        "infinite" | "environment" => {
            warn!("infinite/environment lights not supported");
        }
        "spot" => {
            let intensity = params.get_rgb_or("I", Vec3(1.0, 1.0, 1.0));
            let from = params
                .get_point3("from")
                .map(|p| Vec3(p[0], p[1], p[2]))
                .unwrap_or(Vec3(0.0, 0.0, 0.0));

            let position = state.current_transform.apply_point(from);
            warn!("spot light converted to point light");

            builder.add_light(Light::PointLight {
                position,
                intensity,
            });
            state.has_lights = true;
        }
        other => {
            warn!("unsupported light type '{}', skipping", other);
        }
    }

    Ok(())
}

fn parse_area_light_source_directive(
    toks: &mut TokenStream,
    state: &mut ParserState,
) -> Result<(), ParseError> {
    let light_type = parse_type_string(toks)?;
    let params = parse_parameter_list(toks)?;

    if light_type == "diffuse" {
        let radiance = params.get_rgb_or("L", Vec3(1.0, 1.0, 1.0));
        let scale = params.get_float_or("scale", 1.0);
        state.area_light_radiance = Some(radiance * scale);
    } else {
        warn!("unsupported area light type '{}', ignoring", light_type);
    }

    Ok(())
}

fn parse_world_begin(state: &mut ParserState) {
    state.current_transform = Transform::identity();
    state.in_world_block = true;
}

fn skip_directive(toks: &mut TokenStream) -> Result<(), ParseError> {
    if let Some(tok) = toks.peek() {
        if tok.starts_with('"') {
            toks.next();
        }
    }

    let _ = parse_parameter_list(toks)?;
    Ok(())
}

pub fn scene_from_pbrt_file(filepath: &Path) -> Result<Scene, ParseError> {
    let base_path = filepath.parent().unwrap_or(Path::new("."));

    let content = std::fs::read_to_string(filepath)
        .map_err(|e| ParseError::FileError(format!("{}: {}", filepath.display(), e)))?;

    parse_pbrt_string(&content, base_path)
}

fn parse_pbrt_string(content: &str, base_path: &Path) -> Result<Scene, ParseError> {
    let mut state = ParserState::new();
    let mut builder = SceneBuilder::new();

    parse_pbrt_content(content, base_path, &mut state, &mut builder)?;

    if !state.has_camera {
        warn!("no camera in scene");
        todo!("error?");
    }

    if !state.has_lights {
        warn!("no lights found in scene, adding default light");
        todo!("error?")
    }

    Ok(builder.build())
}

fn parse_pbrt_content(
    content: &str,
    base_path: &Path,
    state: &mut ParserState,
    builder: &mut SceneBuilder,
) -> Result<(), ParseError> {
    let mut toks = TokenStream::new(content);

    while let Some(directive) = toks.next() {
        match directive {
            "Identity" => {
                parse_identity_directive(state);
            }
            "LookAt" => {
                parse_lookat_directive(&mut toks, state)?;
            }
            "Translate" => {
                parse_translate_directive(&mut toks, state)?;
            }
            "Scale" => {
                parse_scale_directive(&mut toks, state)?;
            }
            "Rotate" => {
                parse_rotate_directive(&mut toks, state)?;
            }
            "Transform" => {
                parse_transform_directive(&mut toks, state)?;
            }
            "ConcatTransform" => {
                parse_concat_transform_directive(&mut toks, state)?;
            }
            "Film" => {
                parse_film_directive(&mut toks, state)?;
            }
            "Camera" => {
                parse_camera_directive(&mut toks, state, builder)?;
            }
            "Material" => {
                parse_material_directive(&mut toks, state, builder)?;
            }
            "MakeNamedMaterial" => {
                parse_make_named_material_directive(&mut toks, state, builder)?;
            }
            "NamedMaterial" => {
                parse_named_material_directive(&mut toks, state)?;
            }
            "Texture" => {
                parse_texture_directive(&mut toks, state, builder, base_path)?;
            }
            "Shape" => {
                parse_shape_directive(&mut toks, state, builder, base_path)?;
            }
            "LightSource" => {
                parse_light_source_directive(&mut toks, state, builder)?;
            }
            "AreaLightSource" => {
                parse_area_light_source_directive(&mut toks, state)?;
            }
            "WorldBegin" => {
                parse_world_begin(state);
            }
            "WorldEnd" => {
                break;
            }
            "AttributeBegin" => {
                state.push_attributes();
            }
            "AttributeEnd" => {
                state.pop_attributes();
            }
            "TransformBegin" => {
                state.push_attributes();
            }
            "TransformEnd" => {
                state.pop_attributes();
            }
            "Include" => {
                let include_path_str = parse_type_string(&mut toks)?;
                let include_path = base_path.join(include_path_str);

                let included_content = std::fs::read_to_string(&include_path)
                    .map_err(|e| ParseError::FileError(format!("{}: {}", include_path.display(), e)))?;

                let include_base = include_path.parent().unwrap_or(base_path);
                parse_pbrt_content(&included_content, include_base, state, builder)?;
            }
            "Sampler" | "Integrator" | "PixelFilter" | "Accelerator" | "ColorSpace" => {
                skip_directive(&mut toks)?;
            }
            "ReverseOrientation" => {
                // ignored
            }
            "ObjectBegin" | "ObjectEnd" | "ObjectInstance" => {
                if directive == "ObjectBegin" || directive == "ObjectInstance" {
                    skip_directive(&mut toks)?;
                }
                warn!("instancing (ObjectBegin/End/Instance) not supported");
            }
            "MediumInterface" | "MakeNamedMedium" => {
                skip_directive(&mut toks)?;
                warn!("media/volumes not supported");
            }
            other => {
                if other.starts_with('"') {
                    continue;
                }
                warn!("unknown directive '{}', ignoring", other);
            }
        }
    }

    Ok(())
}
