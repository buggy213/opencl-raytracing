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

use std::{
    collections::HashMap,
    path::Path,
};

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
    Integer(Vec<i32>),
    Float(Vec<f32>),
    Point2(Vec<[f32; 2]>),
    Vector2(Vec<[f32; 2]>),
    Point3(Vec<[f32; 3]>),
    Vector3(Vec<[f32; 3]>),
    Normal3(Vec<[f32; 3]>),
    Rgb(Vec<[f32; 3]>),
    Bool(Vec<bool>),
    String(Vec<String>),
    Texture(Vec<String>),
}

#[derive(Debug, Clone, Default)]
struct ParameterList {
    params: HashMap<String, ParameterValue>,
}

impl ParameterList {
    fn get_float(&self, name: &str) -> Option<f32> {
        match self.params.get(name)? {
            ParameterValue::Float(v) if !v.is_empty() => Some(v[0]),
            _ => None,
        }
    }

    fn get_float_or(&self, name: &str, default: f32) -> f32 {
        self.get_float(name).unwrap_or(default)
    }

    fn get_integer(&self, name: &str) -> Option<i32> {
        match self.params.get(name)? {
            ParameterValue::Integer(v) if !v.is_empty() => Some(v[0]),
            _ => None,
        }
    }

    fn get_integer_or(&self, name: &str, default: i32) -> i32 {
        self.get_integer(name).unwrap_or(default)
    }

    fn get_integers(&self, name: &str) -> Option<&[i32]> {
        match self.params.get(name)? {
            ParameterValue::Integer(v) => Some(v),
            _ => None,
        }
    }

    fn get_floats(&self, name: &str) -> Option<&[f32]> {
        match self.params.get(name)? {
            ParameterValue::Float(v) => Some(v),
            _ => None,
        }
    }

    fn get_point3s(&self, name: &str) -> Option<&[[f32; 3]]> {
        match self.params.get(name)? {
            ParameterValue::Point3(v) => Some(v),
            _ => None,
        }
    }

    fn get_normal3s(&self, name: &str) -> Option<&[[f32; 3]]> {
        match self.params.get(name)? {
            ParameterValue::Normal3(v) => Some(v),
            _ => None,
        }
    }

    fn get_point2s(&self, name: &str) -> Option<&[[f32; 2]]> {
        match self.params.get(name)? {
            ParameterValue::Point2(v) => Some(v),
            _ => None,
        }
    }

    fn get_rgb(&self, name: &str) -> Option<Vec3> {
        match self.params.get(name)? {
            ParameterValue::Rgb(v) if !v.is_empty() => Some(Vec3(v[0][0], v[0][1], v[0][2])),
            ParameterValue::Float(v) if v.len() >= 3 => Some(Vec3(v[0], v[1], v[2])),
            _ => None,
        }
    }

    fn get_rgb_or(&self, name: &str, default: Vec3) -> Vec3 {
        self.get_rgb(name).unwrap_or(default)
    }

    fn get_string(&self, name: &str) -> Option<&str> {
        match self.params.get(name)? {
            ParameterValue::String(v) if !v.is_empty() => Some(&v[0]),
            ParameterValue::Texture(v) if !v.is_empty() => Some(&v[0]),
            _ => None,
        }
    }

    fn get_bool(&self, name: &str) -> Option<bool> {
        match self.params.get(name)? {
            ParameterValue::Bool(v) if !v.is_empty() => Some(v[0]),
            _ => None,
        }
    }

    fn get_bool_or(&self, name: &str, default: bool) -> bool {
        self.get_bool(name).unwrap_or(default)
    }

    fn get_texture(&self, name: &str) -> Option<&str> {
        match self.params.get(name)? {
            ParameterValue::Texture(v) if !v.is_empty() => Some(&v[0]),
            ParameterValue::String(v) if !v.is_empty() => Some(&v[0]),
            _ => None,
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
    named_materials: HashMap<String, MaterialIndex>,
    named_textures: HashMap<String, TextureId>,
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
            named_materials: HashMap::new(),
            named_textures: HashMap::new(),
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
}

fn strip_comments(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    for line in input.lines() {
        if let Some(pos) = line.find('#') {
            result.push_str(&line[..pos]);
        } else {
            result.push_str(line);
        }
        result.push('\n');
    }
    result
}

fn tokenize(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }

        if c == '"' {
            chars.next();
            let mut s = String::new();
            while let Some(&c) = chars.peek() {
                if c == '"' {
                    chars.next();
                    break;
                }
                s.push(c);
                chars.next();
            }
            tokens.push(format!("\"{}\"", s));
        } else if c == '[' {
            tokens.push("[".to_string());
            chars.next();
        } else if c == ']' {
            tokens.push("]".to_string());
            chars.next();
        } else {
            let mut word = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_whitespace() || c == '[' || c == ']' || c == '"' {
                    break;
                }
                word.push(c);
                chars.next();
            }
            if !word.is_empty() {
                tokens.push(word);
            }
        }
    }

    tokens
}

struct TokenStream {
    tokens: Vec<String>,
    pos: usize,
}

impl TokenStream {
    fn new(tokens: Vec<String>) -> Self {
        TokenStream { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&str> {
        self.tokens.get(self.pos).map(|s| s.as_str())
    }

    fn next(&mut self) -> Option<&str> {
        if self.pos < self.tokens.len() {
            let tok = &self.tokens[self.pos];
            self.pos += 1;
            Some(tok)
        } else {
            None
        }
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

    fn is_eof(&self) -> bool {
        self.pos >= self.tokens.len()
    }
}

fn parse_quoted_string(tok: &str) -> Result<String, ParseError> {
    if tok.starts_with('"') && tok.ends_with('"') && tok.len() >= 2 {
        Ok(tok[1..tok.len() - 1].to_string())
    } else {
        Err(ParseError::BadString)
    }
}

fn parse_float(tok: &str) -> Result<f32, ParseError> {
    tok.parse::<f32>()
        .map_err(|_| ParseError::BadFloat(tok.to_string()))
}

fn parse_integer(tok: &str) -> Result<i32, ParseError> {
    tok.parse::<i32>()
        .map_err(|_| ParseError::BadInteger(tok.to_string()))
}

fn parse_float_or_int_as_float(tok: &str) -> Result<f32, ParseError> {
    if let Ok(f) = tok.parse::<f32>() {
        Ok(f)
    } else if let Ok(i) = tok.parse::<i32>() {
        Ok(i as f32)
    } else {
        Err(ParseError::BadFloat(tok.to_string()))
    }
}

fn parse_type_string(toks: &mut TokenStream) -> Result<String, ParseError> {
    let tok = toks.next().ok_or(ParseError::UnexpectedEOF)?;
    parse_quoted_string(tok)
}

fn parse_floats(toks: &mut TokenStream) -> Result<f32, ParseError> {
    let tok = toks.next().ok_or(ParseError::UnexpectedEOF)?;
    parse_float_or_int_as_float(tok)
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
            return Err(ParseError::BadParameter(type_name));
        }

        let param_type = parts[0];
        let param_name = parts[1].to_string();

        let value = parse_parameter_value(toks, param_type)?;
        params.params.insert(param_name, value);
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
                if tok == "]" {
                    break;
                }
                if tok.starts_with('"') {
                    break;
                }
                let v = parse_integer(toks.next().unwrap())?;
                values.push(v);
                if !has_brackets {
                    break;
                }
            }
            ParameterValue::Integer(values)
        }
        "float" => {
            let mut values = Vec::new();
            while let Some(tok) = toks.peek() {
                if tok == "]" {
                    break;
                }
                if tok.starts_with('"') {
                    break;
                }
                let v = parse_float_or_int_as_float(toks.next().unwrap())?;
                values.push(v);
                if !has_brackets {
                    break;
                }
            }
            ParameterValue::Float(values)
        }
        "point2" => {
            let mut values = Vec::new();
            while let Some(tok) = toks.peek() {
                if tok == "]" {
                    break;
                }
                if tok.starts_with('"') {
                    break;
                }
                let x = parse_float_or_int_as_float(toks.next().unwrap())?;
                let y = parse_float_or_int_as_float(toks.next().ok_or(ParseError::UnexpectedEOF)?)?;
                values.push([x, y]);
                if !has_brackets {
                    break;
                }
            }
            ParameterValue::Point2(values)
        }
        "vector2" => {
            let mut values = Vec::new();
            while let Some(tok) = toks.peek() {
                if tok == "]" {
                    break;
                }
                if tok.starts_with('"') {
                    break;
                }
                let x = parse_float_or_int_as_float(toks.next().unwrap())?;
                let y = parse_float_or_int_as_float(toks.next().ok_or(ParseError::UnexpectedEOF)?)?;
                values.push([x, y]);
                if !has_brackets {
                    break;
                }
            }
            ParameterValue::Vector2(values)
        }
        "point3" | "point" => {
            let mut values = Vec::new();
            while let Some(tok) = toks.peek() {
                if tok == "]" {
                    break;
                }
                if tok.starts_with('"') {
                    break;
                }
                let x = parse_float_or_int_as_float(toks.next().unwrap())?;
                let y = parse_float_or_int_as_float(toks.next().ok_or(ParseError::UnexpectedEOF)?)?;
                let z = parse_float_or_int_as_float(toks.next().ok_or(ParseError::UnexpectedEOF)?)?;
                values.push([x, y, z]);
                if !has_brackets {
                    break;
                }
            }
            ParameterValue::Point3(values)
        }
        "vector3" | "vector" => {
            let mut values = Vec::new();
            while let Some(tok) = toks.peek() {
                if tok == "]" {
                    break;
                }
                if tok.starts_with('"') {
                    break;
                }
                let x = parse_float_or_int_as_float(toks.next().unwrap())?;
                let y = parse_float_or_int_as_float(toks.next().ok_or(ParseError::UnexpectedEOF)?)?;
                let z = parse_float_or_int_as_float(toks.next().ok_or(ParseError::UnexpectedEOF)?)?;
                values.push([x, y, z]);
                if !has_brackets {
                    break;
                }
            }
            ParameterValue::Vector3(values)
        }
        "normal3" | "normal" => {
            let mut values = Vec::new();
            while let Some(tok) = toks.peek() {
                if tok == "]" {
                    break;
                }
                if tok.starts_with('"') {
                    break;
                }
                let x = parse_float_or_int_as_float(toks.next().unwrap())?;
                let y = parse_float_or_int_as_float(toks.next().ok_or(ParseError::UnexpectedEOF)?)?;
                let z = parse_float_or_int_as_float(toks.next().ok_or(ParseError::UnexpectedEOF)?)?;
                values.push([x, y, z]);
                if !has_brackets {
                    break;
                }
            }
            ParameterValue::Normal3(values)
        }
        "rgb" | "color" => {
            let mut values = Vec::new();
            while let Some(tok) = toks.peek() {
                if tok == "]" {
                    break;
                }
                if tok.starts_with('"') {
                    break;
                }
                let r = parse_float_or_int_as_float(toks.next().unwrap())?;
                let g = parse_float_or_int_as_float(toks.next().ok_or(ParseError::UnexpectedEOF)?)?;
                let b = parse_float_or_int_as_float(toks.next().ok_or(ParseError::UnexpectedEOF)?)?;
                values.push([r, g, b]);
                if !has_brackets {
                    break;
                }
            }
            ParameterValue::Rgb(values)
        }
        "bool" => {
            let mut values = Vec::new();
            while let Some(tok) = toks.peek() {
                if tok == "]" {
                    break;
                }
                if tok.starts_with('"') && !tok.contains("true") && !tok.contains("false") {
                    break;
                }
                let tok = toks.next().unwrap();
                let tok_clean = tok.trim_matches('"');
                let v = match tok_clean {
                    "true" => true,
                    "false" => false,
                    _ => return Err(ParseError::BadBool(tok.to_string())),
                };
                values.push(v);
                if !has_brackets {
                    break;
                }
            }
            ParameterValue::Bool(values)
        }
        "string" => {
            let mut values = Vec::new();
            while let Some(tok) = toks.peek() {
                if tok == "]" {
                    break;
                }
                if !tok.starts_with('"') {
                    break;
                }
                let s = parse_quoted_string(toks.next().unwrap())?;
                values.push(s);
                if !has_brackets {
                    break;
                }
            }
            ParameterValue::String(values)
        }
        "texture" => {
            let mut values = Vec::new();
            while let Some(tok) = toks.peek() {
                if tok == "]" {
                    break;
                }
                if !tok.starts_with('"') {
                    break;
                }
                let s = parse_quoted_string(toks.next().unwrap())?;
                values.push(s);
                if !has_brackets {
                    break;
                }
            }
            ParameterValue::Texture(values)
        }
        "spectrum" => {
            warn!("spectrum parameters not fully supported, treating as RGB");
            let mut values = Vec::new();
            while let Some(tok) = toks.peek() {
                if tok == "]" {
                    break;
                }
                if tok.starts_with('"') {
                    break;
                }
                let r = parse_float_or_int_as_float(toks.next().unwrap())?;
                let g = parse_float_or_int_as_float(toks.next().ok_or(ParseError::UnexpectedEOF)?)?;
                let b = parse_float_or_int_as_float(toks.next().ok_or(ParseError::UnexpectedEOF)?)?;
                values.push([r, g, b]);
                if !has_brackets {
                    break;
                }
            }
            ParameterValue::Rgb(values)
        }
        _ => {
            warn!("unknown parameter type '{}', skipping", param_type);
            while let Some(tok) = toks.peek() {
                if tok == "]" {
                    break;
                }
                if tok.starts_with('"') {
                    break;
                }
                toks.next();
                if !has_brackets {
                    break;
                }
            }
            ParameterValue::Float(Vec::new())
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
    let eye_x = parse_floats(toks)?;
    let eye_y = parse_floats(toks)?;
    let eye_z = parse_floats(toks)?;

    let look_x = parse_floats(toks)?;
    let look_y = parse_floats(toks)?;
    let look_z = parse_floats(toks)?;

    let up_x = parse_floats(toks)?;
    let up_y = parse_floats(toks)?;
    let up_z = parse_floats(toks)?;

    let eye = Vec3(eye_x, eye_y, eye_z);
    let look = Vec3(look_x, look_y, look_z);
    let up = Vec3(up_x, up_y, up_z);

    let look_at_transform = Transform::look_at(eye, look, up);
    state.current_transform = state.current_transform.compose(look_at_transform.invert());

    Ok(())
}

fn parse_translate_directive(toks: &mut TokenStream, state: &mut ParserState) -> Result<(), ParseError> {
    let x = parse_floats(toks)?;
    let y = parse_floats(toks)?;
    let z = parse_floats(toks)?;

    let translate_transform = Transform::translate(Vec3(x, y, z));
    state.current_transform = state.current_transform.compose(translate_transform);

    Ok(())
}

fn parse_scale_directive(toks: &mut TokenStream, state: &mut ParserState) -> Result<(), ParseError> {
    let x = parse_floats(toks)?;
    let y = parse_floats(toks)?;
    let z = parse_floats(toks)?;

    let scale_transform = Transform::scale(Vec3(x, y, z));
    state.current_transform = state.current_transform.compose(scale_transform);

    Ok(())
}

fn parse_rotate_directive(toks: &mut TokenStream, state: &mut ParserState) -> Result<(), ParseError> {
    let angle = parse_floats(toks)?;
    let x = parse_floats(toks)?;
    let y = parse_floats(toks)?;
    let z = parse_floats(toks)?;

    let rotate_transform = Transform::rotate(angle.to_radians(), Vec3(x, y, z));
    state.current_transform = state.current_transform.compose(rotate_transform);

    Ok(())
}

fn parse_transform_directive(toks: &mut TokenStream, state: &mut ParserState) -> Result<(), ParseError> {
    toks.expect("[")?;

    let mut m = [0.0f32; 16];
    for v in &mut m {
        *v = parse_floats(toks)?;
    }

    toks.expect("]")?;

    let mat = Matrix4x4::from(m);
    let transform = Transform::from(mat);
    state.current_transform = transform;

    Ok(())
}

fn parse_concat_transform_directive(toks: &mut TokenStream, state: &mut ParserState) -> Result<(), ParseError> {
    toks.expect("[")?;

    let mut m = [0.0f32; 16];
    for v in &mut m {
        *v = parse_floats(toks)?;
    }

    toks.expect("]")?;

    let mat = Matrix4x4::from(m);
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

    let camera = match camera_type.as_str() {
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
        if let Some(&tex_id) = state.named_textures.get(tex_name) {
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
        if let Some(&tex_id) = state.named_textures.get(tex_name) {
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

    let material = create_material(&material_type, &params, state, builder)?;
    state.current_material = Some(builder.add_material(material));

    Ok(())
}

fn create_material(
    material_type: &str,
    params: &ParameterList,
    state: &ParserState,
    builder: &mut SceneBuilder,
) -> Result<Material, ParseError> {
    let material = match material_type {
        "diffuse" | "matte" => {
            let albedo = resolve_texture(state, builder, params, "reflectance", Vec3(0.5, 0.5, 0.5));
            Material::Diffuse { albedo }
        }
        "conductor" | "metal" => {
            let eta = resolve_texture(state, builder, params, "eta", Vec3(0.2, 0.2, 0.2));
            let k = resolve_texture(state, builder, params, "k", Vec3(3.0, 3.0, 3.0));
            let roughness = params.get_float("roughness");
            let uroughness = params.get_float("uroughness");
            let vroughness = params.get_float("vroughness");

            let is_rough = roughness.is_some() || uroughness.is_some() || vroughness.is_some();

            if is_rough {
                let u = uroughness.or(roughness).unwrap_or(0.5);
                let v = vroughness.or(roughness).unwrap_or(0.5);
                let roughness_tex = builder.add_constant_texture(Vec4(u, v, 0.0, 0.0));
                Material::RoughConductor {
                    eta,
                    kappa: k,
                    roughness: roughness_tex,
                }
            } else {
                Material::SmoothConductor { eta, kappa: k }
            }
        }
        "dielectric" | "glass" => {
            let ior = params.get_float_or("eta", 1.5);
            let eta = builder.add_constant_texture(Vec4(ior, 0.0, 0.0, 0.0));

            let roughness = params.get_float("roughness");
            let uroughness = params.get_float("uroughness");
            let vroughness = params.get_float("vroughness");

            let is_rough = roughness.is_some() || uroughness.is_some() || vroughness.is_some();

            if is_rough {
                let u = uroughness.or(roughness).unwrap_or(0.5);
                let v = vroughness.or(roughness).unwrap_or(0.5);
                let roughness_tex = builder.add_constant_texture(Vec4(u, v, 0.0, 0.0));
                Material::RoughDielectric { eta, roughness: roughness_tex }
            } else {
                Material::SmoothDielectric { eta }
            }
        }
        "coateddiffuse" => {
            let albedo = resolve_texture(state, builder, params, "reflectance", Vec3(0.5, 0.5, 0.5));
            Material::Diffuse { albedo }
        }
        "coatedconductor" => {
            let eta = resolve_texture(state, builder, params, "conductor.eta", Vec3(0.2, 0.2, 0.2));
            let k = resolve_texture(state, builder, params, "conductor.k", Vec3(3.0, 3.0, 3.0));
            let roughness = params.get_float_or("roughness", 0.0);

            if roughness > 0.0 {
                let roughness_tex = builder.add_constant_texture(Vec4(roughness, roughness, 0.0, 0.0));
                Material::RoughConductor {
                    eta,
                    kappa: k,
                    roughness: roughness_tex,
                }
            } else {
                Material::SmoothConductor { eta, kappa: k }
            }
        }
        "plastic" => {
            let albedo = resolve_texture(state, builder, params, "Kd", Vec3(0.25, 0.25, 0.25));
            Material::Diffuse { albedo }
        }
        "uber" => {
            let albedo = resolve_texture(state, builder, params, "Kd", Vec3(0.25, 0.25, 0.25));
            Material::Diffuse { albedo }
        }
        "mirror" => {
            let eta = builder.add_constant_texture(Vec4(100.0, 100.0, 100.0, 0.0));
            let kappa = builder.add_constant_texture(Vec4(1.0, 1.0, 1.0, 0.0));
            Material::SmoothConductor { eta, kappa }
        }
        "substrate" => {
            let albedo = resolve_texture(state, builder, params, "Kd", Vec3(0.5, 0.5, 0.5));
            Material::Diffuse { albedo }
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
    state.named_materials.insert(name, material_id);

    Ok(())
}

fn parse_named_material_directive(
    toks: &mut TokenStream,
    state: &mut ParserState,
) -> Result<(), ParseError> {
    let name = parse_type_string(toks)?;

    if let Some(&material_id) = state.named_materials.get(&name) {
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

    let texture = match tex_type.as_str() {
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
    state.named_textures.insert(name, tex_id);

    Ok(())
}

fn parse_shape_directive(
    toks: &mut TokenStream,
    state: &mut ParserState,
    builder: &mut SceneBuilder,
) -> Result<(), ParseError> {
    let shape_type = parse_type_string(toks)?;
    let params = parse_parameter_list(toks)?;

    let material_id = state.current_material.unwrap_or_else(|| {
        let albedo = builder.add_constant_texture(Vec4(0.5, 0.5, 0.5, 1.0));
        builder.add_material(Material::Diffuse { albedo })
    });

    let shape = match shape_type.as_str() {
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
            warn!("plymesh not supported, creating placeholder sphere");
            Shape::Sphere {
                center: Vec3(0.0, 0.0, 0.0),
                radius: 1.0,
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

    match light_type.as_str() {
        "point" => {
            let intensity = params.get_rgb_or("I", Vec3(1.0, 1.0, 1.0));
            let scale = params.get_float_or("scale", 1.0);
            let from = params
                .get_point3s("from")
                .map(|p| Vec3(p[0][0], p[0][1], p[0][2]))
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
                .get_point3s("from")
                .map(|p| Vec3(p[0][0], p[0][1], p[0][2]))
                .unwrap_or(Vec3(0.0, 0.0, 1.0));
            let to = params
                .get_point3s("to")
                .map(|p| Vec3(p[0][0], p[0][1], p[0][2]))
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
                .get_point3s("from")
                .map(|p| Vec3(p[0][0], p[0][1], p[0][2]))
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

fn load_and_preprocess(filepath: &Path) -> Result<String, ParseError> {
    let content = std::fs::read_to_string(filepath)
        .map_err(|e| ParseError::FileError(format!("{}: {}", filepath.display(), e)))?;

    Ok(strip_comments(&content))
}

fn parse_include_directive(
    toks: &mut TokenStream,
    base_path: &Path,
    all_tokens: &mut Vec<String>,
    insert_pos: &mut usize,
) -> Result<(), ParseError> {
    let include_path_str = parse_type_string(toks)?;
    let include_path = base_path.join(&include_path_str);

    let included_content = load_and_preprocess(&include_path)?;
    let included_tokens = tokenize(&included_content);

    for tok in included_tokens.into_iter().rev() {
        all_tokens.insert(*insert_pos, tok);
    }

    Ok(())
}

pub fn scene_from_pbrt_file(filepath: &Path) -> Result<Scene, ParseError> {
    let base_path = filepath.parent().unwrap_or(Path::new("."));

    let content = load_and_preprocess(filepath)?;
    let mut all_tokens = tokenize(&content);

    let mut state = ParserState::new();
    let mut builder = SceneBuilder::new();

    let mut pos = 0;
    while pos < all_tokens.len() {
        let tok = &all_tokens[pos];
        pos += 1;

        let mut temp_stream = TokenStream {
            tokens: all_tokens.clone(),
            pos,
        };

        match tok.as_str() {
            "Identity" => {
                parse_identity_directive(&mut state);
            }
            "LookAt" => {
                parse_lookat_directive(&mut temp_stream, &mut state)?;
                pos = temp_stream.pos;
            }
            "Translate" => {
                parse_translate_directive(&mut temp_stream, &mut state)?;
                pos = temp_stream.pos;
            }
            "Scale" => {
                parse_scale_directive(&mut temp_stream, &mut state)?;
                pos = temp_stream.pos;
            }
            "Rotate" => {
                parse_rotate_directive(&mut temp_stream, &mut state)?;
                pos = temp_stream.pos;
            }
            "Transform" => {
                parse_transform_directive(&mut temp_stream, &mut state)?;
                pos = temp_stream.pos;
            }
            "ConcatTransform" => {
                parse_concat_transform_directive(&mut temp_stream, &mut state)?;
                pos = temp_stream.pos;
            }
            "Film" => {
                parse_film_directive(&mut temp_stream, &mut state)?;
                pos = temp_stream.pos;
            }
            "Camera" => {
                parse_camera_directive(&mut temp_stream, &mut state, &mut builder)?;
                pos = temp_stream.pos;
            }
            "Material" => {
                parse_material_directive(&mut temp_stream, &mut state, &mut builder)?;
                pos = temp_stream.pos;
            }
            "MakeNamedMaterial" => {
                parse_make_named_material_directive(&mut temp_stream, &mut state, &mut builder)?;
                pos = temp_stream.pos;
            }
            "NamedMaterial" => {
                parse_named_material_directive(&mut temp_stream, &mut state)?;
                pos = temp_stream.pos;
            }
            "Texture" => {
                parse_texture_directive(&mut temp_stream, &mut state, &mut builder, base_path)?;
                pos = temp_stream.pos;
            }
            "Shape" => {
                parse_shape_directive(&mut temp_stream, &mut state, &mut builder)?;
                pos = temp_stream.pos;
            }
            "LightSource" => {
                parse_light_source_directive(&mut temp_stream, &mut state, &mut builder)?;
                pos = temp_stream.pos;
            }
            "AreaLightSource" => {
                parse_area_light_source_directive(&mut temp_stream, &mut state)?;
                pos = temp_stream.pos;
            }
            "WorldBegin" => {
                parse_world_begin(&mut state);
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
                let include_path_str = parse_type_string(&mut temp_stream)?;
                pos = temp_stream.pos;

                let include_path = base_path.join(&include_path_str);
                let included_content = load_and_preprocess(&include_path)?;
                let included_tokens = tokenize(&included_content);

                for tok in included_tokens.into_iter().rev() {
                    all_tokens.insert(pos, tok);
                }
            }
            "Sampler" | "Integrator" | "PixelFilter" | "Accelerator" | "ColorSpace" => {
                skip_directive(&mut temp_stream)?;
                pos = temp_stream.pos;
            }
            "ReverseOrientation" => {
                // ignored
            }
            "ObjectBegin" | "ObjectEnd" | "ObjectInstance" => {
                if tok.as_str() == "ObjectBegin" || tok.as_str() == "ObjectInstance" {
                    skip_directive(&mut temp_stream)?;
                    pos = temp_stream.pos;
                }
                warn!("instancing (ObjectBegin/End/Instance) not supported");
            }
            "MediumInterface" | "MakeNamedMedium" => {
                skip_directive(&mut temp_stream)?;
                pos = temp_stream.pos;
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

    if !state.has_camera {
        builder.add_camera(Camera::lookat_camera_perspective(
            Vec3(0.0, 0.0, 0.0),
            Vec3(0.0, 0.0, -1.0),
            Vec3(0.0, 1.0, 0.0),
            (90.0_f32).to_radians(),
            state.film_width,
            state.film_height,
        ));
    }

    if !state.has_lights {
        warn!("no lights found in scene, adding default light");
        builder.add_light(Light::DirectionLight {
            direction: Vec3(0.0, 0.0, -1.0),
            radiance: Vec3(1.0, 1.0, 1.0),
        });
    }

    Ok(builder.build())
}
