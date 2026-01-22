//! PBRT scene importer

use std::{iter::Peekable, num::ParseFloatError, path::Path, str::SplitAsciiWhitespace};

use crate::{geometry::{Matrix4x4, Transform, Vec3}, scene::Scene};

struct ParserState {
    current_transform: Transform
}

enum ParseError {
    UnknownDirective,

    BadFloat(ParseFloatError),
    BadString,

    UnexpectedEOF,
}

impl From<ParseFloatError> for ParseError {
    fn from(value: ParseFloatError) -> Self {
        ParseError::BadFloat(value)
    }
}

enum Directive {
    Option,
    Camera,
    Sampler,
    Integrator,
    Film,
    WorldBegin,
    LightSource,
    AttributeBegin,
    AttributeEnd,
    Material,
    Shape,

    Identity,
    Translate,
    Scale,
    Rotate,
    Transform,
    ConcatTransform,
    LookAt,
}

struct ParserInput {
    open_files: Vec<String>
}

struct TokenStack<'input> {
    cursors: Vec<SplitAsciiWhitespace<'input>>
}

type TokenStream<'input> = Peekable<TokenStack<'input>>;

impl<'input> Iterator for TokenStack<'input> {
    type Item = &'input str;
    
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let top_of_stack = self.cursors.last_mut()?;

            match top_of_stack.next() {
                Some(v) => break Some(v),
                None => { self.cursors.pop(); },
            }
        }
    }
}

enum Parameter {
    Integer(i32),
    Float(f32),
    Point2(f32, f32),
    Vector2(f32, f32),
    Point3(f32, f32, f32),
    Vector3(f32, f32, f32),
    Normal3(f32, f32, f32),
    Rgb(f32, f32, f32),
    Bool(bool),
    String(String)
}

struct ParameterList {
    parameters: Vec<(String, Parameter)>
}

fn parse_directive(toks: &mut TokenStream<'_>) -> Result<Directive, ParseError> {
    let Some(directive_tok) = toks.next() else {
        return Err(ParseError::UnexpectedEOF);
    };

    let directive = match directive_tok {
        "Option" => Directive::Option,
        "Camera" => Directive::Camera,
        "Sampler" => Directive::Sampler,
        "Integrator" => Directive::Integrator,
        "Film" => Directive::Film,
        "WorldBegin" => Directive::WorldBegin,
        "LightSource" => Directive::LightSource,
        "AttributeBegin" => Directive::AttributeBegin,
        "AttributeEnd" => Directive::AttributeEnd,
        "Material" => Directive::Material,
        "Shape" => Directive::Shape,
        "Identity" => Directive::Identity,
        "Translate" => Directive::Translate,
        "Scale" => Directive::Scale,
        "Rotate" => Directive::Rotate,
        "Transform" => Directive::Transform,
        "ConcatTransform" => Directive::ConcatTransform,
        "LookAt" => Directive::LookAt,
        _ => return Err(ParseError::UnknownDirective),
    };

    Ok(directive)
}

fn parse_string<'t>(toks: &mut TokenStream<'t>) -> Result<&'t str, ParseError> {
    let Some("\"") = toks.next() else {
        return Err(ParseError::BadString)
    };

    let Some(s) = toks.next() else {
        return Err(ParseError::BadString)  
    };

    let Some("\"") = toks.next() else {
        return Err(ParseError::BadString)
    };
    
    Ok(s)
}

fn parse_float(toks: &mut TokenStream<'_>) -> Result<f32, ParseError> {
    match toks.next() {
        Some(float_tok) => {
            Ok(float_tok.parse::<f32>()?)
        },
        None => Err(ParseError::UnexpectedEOF),
    }
}

fn parse_integer_param(toks: &mut TokenStream<'_>) -> Result<Parameter, ParseError> {
    todo!();
}

fn parse_float_param(toks: &mut TokenStream<'_>) -> Result<Parameter, ParseError> {
    todo!();
}

fn parse_point2_param(toks: &mut TokenStream<'_>) -> Result<Parameter, ParseError> {
    todo!();
}

fn parse_parameter(toks: &mut TokenStream<'_>) -> Result<Option<(String, Parameter)>, ParseError> {
    todo!();
}

fn parse_parameter_list(toks: &mut TokenStream<'_>) -> Result<ParameterList, ParseError> {
    while let Some(parameter) = parse_parameter(toks)? {
        
    }

    todo!()
}

fn parse_identity_directive(_toks: &mut TokenStream<'_>, state: &mut ParserState) -> Result<(), ParseError> {
    state.current_transform = Transform::identity();
    
    Ok(())
}

fn parse_lookat_directive(toks: &mut TokenStream<'_>, state: &mut ParserState) -> Result<(), ParseError> {
    let camera_pos = Vec3(
        parse_float(toks)?,
        parse_float(toks)?,
        parse_float(toks)?
    );

    let target_pos = Vec3(
        parse_float(toks)?,
        parse_float(toks)?,
        parse_float(toks)?
    );

    let up = Vec3(
        parse_float(toks)?,
        parse_float(toks)?,
        parse_float(toks)?
    );


    let look_at_transform = Transform::look_at(camera_pos, target_pos, up);
    state.current_transform = state.current_transform.compose(look_at_transform);

    Ok(())
}

fn parse_translate_directive(toks: &mut TokenStream<'_>, state: &mut ParserState) -> Result<(), ParseError> {
    let direction = Vec3(
        parse_float(toks)?,
        parse_float(toks)?,
        parse_float(toks)?
    );

    let translate_transform = Transform::translate(direction);
    state.current_transform = state.current_transform.compose(translate_transform);

    Ok(())
}

fn parse_scale_directive(toks: &mut TokenStream<'_>, state: &mut ParserState) -> Result<(), ParseError> {
    let scale = Vec3(
        parse_float(toks)?,
        parse_float(toks)?,
        parse_float(toks)?
    );

    let scale_transform = Transform::scale(scale);
    state.current_transform = state.current_transform.compose(scale_transform);

    Ok(())
}

fn parse_rotate_directive(toks: &mut TokenStream<'_>, state: &mut ParserState) -> Result<(), ParseError> {
    let angle = parse_float(toks)?;
    let x = parse_float(toks)?;
    let y = parse_float(toks)?;
    let z = parse_float(toks)?;
    let rotate_transform = Transform::rotate(angle, Vec3(x, y, z));

    state.current_transform = state.current_transform.compose(rotate_transform);
    Ok(())
}

fn parse_transform_directive(toks: &mut TokenStream<'_>, state: &mut ParserState) -> Result<(), ParseError> {
    // TODO: check that pbrt directive is row-major
    let mut m = [0.0f32; 16];
    for v in &mut m {
        *v = parse_float(toks)?;
    }

    let mat = Matrix4x4::from(m);
    let transform = Transform::from(mat);
    
    state.current_transform = transform;
    Ok(())
}

fn parse_transform_concat_directive(toks: &mut TokenStream<'_>, state: &mut ParserState) -> Result<(), ParseError> {
    let mut m = [0.0f32; 16];
    for v in &mut m {
        *v = parse_float(toks)?;
    }
    let mat = Matrix4x4::from(m);
    let transform = Transform::from(mat);
    
    state.current_transform = state.current_transform.compose(transform);
    Ok(())
}

fn parse_camera_directive(toks: &mut TokenStream<'_>, state: &mut ParserState) -> Result<(), ParseError> {
    let camera_type = parse_string(toks)?;

    let camera_type = match camera_type {
        "orthographic" => todo!(),
        "perspective" => todo!(),
        "realistic" | "spherical" => todo!("support more advanced cameras"),
        _ => todo!()
    };


}

fn parse_option_directive(toks: &mut TokenStream<'_>, state: &mut ParserState) -> Result<(), ParseError> {
    todo!();
}

pub fn scene_from_pbrt_file(filepath: &Path) -> Scene {
    let mut state = ParserState {
        current_transform: Transform::identity()
    };

    let file_contents = std::fs::read_to_string(filepath).expect("failed to read pbrt file");
    let mut tokens = file_contents.split_ascii_whitespace().peekable();


    todo!();   
}