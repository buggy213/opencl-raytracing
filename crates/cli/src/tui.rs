use std::io;

use crossterm::
    event::{self, Event, KeyCode, KeyEvent}
;
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Wrap},
};
use crate::{CommandLineArguments, InputScene, RenderCommand};

use super::{Backend, OutputFormat, SamplerType};
use raytracing::scene::test_scenes;


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneInputType {
    Path,
    BuiltinName,
}

impl SceneInputType {
    fn cycle_next(self) -> Self {
        match self {
            Self::Path => Self::BuiltinName,
            Self::BuiltinName => Self::Path,
        }
    }

    fn cycle_prev(self) -> Self {
        match self {
            Self::Path => Self::BuiltinName,
            Self::BuiltinName => Self::Path,
        }
    }
}

impl Backend {
    fn cycle_next(self) -> Self {
        match self {
            Self::Cpu => Self::Optix,
            Self::Optix => Self::Cpu,
        }
    }
    fn cycle_prev(self) -> Self {
        match self {
            Self::Cpu => Self::Optix,
            Self::Optix => Self::Cpu,
        }
    }
}

impl OutputFormat {
    fn cycle_next(self) -> Self {
        match self {
            Self::Png => Self::Exr,
            Self::Exr => Self::Png,
        }
    }
    fn cycle_prev(self) -> Self {
        match self {
            Self::Png => Self::Exr,
            Self::Exr => Self::Png,
        }
    }
}

impl SamplerType {
    fn cycle_next(self) -> Self {
        match self {
            Self::Independent => Self::Stratified,
            Self::Stratified => Self::Independent,
        }
    }
    fn cycle_prev(self) -> Self {
        match self {
            Self::Independent => Self::Stratified,
            Self::Stratified => Self::Independent,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderMode {
    Full,
    Pixel,
}

impl RenderMode {
    fn cycle_next(self) -> Self {
        match self {
            Self::Full => Self::Pixel,
            Self::Pixel => Self::Full,
        }
    }
    fn cycle_prev(self) -> Self {
        match self {
            Self::Full => Self::Pixel,
            Self::Pixel => Self::Full,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusedField {
    SceneInputType,
    ScenePath,
    SceneName,
    Backend,
    OutputPath,
    OutputFormat,
    NumThreads,
    RayDepth,
    Spp,
    LightSamples,
    Sampler,
    RenderMode,
    PixelX,
    PixelY,
}

impl FocusedField {
    fn numeric_validation(self) -> bool {
        match self {
            FocusedField::NumThreads 
            | FocusedField::RayDepth 
            | FocusedField::Spp 
            | FocusedField::LightSamples 
            | FocusedField::PixelX 
            | FocusedField::PixelY => true,
            _ => false
        }
    }
}

pub struct Model {
    // Scene input
    pub scene_input_type: SceneInputType,
    pub scene_path: String,
    pub scene_name_index: usize,
    pub available_scenes: Vec<String>,

    // Output
    pub output_path: String,
    pub output_format: OutputFormat,

    // Backend
    pub backend: Backend,

    // Render settings
    pub num_threads: u32,
    pub ray_depth: u32,
    pub spp: u32,
    pub light_samples: u32,
    pub sampler: SamplerType,

    // Render mode
    pub render_mode: RenderMode,
    pub pixel_x: u32,
    pub pixel_y: u32,

    // UI state
    pub focused_field: FocusedField,
    pub editing_text: bool,
    pub text_buffer: String,
    pub should_quit: bool,
    pub should_render: bool,
    pub error_message: Option<String>,
}

impl Default for Model {
    fn default() -> Self {
        let available_scenes: Vec<String> = test_scenes::all_test_scenes()
            .iter()
            .map(|s| s.name.to_string())
            .collect();

        Self {
            scene_input_type: SceneInputType::BuiltinName,
            scene_path: String::new(),
            scene_name_index: 0,
            available_scenes,

            output_path: "output.exr".to_string(),
            output_format: OutputFormat::Exr,

            backend: Backend::Cpu,

            num_threads: num_cpus::get() as u32,
            ray_depth: 8,
            spp: 4,
            light_samples: 4,
            sampler: SamplerType::Independent,

            render_mode: RenderMode::Full,
            pixel_x: 256,
            pixel_y: 256,

            focused_field: FocusedField::SceneInputType,
            editing_text: false,
            text_buffer: String::new(),
            should_quit: false,
            should_render: false,
            error_message: None,
        }
    }
}

impl Model {
    fn get_left_fields(&self) -> Vec<FocusedField> {
        let mut fields = vec![FocusedField::SceneInputType];

        match self.scene_input_type {
            SceneInputType::Path => fields.push(FocusedField::ScenePath),
            SceneInputType::BuiltinName => fields.push(FocusedField::SceneName),
        }

        fields.push(FocusedField::OutputPath);
        fields.push(FocusedField::OutputFormat);

        fields
    }

    fn get_right_fields(&self) -> Vec<FocusedField> {
        let mut fields = vec![FocusedField::Backend];

        if matches!(self.backend, Backend::Cpu) {
            fields.push(FocusedField::NumThreads);
        }

        fields.push(FocusedField::RayDepth);
        fields.push(FocusedField::Spp);
        fields.push(FocusedField::LightSamples);
        fields.push(FocusedField::Sampler);
        fields.push(FocusedField::RenderMode);

        if matches!(self.render_mode, RenderMode::Pixel) {
            fields.push(FocusedField::PixelX);
            fields.push(FocusedField::PixelY);
        }

        fields
    }

    fn get_visible_fields(&self) -> Vec<FocusedField> {
        let mut fields = self.get_left_fields();
        fields.extend(self.get_right_fields());
        fields
    }

    fn focus_next(&mut self) {
        let visible = self.get_visible_fields();
        if let Some(pos) = visible.iter().position(|&f| f == self.focused_field) {
            self.focused_field = visible[(pos + 1) % visible.len()];
        }
    }

    fn focus_prev(&mut self) {
        let visible = self.get_visible_fields();
        if let Some(pos) = visible.iter().position(|&f| f == self.focused_field) {
            self.focused_field = visible[(pos + visible.len() - 1) % visible.len()];
        }
    }

    fn ensure_valid_focus(&mut self) {
        let visible = self.get_visible_fields();
        if !visible.contains(&self.focused_field) {
            self.focused_field = visible[0];
        }
    }

    pub fn generate_command(&self) -> Result<String, String> {
        let mut cmd = vec!["cargo run -p cli --".to_string()];

        match self.scene_input_type {
            SceneInputType::Path => {
                if self.scene_path.is_empty() {
                    return Err("Scene path is empty".to_string());
                }
                let absolute_scene_path = match std::fs::canonicalize(&self.scene_path) {
                    Ok(p) => p,
                    Err(_) => { return Err("not able to get realpath".to_string()); }
                };
                    
                cmd.push(format!("--scene-path \"{}\"", absolute_scene_path.to_string_lossy()));
            }
            SceneInputType::BuiltinName => {
                let name = &self.available_scenes[self.scene_name_index];
                cmd.push(format!("--scene-name {}", name));
            }
        }

        cmd.push(format!("-o \"{}\"", self.output_path));

        match self.output_format {
            OutputFormat::Png => cmd.push("--output-format png".to_string()),
            OutputFormat::Exr => cmd.push("--output-format exr".to_string()),
        }

        match self.backend {
            Backend::Cpu => {
                cmd.push("--backend cpu".to_string());
                cmd.push(format!("-t {}", self.num_threads));
            }
            Backend::Optix => cmd.push("--backend optix".to_string()),
        }

        cmd.push(format!("-d {}", self.ray_depth));
        cmd.push(format!("-s {}", self.spp));
        cmd.push(format!("-l {}", self.light_samples));

        match self.sampler {
            SamplerType::Independent => cmd.push("--sampler independent".to_string()),
            SamplerType::Stratified => cmd.push("--sampler stratified".to_string()),
        }

        match self.render_mode {
            RenderMode::Full => cmd.push("full".to_string()),
            RenderMode::Pixel => cmd.push(format!("pixel {} {}", self.pixel_x, self.pixel_y)),
        }

        Ok(cmd.join(" "))
    }

    fn validate(&self) -> Result<(), String> {
        match self.scene_input_type {
            SceneInputType::Path if self.scene_path.is_empty() => {
                return Err("Please enter a scene path".to_string())
            }
            _ => {}
        }
        Ok(())
    }

    pub fn into_args(self) -> CommandLineArguments {
        let input = match self.scene_input_type {
            SceneInputType::Path => InputScene {
                scene_path: Some(self.scene_path.into()),
                scene_name: None,
            },
            SceneInputType::BuiltinName => InputScene {
                scene_path: None,
                scene_name: Some(self.available_scenes[self.scene_name_index].clone()),
            },
        };

        let render_command = match self.render_mode {
            RenderMode::Full => RenderCommand::Full {
                aov: None,
                no_beauty: false,
            },
            RenderMode::Pixel => RenderCommand::Pixel {
                x: self.pixel_x,
                y: self.pixel_y,
                sample_count: None,
                sample_offset: None
            },
        };

        CommandLineArguments {
            interactive: false,
            input,
            output: Some(self.output_path.into()),
            output_format: Some(self.output_format),
            backend: self.backend,
            num_threads: Some(self.num_threads),
            ray_depth: Some(self.ray_depth),
            spp: Some(self.spp),
            light_samples: Some(self.light_samples),
            sampler: Some(self.sampler),
            render_command: Some(render_command),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Message {
    FocusNext,
    FocusPrevious,
    CycleNext,
    CyclePrev,
    StartTextEdit,
    EndTextEdit,
    CancelTextEdit,
    TextInput(char),
    TextBackspace,
    Quit,
    ConfirmRender,
    DismissError,
}

fn key_to_message(key: KeyEvent, model: &Model) -> Option<Message> {
    if model.editing_text {
        return match key.code {
            KeyCode::Enter => Some(Message::EndTextEdit),
            KeyCode::Esc => Some(Message::CancelTextEdit),
            KeyCode::Backspace => Some(Message::TextBackspace),
            KeyCode::Char(c) => Some(Message::TextInput(c)),
            _ => None,
        };
    }

    if model.error_message.is_some() {
        return match key.code {
            KeyCode::Enter | KeyCode::Esc => Some(Message::DismissError),
            _ => None,
        };
    }

    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => Some(Message::Quit),
        KeyCode::Tab | KeyCode::Down | KeyCode::Char('j') => Some(Message::FocusNext),
        KeyCode::BackTab | KeyCode::Up | KeyCode::Char('k') => Some(Message::FocusPrevious),
        KeyCode::Left | KeyCode::Char('h') => Some(Message::CyclePrev),
        KeyCode::Right | KeyCode::Char('l') => Some(Message::CycleNext),
        KeyCode::Enter => {
            let text_fields = [
                FocusedField::ScenePath,
                FocusedField::OutputPath,
                FocusedField::PixelX,
                FocusedField::PixelY,
                FocusedField::LightSamples,
                FocusedField::NumThreads,
                FocusedField::Spp,
                FocusedField::RayDepth
            ];
            if text_fields.contains(&model.focused_field) {
                Some(Message::StartTextEdit)
            } else {
                None
            }
        }
        KeyCode::Char('r') => Some(Message::ConfirmRender),
        _ => None,
    }
}

fn update(model: &mut Model, msg: Message) {
    match msg {
        Message::FocusNext => model.focus_next(),
        Message::FocusPrevious => model.focus_prev(),

        Message::CycleNext => {
            match model.focused_field {
                FocusedField::SceneInputType => {
                    model.scene_input_type = model.scene_input_type.cycle_next();
                    model.ensure_valid_focus();
                }
                FocusedField::SceneName => {
                    if !model.available_scenes.is_empty() {
                        model.scene_name_index = (model.scene_name_index + 1) % model.available_scenes.len();
                    }
                }
                FocusedField::Backend => {
                    model.backend = model.backend.cycle_next();
                    model.ensure_valid_focus();
                }
                FocusedField::OutputFormat => model.output_format = model.output_format.cycle_next(),
                FocusedField::Sampler => model.sampler = model.sampler.cycle_next(),
                FocusedField::RenderMode => {
                    model.render_mode = model.render_mode.cycle_next();
                    model.ensure_valid_focus();
                }
                _ => {}
            }
        }

        Message::CyclePrev => match model.focused_field {
            FocusedField::SceneInputType => {
                model.scene_input_type = model.scene_input_type.cycle_prev();
                model.ensure_valid_focus();
            }
            FocusedField::SceneName => {
                if !model.available_scenes.is_empty() {
                    let num_scenes = model.available_scenes.len();
                    model.scene_name_index = (model.scene_name_index + num_scenes - 1) % num_scenes;
                }
            }
            FocusedField::Backend => {
                model.backend = model.backend.cycle_prev();
                model.ensure_valid_focus();
            }
            FocusedField::OutputFormat => model.output_format = model.output_format.cycle_prev(),
            FocusedField::Sampler => model.sampler = model.sampler.cycle_prev(),
            FocusedField::RenderMode => {
                model.render_mode = model.render_mode.cycle_prev();
                model.ensure_valid_focus();
            }
            _ => {}
        },

        Message::StartTextEdit => {
            model.editing_text = true;
            model.text_buffer = match model.focused_field {
                FocusedField::ScenePath => model.scene_path.clone(),
                FocusedField::OutputPath => model.output_path.clone(),

                FocusedField::NumThreads => model.num_threads.to_string(),
                FocusedField::RayDepth => model.ray_depth.to_string(),
                FocusedField::Spp => model.spp.to_string(),
                FocusedField::LightSamples => model.light_samples.to_string(),
                FocusedField::PixelX => model.pixel_x.to_string(),
                FocusedField::PixelY => model.pixel_y.to_string(),
                _ => String::new(),
            };
        }

        Message::EndTextEdit => {
            model.editing_text = false;
            match model.focused_field {
                FocusedField::ScenePath => model.scene_path = model.text_buffer.clone(),
                FocusedField::OutputPath => model.output_path = model.text_buffer.clone(),
                
                FocusedField::NumThreads => model.num_threads = model.text_buffer.parse().unwrap_or(1),
                FocusedField::RayDepth => model.ray_depth = model.text_buffer.parse().unwrap_or(1),
                FocusedField::Spp => model.spp = model.text_buffer.parse().unwrap_or(1),
                FocusedField::LightSamples => model.light_samples = model.text_buffer.parse().unwrap_or(1),
                FocusedField::PixelX => model.pixel_x = model.text_buffer.parse().unwrap_or(0),
                FocusedField::PixelY => model.pixel_y = model.text_buffer.parse().unwrap_or(0),
                _ => {}
            }
            model.text_buffer.clear();
        }

        Message::CancelTextEdit => {
            model.editing_text = false;
            model.text_buffer.clear();
        }

        Message::TextInput(c) => {
            if !model.focused_field.numeric_validation() || c.is_numeric() {
                model.text_buffer.push(c);
            }
        }

        Message::TextBackspace => {
            model.text_buffer.pop();
        }

        Message::Quit => {
            model.should_quit = true;
        }

        Message::ConfirmRender => {
            if let Err(e) = model.validate() {
                model.error_message = Some(e);
            } else {
                model.should_render = true;
            }
        }

        Message::DismissError => {
            model.error_message = None;
        }
    }
}

fn view(frame: &mut Frame, model: &Model) {
    let area = frame.area();

    // Main layout: header, form, command preview, help
    let main_layout = Layout::vertical([
        Constraint::Length(3),  // Title
        Constraint::Min(10),    // Form
        Constraint::Length(5),  // Command preview
        Constraint::Length(3),  // Help bar
    ])
    .split(area);

    // Title
    let title = Paragraph::new("🎨 Raytracer Interactive Configuration")
        .style(Style::default().fg(Color::Cyan).bold())
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL));
    frame.render_widget(title, main_layout[0]);

    // Form area split into two columns
    let form_area = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(main_layout[1]);

    render_form_column(frame, model, form_area[0], 0);
    render_form_column(frame, model, form_area[1], 1);

    // Command preview
    let cmd_result = model.generate_command();
    let cmd_style = if cmd_result.is_ok() {
        Style::default().fg(Color::Green)
    } else {
        Style::default().fg(Color::Red)
    };
    let cmd_text = cmd_result.unwrap_or_else(|e| format!("Error: {}", e));
    let command_preview = Paragraph::new(cmd_text)
        .style(cmd_style)
        .wrap(Wrap { trim: false })
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Command Preview"),
        );
    frame.render_widget(command_preview, main_layout[2]);

    // Help bar
    let help_text = if model.editing_text {
        "Enter: Confirm | Esc: Cancel"
    } else {
        "Tab/j/k: Navigate | ←/→/Space: Change | Enter: Edit | r: Render | q/Esc: Quit"
    };
    let help = Paragraph::new(help_text)
        .style(Style::default().fg(Color::DarkGray))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL));
    frame.render_widget(help, main_layout[3]);

    // Error popup
    if let Some(ref error) = model.error_message {
        let popup_area = centered_rect(50, 20, area);
        frame.render_widget(Clear, popup_area);
        let error_popup = Paragraph::new(format!("{}\n\nPress Enter to dismiss", error))
            .style(Style::default().fg(Color::Red))
            .alignment(Alignment::Center)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Error")
                    .border_style(Style::default().fg(Color::Red)),
            );
        frame.render_widget(error_popup, popup_area);
    }
}

fn render_form_column(frame: &mut Frame, model: &Model, area: Rect, column: usize) {
    let fields = if column == 0 {
        model.get_left_fields()
    } else {
        model.get_right_fields()
    };

    let mut items: Vec<ListItem> = Vec::new();

    for field in &fields {
        let is_focused = model.focused_field == *field;
        let prefix = if is_focused { "> " } else { "  " };

        let (label, value) = get_field_display(model, *field);

        let display_value = if model.editing_text && is_focused {
            format!("{}▏", model.text_buffer)
        } else {
            value
        };

        let line = format!("{}{}: {}", prefix, label, display_value);

        let style = if is_focused {
            Style::default().fg(Color::Yellow).bold()
        } else {
            Style::default()
        };

        items.push(ListItem::new(line).style(style));
    }

    let list = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .title(if column == 0 { "Scene & Output" } else { "Settings" }),
    );

    frame.render_widget(list, area);
}

fn get_field_display(model: &Model, field: FocusedField) -> (&'static str, String) {
    match field {
        FocusedField::SceneInputType => (
            "Scene Input",
            match model.scene_input_type {
                SceneInputType::Path => "[File Path]".to_string(),
                SceneInputType::BuiltinName => "[Builtin Scene]".to_string(),
            },
        ),
        FocusedField::ScenePath => ("  Path", model.scene_path.clone()),
        FocusedField::SceneName => (
            "  Scene",
            if model.available_scenes.is_empty() {
                "(none available)".to_string()
            } else {
                format!(
                    "{} ({}/{})",
                    model.available_scenes[model.scene_name_index],
                    model.scene_name_index + 1,
                    model.available_scenes.len()
                )
            },
        ),
        FocusedField::Backend => (
            "Backend",
            match model.backend {
                Backend::Cpu => "CPU (Embree)".to_string(),
                Backend::Optix => "OptiX (NVIDIA)".to_string(),
            },
        ),
        FocusedField::OutputPath => ("Output", model.output_path.clone()),
        FocusedField::OutputFormat => (
            "Format",
            match model.output_format {
                OutputFormat::Png => "PNG".to_string(),
                OutputFormat::Exr => "EXR".to_string(),
            },
        ),
        FocusedField::NumThreads => ("Threads", format!("{}", model.num_threads)),
        FocusedField::RayDepth => ("Ray Depth", format!("{}", model.ray_depth)),
        FocusedField::Spp => ("SPP", format!("{}", model.spp)),
        FocusedField::LightSamples => ("Light Samples", format!("{}", model.light_samples)),
        FocusedField::Sampler => (
            "Sampler",
            match model.sampler {
                SamplerType::Independent => "Independent".to_string(),
                SamplerType::Stratified => "Stratified".to_string(),
            },
        ),
        FocusedField::RenderMode => (
            "Mode",
            match model.render_mode {
                RenderMode::Full => "Full Frame".to_string(),
                RenderMode::Pixel => "Single Pixel".to_string(),
            },
        ),
        FocusedField::PixelX => ("  Pixel X", format!("{}", model.pixel_x)),
        FocusedField::PixelY => ("  Pixel Y", format!("{}", model.pixel_y)),
    }
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::vertical([
        Constraint::Percentage((100 - percent_y) / 2),
        Constraint::Percentage(percent_y),
        Constraint::Percentage((100 - percent_y) / 2),
    ])
    .split(r);

    Layout::horizontal([
        Constraint::Percentage((100 - percent_x) / 2),
        Constraint::Percentage(percent_x),
        Constraint::Percentage((100 - percent_x) / 2),
    ])
    .split(popup_layout[1])[1]
}

pub fn run() -> io::Result<Option<CommandLineArguments>> {
    ratatui::run(|terminal| {
        let mut model = Model::default();

        loop {
            terminal.draw(|frame| view(frame, &model))?;
    
            if event::poll(std::time::Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    if let Some(msg) = key_to_message(key, &model) {
                        update(&mut model, msg);
                    }
                }
            }
    
            if model.should_quit {
                return Ok(None);
            }
    
            if model.should_render {
                return Ok(Some(model.into_args()));
            }
        }
    })
}
