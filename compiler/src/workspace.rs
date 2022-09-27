use codespan_reporting::{
    diagnostic::Diagnostic,
    files::SimpleFiles,
    term::{
        emit,
        termcolor::{ColorChoice, StandardStream},
        DisplayStyle,
    },
};

pub struct Workspace {
    pub files: SimpleFiles<String, String>,
    pub diagnostics: Vec<Diagnostic<usize>>,
}

pub type Result<T> = core::result::Result<T, Diagnostic<usize>>;

impl Workspace {
    pub fn new() -> Self {
        Self {
            files: SimpleFiles::new(),
            diagnostics: vec![],
        }
    }

    pub fn has_errors(&self) -> bool {
        !self.diagnostics.is_empty()
    }

    pub fn print_errors(&self) {
        let writer = StandardStream::stderr(ColorChoice::Always);
        let config = codespan_reporting::term::Config {
            display_style: DisplayStyle::Rich,
            ..Default::default()
        };
        for diagnostic in &self.diagnostics {
            emit(&mut writer.lock(), &config, &self.files, diagnostic)
                .expect("failed to emit diagnostics");
        }
    }
}

impl Default for Workspace {
    fn default() -> Self {
        Self::new()
    }
}
