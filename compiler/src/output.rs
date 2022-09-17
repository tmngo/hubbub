use std::string::String;

pub trait Color {
    fn red(self) -> String;
}

impl Color for &str {
    fn red(self) -> String {
        self.to_string()
    }
}

#[macro_export]
macro_rules! BRIGHT_RED {
    () => {
        "\x1b[91m"
    };
}

#[macro_export]
macro_rules! RESET {
    () => {
        "\x1b[0m"
    };
}

#[macro_export]
macro_rules! format_red {
    ($fmt_str:literal) => {{
        format!(concat!($crate::BRIGHT_RED!(), $fmt_str, $crate::RESET!()))
    }};
    ($fmt_str:literal, $($args:expr),*) => {{
        format!(concat!($crate::BRIGHT_RED!(), $fmt_str, $crate::RESET!()), $($args),*)
    }};
}
