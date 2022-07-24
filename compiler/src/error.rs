//! error/mod.rs - Defines the error, warning, and note macros
//! used to issue compiler errors. There is also an ErrorMessage type
//! for storing messages that may be issued later. Note that all issuing
//! an error does is print it to stderr and update the global ERROR_COUNT.
//!
//! Compiler passes are expected to continue even after issuing errors so
//! that as many can be issued as possible. A possible future improvement
//! would be to implement poisoning so that repeated errors are hidden.

use colored::ColoredString;
use colored::*;
use std::fmt::{Display, Formatter};

/// Return an error which may be issued later
macro_rules! make_error {
    ( $fmt_string:expr $( , $($msg:tt)* )? ) => ({
        let message = format!($fmt_string $( , $($msg)* )? );
        $crate::error::ErrorMessage::error(&message[..])
    });
}

/// Issue an error message to stderr and increment the error count
macro_rules! error {
    ( $fmt_string:expr $( , $($msg:tt)* )? ) => {{
        eprintln!("{}", $crate::error::make_error!($fmt_string $( , $($msg)* )?));
        panic!($fmt_string $( , $($msg)* )? );
        // std::process::exit(1);
    }};
}

pub(crate) use error;
pub(crate) use make_error;

/// Return a warning which may be issued later
macro_rules! make_warning {
    ( $fmt_string:expr $( , $($msg:tt)* )? ) => ({
        let message = format!($fmt_string $( , $($msg)* )? );
        $crate::error::ErrorMessage::warning(&message[..])
    });
}

/// Issues a warning to stderr
macro_rules! warning {
    ( $fmt_string:expr $( , $($msg:tt)* )? ) => ({
        eprintln!("{}", make_warning!($fmt_string $( , $($msg)* )?));
    });
}

/// Return a note which may be issued later
macro_rules! make_note {
    ( $fmt_string:expr $( , $($msg:tt)* )? ) => ({
        let message = format!($fmt_string $( , $($msg)* )? );
        $crate::error::ErrorMessage::note(&message[..])
    });
}

/// Issues a note to stderr
macro_rules! note {
    ( $fmt_string:expr $( , $($msg:tt)* )? ) => ({
        eprintln!("{}", make_note!($location, $fmt_string $( , $($msg)* )?));
    });
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorType {
    Error,
    Warning,
    Note,
}

/// An error (or warning/note) message to be printed out on screen.
#[derive(Debug, PartialEq, Eq)]
pub struct ErrorMessage {
    msg: ColoredString,
    error_type: ErrorType,
}

impl ErrorMessage {
    pub fn error<T: Into<ColoredString>>(msg: T) -> ErrorMessage {
        ErrorMessage {
            msg: msg.into(),
            error_type: ErrorType::Error,
        }
    }

    pub fn warning<T: Into<ColoredString>>(msg: T) -> ErrorMessage {
        ErrorMessage {
            msg: msg.into(),
            error_type: ErrorType::Warning,
        }
    }

    pub fn note<T: Into<ColoredString>>(msg: T) -> ErrorMessage {
        ErrorMessage {
            msg: msg.into(),
            error_type: ErrorType::Note,
        }
    }

    fn marker(&self) -> ColoredString {
        match self.error_type {
            ErrorType::Error => self.color("error:"),
            ErrorType::Warning => self.color("warning:"),
            ErrorType::Note => self.color("note:"),
        }
    }

    /// Color the given string in either the error, warning, or note color
    fn color(&self, msg: &str) -> ColoredString {
        match self.error_type {
            ErrorType::Error => msg.red(),
            ErrorType::Warning => msg.yellow(),
            ErrorType::Note => msg.purple(),
        }
    }
}

impl<'a> Display for ErrorMessage {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        writeln!(f, "{} {}", self.marker(), self.msg)?;

        Ok(())
    }
}
