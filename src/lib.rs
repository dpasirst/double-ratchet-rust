//! Crate documentation provided in README.md

// TODO: include README.md documentation: https://github.com/rust-lang/rust/issues/44732
// TODO: test examples in README.md

#![no_std]
#![warn(clippy::pedantic)]
#![warn(missing_docs)]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

/// synchronous runtime implementation
pub mod sync;

#[cfg(not(feature = "std"))]
use core::{error::Error, fmt};
#[cfg(feature = "std")]
use std::{error::Error, fmt};

/// General Errors
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DRError {
    /// Data is invalid or cannot be processed
    InvalidData,
    /// Key is invalid or cannot be processed
    InvalidKey,
}

#[cfg(feature = "std")]
impl Error for DRError {}

impl fmt::Display for DRError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use DRError::{InvalidData, InvalidKey};
        match self {
            InvalidData => write!(f, "Data is invalid or cannot be processed"),
            InvalidKey => write!(f, "Key is invalid or cannot be processed"),
        }
    }
}