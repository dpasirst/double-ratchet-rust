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

/// async runtime implementation
#[cfg(feature = "async")]
pub mod async_;


#[cfg(not(feature = "std"))]
use core::{error::Error, fmt};
#[cfg(feature = "std")]
use std::{error::Error, fmt};


/// Upper limit on the receive chain ratchet steps when trying to decrypt. Prevents a
/// denial-of-service attack where the attacker attempts to drive up the number of
/// message keys that must be stored tied to a single public key
pub const DEFAULT_MAX_SKIP: usize = 1000;

/// Maximum amount of skipped message keys that can be stored per double ratchet id.
/// The id represents a 1:1 (alice to bob, where same alice to different bob would
/// be a different id.) Prevents a denial-of-service attack where the attacker 
/// attempts drive up the number of message keys that must be stored by rotating 
/// public keys. Thus the key cache would store a maximum number of entries per 
/// across multiple chains.
pub const DEFAULT_MKS_CAPACITY: usize = 3000;


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

/// Error that occurs on `try_ratchet_encrypt` before the state is initialized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncryptUninit;

#[cfg(feature = "std")]
impl Error for EncryptUninit {}

impl fmt::Display for EncryptUninit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Encrypt not yet initialized (you must receive a message first)"
        )
    }
}

/// Error that may occur during `ratchet_decrypt`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecryptError {
    /// Could not verify-decrypt the ciphertext + associated data + header
    DecryptFailure,

    /// Could not find the message key required for decryption
    ///
    /// Note that this implementation is not always able to detect when an old `MessageKey` can't
    /// be found: a `DecryptFailure` may be triggered instead.
    MessageKeyNotFound,

    /// Header message counter is too large (either `n` or `pn`)
    SkipTooLarge,

    /// Storage of skipped message keys is full
    StorageFull,
}

#[cfg(feature = "std")]
impl Error for DecryptError {}

impl fmt::Display for DecryptError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use DecryptError::{DecryptFailure, MessageKeyNotFound, SkipTooLarge, StorageFull};
        match self {
            DecryptFailure => write!(f, "Error during verify-decrypting"),
            MessageKeyNotFound => {
                write!(f, "Could not find the message key required for decryption")
            }
            SkipTooLarge => write!(f, "Header message counter is too large"),
            StorageFull => write!(f, "Storage for skipped messages is full"),
        }
    }
}