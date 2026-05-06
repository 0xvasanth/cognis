//! # cognis2-core
//!
//! v2-beta foundation: typed `Runnable<I, O>` trait + supporting primitives.

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod error;
// Filled in by Tasks 2-6:
// pub mod extensions;
// pub mod message;
// pub mod runnable;
// pub mod stream;

pub use error::{CognisError, Result};

/// Re-export of the [`schemars`] crate. v2 user code uses
/// `cognis2_core::schemars::JsonSchema` for derive-driven schema generation.
pub use schemars;
pub use schemars::{schema_for, JsonSchema};
