//! # cognis2-core
//!
//! v2-beta foundation: typed `Runnable<I, O>` trait + supporting primitives.

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod error;
pub mod extensions;
pub mod message;
pub mod stream;
// Filled in by Tasks 6-7:
// pub mod runnable;

pub use error::{CognisError, Result};
pub use extensions::Extensions;
pub use message::{AiMessage, HumanMessage, Message, SystemMessage, ToolCall, ToolMessage};
pub use stream::{Event, EventStream, Observer};

/// Re-export of the [`schemars`] crate. v2 user code uses
/// `cognis2_core::schemars::JsonSchema` for derive-driven schema generation.
pub use schemars;
pub use schemars::{schema_for, JsonSchema};
