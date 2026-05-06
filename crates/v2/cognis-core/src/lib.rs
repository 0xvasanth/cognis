//! # cognis2-core
//!
//! v2-beta foundation: typed `Runnable<I, O>` trait + supporting primitives.

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod error;
pub mod extensions;
pub mod message;
pub mod runnable;
pub mod stream;

pub use error::{CognisError, Result};
pub use extensions::Extensions;
pub use message::{AiMessage, HumanMessage, Message, SystemMessage, ToolCall, ToolMessage};
pub use runnable::{Runnable, RunnableConfig};
pub use stream::{Event, EventStream, Observer, RunnableStream};

/// Re-export of the [`schemars`] crate. v2 user code uses
/// `cognis2_core::schemars::JsonSchema` for derive-driven schema generation.
pub use schemars;
pub use schemars::{schema_for, JsonSchema};

/// Common imports for v2 user code.
pub mod prelude {
    pub use crate::{
        AiMessage, CognisError, Event, EventStream, Extensions, HumanMessage, JsonSchema, Message,
        Observer, Result, Runnable, RunnableConfig, RunnableStream, SystemMessage, ToolCall,
        ToolMessage,
    };
    pub use crate::schema_for;
    pub use async_trait::async_trait;
}
