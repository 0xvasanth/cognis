//! # cognis-core
//!
//! v2-beta foundation: typed `Runnable<I, O>` trait + supporting primitives.
//!
//! Top-level modules:
//! - [`runnable`] — the `Runnable<I, O>` trait and `RunnableConfig`.
//! - [`runnable_ext`] — fluent `.pipe()`, `.with_retry()`, etc. on any runnable.
//! - [`message`] — typed chat messages.
//! - [`stream`] — `Event` taxonomy, `Observer`, `RunnableStream`, `EventStream`.
//! - [`error`] — `CognisError`, `Result`, `InterruptKind`.
//! - [`extensions`] — typed key-value bag (`tower::Extensions`-style).
//! - [`prompts`] — typed prompt templates.
//! - [`output_parsers`] — typed parsers from `String` to a target type.
//! - [`compose`] — `Pipe`, `Parallel`, `Branch`, `Lambda`, `Each`, `Passthrough`.
//! - [`wrappers`] — `Retry`, `Fallback`, `Timeout`, `Cache`.

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod callbacks;
pub mod compose;
pub mod content;
pub mod error;
pub mod extensions;
pub mod json_merge;
pub mod message;
pub mod output_parsers;
pub mod prompts;
pub mod runnable;
pub mod runnable_ext;
pub mod security;
pub mod serialization;
pub mod stream;
pub mod tokenizer;
pub mod wrappers;

pub use callbacks::{
    BuiltHandler, CallbackHandler, CallbackManager, HandlerBuilder, HandlerObserver,
};
pub use content::{
    base64_decode, base64_encode, image_source_from_path, mime_from_path, AudioSource, ContentPart,
    ImageSource,
};
pub use error::{CognisError, InterruptKind, Result};
pub use extensions::Extensions;
pub use json_merge::deep_merge_json;
pub use message::{
    merge_message_runs, message_from_chunks, trim_messages, trim_messages_custom, AiChunk,
    AiMessage, HumanChunk, HumanMessage, Message, MessageChunk, RemoveMessage, SystemChunk,
    SystemMessage, ToolCall, ToolCallChunk, ToolChunk, ToolMessage, TrimStrategy,
};
pub use runnable::{Runnable, RunnableConfig};
pub use runnable_ext::RunnableExt;
pub use security::is_public_unicast;
pub use serialization::{Loader, RunnableDefinition, Serializable};
pub use stream::{Event, EventStream, Observer, RunnableStream};
pub use tokenizer::{CharTokenizer, FnTokenizer, Tokenizer};

/// Re-export of the [`schemars`] crate. v2 user code uses
/// `cognis_core::schemars::JsonSchema` for derive-driven schema generation.
pub use schemars;
pub use schemars::{schema_for, JsonSchema};

/// Common imports for v2 user code.
pub mod prelude {
    pub use crate::compose::{lambda, pipe, Branch, Each, Lambda, Parallel, Passthrough, Pipe};
    pub use crate::output_parsers::{
        BooleanParser, CommaListParser, JsonExtraction, JsonExtractor, JsonParser,
        NumberedListParser, OutputParser, StringParser, StructuredOutputConfig,
        StructuredOutputParser, XmlParser,
    };
    pub use crate::prompts::{ChatPromptTemplate, FewShotTemplate, PromptTemplate, Role};
    pub use crate::schema_for;
    pub use crate::wrappers::{
        Assign, AssignOutput, Bind, Cache, CacheBackend, ConfigKey, Configurable, Fallback,
        MemoryCache, Retry, RetryPolicy, Timeout,
    };
    pub use crate::{
        AiMessage, CognisError, Event, EventStream, Extensions, HumanMessage, InterruptKind,
        JsonSchema, Message, Observer, Result, Runnable, RunnableConfig, RunnableExt,
        RunnableStream, SystemMessage, ToolCall, ToolMessage,
    };
    pub use async_trait::async_trait;
}
