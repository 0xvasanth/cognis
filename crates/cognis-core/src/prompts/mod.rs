//! Typed prompt templates.
//!
//! [`PromptTemplate<I>`] renders a string prompt from any `Serialize` input.
//! [`ChatPromptTemplate<I>`] renders a `Vec<Message>`. [`FewShotTemplate`]
//! interpolates a per-example template into a prefix/suffix wrapper.
//!
//! Templates use `{name}` for substitution. Use `{{` and `}}` for literal
//! braces.
//!
//! ```no_run
//! use cognis_core::prompts::{ChatPromptTemplate, Role};
//! use serde::Serialize;
//!
//! #[derive(Serialize)]
//! struct Input { topic: String }
//!
//! let prompt: ChatPromptTemplate<Input> = ChatPromptTemplate::new()
//!     .system("You are a {topic} expert.")
//!     .human("Write a haiku about {topic}.");
//! ```

pub mod chat;
pub mod example_selector;
pub mod few_shot;
pub mod template;

pub use chat::{ChatPromptTemplate, Role};
pub use example_selector::{
    ExampleRenderFn, ExampleSelector, LengthBasedExampleSelector, StaticExampleSelector,
};
pub use few_shot::FewShotTemplate;
pub use template::PromptTemplate;
