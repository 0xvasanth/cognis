//! # cognis-llm
//!
//! v2-beta LLM client. Re-exports `error`, `schemars`, and `Message` from
//! cognis-core so v2 user code can target a single `crate_path =
//! "cognis_llm"` and have all macro-generated paths resolve.

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

// Re-export the cognis-core error module so macro-generated code that
// targets `cognis_llm::error::*` resolves transparently.
pub mod error;

// Re-export schemars for the same reason.
pub use cognis_core::schema_for;
pub use cognis_core::schemars;
pub use cognis_core::schemars::JsonSchema;

// Re-export Message + variants since they're the LLM-conversation primitive.
pub use cognis_core::message;
pub use cognis_core::{AiMessage, HumanMessage, Message, SystemMessage, ToolCall, ToolMessage};

pub mod schema;
pub use schema::schema_for_tool;

pub mod chat;
pub use chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk, ToolCallDelta, Usage};

pub mod tools;
pub use tools::{
    BaseTool, SchemaBasedTool, Tool, ToolDefinition, ToolInput, ToolOutput, ToolRegistry,
};
pub mod client;
pub use client::{Client, ClientBuilder};

pub mod factory;
pub use factory::{ProviderConstructor, ProviderRegistry, ProviderSpec};

pub mod provider;
#[cfg(feature = "openai")]
pub use provider::openrouter::{OpenRouterBuilder, OpenRouterProvider};
pub use provider::wrappers::{
    Capability, ChatInterceptor, CircuitBreakerProvider, CircuitState, CircuitStats,
    FailureClassifier, FnChatInterceptor, GracefulDegradationProvider, InterceptorProvider,
    LoadBalancerProvider, LoadBalancingStrategy, ProviderRoute, RandomStrategy,
    RetryableClassifier, RoundRobinStrategy, RoutingProvider, RoutingStrategy,
    WeightedRoundRobinStrategy,
};
pub use provider::{LLMProvider, Provider};

pub mod streaming;
pub use streaming::{Aggregated, StreamAggregator};

pub mod usage;
pub use usage::UsageTracker;

pub mod structured;
pub use structured::StructuredClient;

/// Common imports for v2 user code building agents and tools.
pub mod prelude {
    pub use crate::schema_for;
    pub use crate::{
        AiMessage, HumanMessage, JsonSchema, Message, SystemMessage, ToolCall, ToolMessage,
    };
    pub use cognis_core::prelude::*;
}
