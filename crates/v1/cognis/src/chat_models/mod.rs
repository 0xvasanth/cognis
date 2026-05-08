//! Chat model implementations, wrappers, and provider registry.
//!
//! Includes provider-specific implementations (Anthropic, OpenAI, Google, Ollama,
//! Azure) gated behind feature flags, plus composable wrappers (cached, circuit
//! breaker, rate limited, retrying, structured, token counting, graceful,
//! interceptor).
//!
//! The [`factory`] module provides [`ChatModelFactory`](factory::ChatModelFactory)
//! for dynamically creating chat model instances by provider name, and a global
//! [`ModelRegistry`](factory::ModelRegistry) singleton for application-wide
//! provider registration. Use [`init_chat_model`](factory::init_chat_model) for
//! one-line model creation.

#[cfg(feature = "anthropic")]
pub mod anthropic;
#[cfg(feature = "azure")]
pub mod azure;
pub mod base;
pub mod cached;
pub mod circuit_breaker;
#[cfg(feature = "google")]
pub mod google;
pub mod graceful;
pub mod interceptor;
pub mod load_balancer;
#[cfg(feature = "ollama")]
pub mod ollama;
#[cfg(feature = "openai")]
pub mod openai;
#[cfg(feature = "openai")]
pub mod openrouter;
pub mod rate_limited;
pub mod retrying;
pub mod routing;
pub mod structured;
pub mod token_counting;

pub mod factory;
pub mod registry;

pub use base::*;
pub use structured::*;
