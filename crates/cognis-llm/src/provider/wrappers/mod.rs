//! Provider wrappers — composable [`super::LLMProvider`] implementations
//! that wrap an inner provider with additional behavior.
//!
//! Each wrapper itself implements `LLMProvider`, so they compose with
//! each other and with `Client`. Order matters — the outermost wrapper
//! sees requests first and responses last.
//!
//! The wrappers in this module:
//!
//! - [`circuit_breaker::CircuitBreakerProvider`] — opens a circuit after
//!   N consecutive failures, half-opens after a cooldown.
//! - [`load_balancer::LoadBalancerProvider`] — distributes calls across
//!   a fleet of inner providers with a pluggable strategy.
//! - [`routing::RoutingProvider`] — dispatches to one of N providers
//!   based on a user-supplied predicate.
//! - [`graceful::GracefulDegradationProvider`] — drops unsupported
//!   features (tools, streaming) rather than erroring.
//! - [`interceptor::InterceptorProvider`] — chat-shape before/after/error
//!   hooks for request rewriting and response transformation.
//!
//! Customization:
//! - Implement [`super::LLMProvider`] directly for a fully custom wrapper.
//! - Most wrappers expose pluggable strategy traits ([`load_balancer::LoadBalancingStrategy`],
//!   [`circuit_breaker::FailureClassifier`], [`interceptor::ChatInterceptor`])
//!   so the common knobs are swappable without subclassing.

pub mod circuit_breaker;
pub mod graceful;
pub mod interceptor;
pub mod load_balancer;
pub mod routing;

pub use circuit_breaker::{
    CircuitBreakerProvider, CircuitState, CircuitStats, FailureClassifier, RetryableClassifier,
};
pub use graceful::{Capability, GracefulDegradationProvider};
pub use interceptor::{ChatInterceptor, FnChatInterceptor, InterceptorProvider};
pub use load_balancer::{
    LoadBalancerProvider, LoadBalancingStrategy, RandomStrategy, RoundRobinStrategy,
    WeightedRoundRobinStrategy,
};
pub use routing::{ProviderRoute, RoutingProvider, RoutingStrategy};
