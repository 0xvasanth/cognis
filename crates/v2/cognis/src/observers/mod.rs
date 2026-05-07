//! `Observer` implementations that bridge cognis2's [`Event`] stream to
//! the surrounding ecosystem.
//!
//! V1 shipped bespoke LangSmith / OTel tracers. V2 takes the Rust-native
//! route: emit to the [`tracing`] crate. Then any `tracing-subscriber`
//! plugin — `tracing-opentelemetry`, `tracing-bunyan-formatter`,
//! `tracing-loki`, plain stdout — gets your trace stream for free.

pub mod tracing_observer;

pub use tracing_observer::TracingObserver;
