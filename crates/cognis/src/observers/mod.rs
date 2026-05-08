//! `Observer` implementations that bridge cognis's [`Event`] stream to
//! the surrounding ecosystem.
//!
//! V2 takes the Rust-native route: emit to the [`tracing`] crate. Any
//! `tracing-subscriber` plugin — `tracing-opentelemetry`,
//! `tracing-bunyan-formatter`, `tracing-loki`, plain stdout — picks up
//! the trace stream for free.

pub mod tracing_observer;

pub use tracing_observer::TracingObserver;
