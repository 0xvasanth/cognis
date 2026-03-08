//! Centralized callback system for the entire execution lifecycle.
//!
//! This module provides a phase-based callback manager that covers chains,
//! LLMs, tools, and retrievers. It complements the chain-specific callbacks
//! in [`crate::chains::callbacks`] by offering a unified event bus that spans
//! every layer of the framework.
//!
//! # Key Types
//!
//! - [`CallbackPhase`] -- Identifies the lifecycle phase of an event.
//! - [`CallbackData`] -- Structured event payload with phase, run id, and data.
//! - [`CallbackHandler`] -- Trait for consuming callback events.
//! - [`ConsoleCallbackHandler`] -- Buffers human-readable log lines.
//! - [`MetricsCallbackHandler`] -- Collects timing and count metrics.
//! - [`CallbackManager`] -- Dispatches events to registered handlers.
//! - [`CallbackScope`] -- RAII guard that emits start/end events automatically.
//! - [`FilteredHandler`] -- Wraps a handler with phase-based filtering.

pub mod manager;

pub use manager::*;
