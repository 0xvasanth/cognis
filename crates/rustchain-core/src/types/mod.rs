//! Type system utilities for the RustChain framework.
//!
//! Provides dynamic values, type coercion, validation schemas, and a type registry
//! for safely converting between value representations used across the pipeline.

pub mod coercion;

pub use coercion::*;
