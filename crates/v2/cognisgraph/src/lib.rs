//! # cognis2-graph
//!
//! v2-beta graph engine: typed `Graph<S>` + Pregel superstep executor.
//! `CompiledGraph<S>` implements `cognis2_core::Runnable<S, S>`.

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod state;
// pub mod reducer;
// pub mod node;
// pub mod goto;
// pub mod builder;
// pub mod validate;
// pub mod compiled;
// pub mod engine;
// pub mod checkpoint;

pub use state::GraphState;

/// Derive macro — generates `impl GraphState for <T>` with per-field reducers.
/// The derive name shadows the trait name; both are imported via
/// `use cognis2_graph::GraphState;` because Rust allows a trait and a
/// derive macro to share the same identifier (they live in different namespaces).
pub use cognis_macros::GraphStateV2 as GraphState;

/// Re-export of [`cognis2_core`] — graph users import from this crate
/// and `cognis2_core::*` is implicitly in scope via re-export.
pub use cognis2_core;

/// Re-export of the [`schemars`] crate (via cognis2-core).
pub use cognis2_core::schemars;
