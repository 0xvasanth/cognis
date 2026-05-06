//! Error module re-exported from cognis2-core.
//!
//! Macro-generated code targeting `crate_path = "cognis2_llm"` emits paths
//! like `::cognis2_llm::error::Result` and
//! `::cognis2_llm::error::CognisError::ToolValidationError(...)`.
//!
//! cognis2-core's `CognisError` uses the variant name `ToolValidation`; the
//! v1 macros emit `ToolValidationError`. We bridge this by re-exporting
//! cognis2-core's error types and adding a compatibility shim.

pub use cognis2_core::error::{CognisError, Result};
