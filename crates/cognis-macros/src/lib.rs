//! Proc macros for the Cognis framework.
//!
//! # Macros
//!
//! - [`tool`] — `#[cognis::tool]` attribute that generates a
//!   `cognis_core::tools::BaseTool` implementation from an `async fn` (or
//!   from an `impl` block containing one). Uses [`schemars`] under the hood
//!   for parameter schema generation.
//! - [`GraphState`] — derive macro for graph state schemas with per-field
//!   reducers (see attributes `#[reducer(append|last_value|add|merge)]`).
//!
//! # JSON Schema generation
//!
//! For `#[derive(JsonSchema)]` on your own structs/enums, use the re-export
//! at `cognis_core::JsonSchema` (which is `schemars::JsonSchema`). The legacy
//! hand-rolled `JsonSchema` / `ToolSchema` derives in this crate were removed
//! in favor of the upstream `schemars` crate, which supports recursive types,
//! `$ref`/`definitions`, doc-comment → `description`, and richer attribute
//! syntax.
//!
//! ```ignore
//! use cognis_core::JsonSchema;
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(JsonSchema, Serialize, Deserialize)]
//! struct SearchFilter {
//!     /// Minimum relevance score
//!     min_score: f64,
//!     /// Categories to include
//!     categories: Vec<String>,
//! }
//!
//! let schema = serde_json::to_value(schemars::schema_for!(SearchFilter)).unwrap();
//! ```

mod graph_state;
mod graph_state_v2;
mod schema_attr;
mod tool_attr;
mod tools_impl_attr;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use syn::{parse_macro_input, DeriveInput};

// ---------------------------------------------------------------------------
// #[derive(GraphState)]
// ---------------------------------------------------------------------------

/// Derive macro for generating graph state schemas with per-field reducers.
///
/// # Attributes
///
/// - `#[reducer(append)]` — append arrays/values
/// - `#[reducer(last_value)]` — overwrite with latest (default)
/// - `#[reducer(add)]` — add numeric values
/// - `#[reducer(merge)]` — deep-merge JSON objects
#[proc_macro_derive(GraphState, attributes(reducer))]
pub fn derive_graph_state(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    graph_state::derive_graph_state(input).into()
}

// ---------------------------------------------------------------------------
// #[cognis::tool] attribute macro
// ---------------------------------------------------------------------------

/// `#[cognis::tool]` — attribute macro that generates a
/// [`cognis_core::tools::BaseTool`] implementation from an `async fn`.
///
/// Applies to either a standalone `async fn` or an `impl` block that
/// contains exactly one `async fn` (the stateful form, where the tool
/// struct holds configuration such as HTTP clients or API keys).
///
/// # Arguments
///
/// - `name = "..."` — overrides the tool name (defaults to the fn name).
/// - `description = "..."` — overrides the fn's doc-comment description.
/// - `return_direct = true|false` — passes through to `BaseTool::return_direct`.
///
/// # Field validators
///
/// `#[schema(range(...))]`, `#[schema(length(...))]`, `#[schema(pattern(...))]`,
/// `#[schema(enum_values(...))]`, and `#[schema(format(...))]` on fn arguments
/// emit matching runtime checks. Range/length/pattern are also translated into
/// `#[schemars(...)]` attributes on the generated args struct so they appear
/// in the JSON Schema produced via `schemars::schema_for!`. Enum-values and
/// format are merged into the schema during `args_schema()` post-processing.
///
/// See `cognis_core::tools::validation` for the runtime helpers the
/// generated code invokes.
#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as tool_attr::ToolArgs);
    let item_ts: TokenStream2 = item.into();
    match tool_attr::expand(args, item_ts) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// `#[derive(GraphStateV2)]` — v2-shape state derive that emits a typed
/// sibling `<Name>Update` struct and an `impl GraphState for <Name>`. Use
/// in v2 code via the re-export `cognis2_core::GraphState` (the rename
/// happens in cognis2-core's lib.rs in Plan #2).
#[proc_macro_derive(GraphStateV2, attributes(reducer, graph_state))]
pub fn derive_graph_state_v2(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    graph_state_v2::derive_graph_state_v2(input).into()
}

/// `#[tools_impl]` — outer attribute that scans an `impl` block for inner
/// `#[tool]`-marked async methods and generates one [`cognis_core::tools::BaseTool`]
/// wrapper per method, plus an `into_tools()` collector method on the user's
/// struct.
///
/// # Arguments
///
/// - `crate_path = "..."` — override the framework crate (default: `"cognis_core"`).
///
/// Inner `#[tool]` markers on methods follow the same syntax as the standalone
/// `#[tool]` attribute (name, description, return_direct, crate_path).
#[proc_macro_attribute]
pub fn tools_impl(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as tools_impl_attr::ToolsImplArgs);
    let item_ts: TokenStream2 = item.into();
    match tools_impl_attr::expand(args, item_ts) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}
