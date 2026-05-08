//! `#[tool]` applied to a sync fn — must fail with a clear error.

use cognis_core::tool;

#[tool]
fn not_async(_q: String) -> cognis_core::error::Result<cognis_core::tools::ToolOutput> {
    Ok(cognis_core::tools::ToolOutput::Content(
        serde_json::json!(null),
    ))
}

fn main() {}
