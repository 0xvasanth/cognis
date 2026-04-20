//! `#[tool]` with a reference-typed arg — must fail because the macro
//! deserializes owned types via `serde_json::from_value`.

use cognis_core::tool;

#[tool]
async fn borrow_arg(_q: &str) -> cognis_core::error::Result<cognis_core::tools::ToolOutput> {
    Ok(cognis_core::tools::ToolOutput::Content(
        serde_json::json!(null),
    ))
}

fn main() {}
