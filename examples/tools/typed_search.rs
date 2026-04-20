//! `#[cognis::tool]` on a standalone `async fn`, using schema
//! validators on the fn parameters.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example tool_typed_search -p cognis-examples
//! ```

use cognis_core::error::Result;
use cognis_core::tool;
use cognis_core::tools::{BaseTool, ToolInput, ToolOutput};
use serde_json::json;
use std::collections::HashMap;

/// Search a mock knowledge base. The generated schema surfaces `query`,
/// `limit`, and their constraints to any LLM that consults it.
#[tool(name = "search_kb")]
async fn search_kb(
    /// The search query — must be 1-200 chars.
    #[schema(length(min = 1, max = 200))]
    query: String,
    /// Max results to return (1-50).
    #[schema(range(min = 1, max = 50))]
    limit: Option<u32>,
) -> Result<ToolOutput> {
    let limit = limit.unwrap_or(3);
    let items: Vec<_> = (1..=limit)
        .map(|i| {
            json!({
                "id": i,
                "snippet": format!("Doc #{i} mentions `{query}`"),
            })
        })
        .collect();
    Ok(ToolOutput::Content(
        json!({ "query": query, "results": items }),
    ))
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let tool = SearchKb;

    println!("tool name: {}", tool.name());
    println!("tool description: {}", tool.description());
    println!(
        "schema:\n{}",
        serde_json::to_string_pretty(&tool.args_schema().unwrap())?
    );

    let mut args = HashMap::new();
    args.insert("query".to_string(), json!("typed tool inputs"));
    args.insert("limit".to_string(), json!(2));

    let out = tool._run(ToolInput::Structured(args)).await?;
    if let ToolOutput::Content(v) = out {
        println!("\nresult:\n{}", serde_json::to_string_pretty(&v)?);
    }

    // Deliberately trip a validator to show the error path.
    let mut bad = HashMap::new();
    bad.insert("query".to_string(), json!("x"));
    bad.insert("limit".to_string(), json!(100));
    match tool._run(ToolInput::Structured(bad)).await {
        Ok(_) => unreachable!("validator should reject limit=100"),
        Err(e) => println!("\nexpected validation error: {e}"),
    }

    Ok(())
}
