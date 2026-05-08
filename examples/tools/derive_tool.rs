//! `#[derive(JsonSchema)]` to auto-generate parameter schemas, then plug
//! them into a hand-written `Tool` implementation. The schema is what
//! the LLM sees when picking arguments.

use async_trait::async_trait;
use cognis::prelude::*;
use cognis_core::schemars::{self, schema_for, JsonSchema};
use cognis_llm::tools::{Tool, ToolInput, ToolOutput};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct SearchArgs {
    /// The search query.
    query: String,
    /// Max results (default 5).
    #[serde(default)]
    limit: Option<u32>,
}

struct SearchTool;

#[async_trait]
impl Tool for SearchTool {
    fn name(&self) -> &str {
        "search"
    }
    fn description(&self) -> &str {
        "Searches a mock knowledge base."
    }
    fn args_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::to_value(schema_for!(SearchArgs)).unwrap())
    }
    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let args: SearchArgs = serde_json::from_value(input.into_json())?;
        let limit = args.limit.unwrap_or(5);
        let hits: Vec<_> = (1..=limit)
            .map(|i| json!({"rank": i, "title": format!("Result {i} for {}", args.query)}))
            .collect();
        Ok(ToolOutput::Content(json!({"results": hits})))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let tool = SearchTool;
    println!("name: {}", tool.name());
    println!("description: {}", tool.description());
    println!("schema:\n{:#}", tool.args_schema().unwrap());
    let mut args = std::collections::HashMap::new();
    args.insert("query".into(), json!("rust async"));
    args.insert("limit".into(), json!(2));
    let out = tool._run(ToolInput::Structured(args)).await?;
    println!("output: {out:?}");
    Ok(())
}
