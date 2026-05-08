//! Typed args via SchemaBasedTool — no manual schema authoring.

use async_trait::async_trait;
use cognis::prelude::*;
use cognis_core::schemars::{self, JsonSchema};
use cognis_llm::tools::{SchemaBasedTool, Tool};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct SearchArgs {
    query: String,
    limit: Option<u32>,
}

struct SearchKb;

#[async_trait]
impl SchemaBasedTool for SearchKb {
    type Params = SearchArgs;
    type Output = Value;
    fn name(&self) -> &str {
        "search_kb"
    }
    fn description(&self) -> &str {
        "Search a mock knowledge base."
    }
    async fn execute_typed(&self, args: SearchArgs) -> Result<Value> {
        let limit = args.limit.unwrap_or(3).min(50);
        let items: Vec<_> = (1..=limit)
            .map(|i| json!({"rank": i, "title": format!("Doc {i} for {}", args.query)}))
            .collect();
        Ok(json!({"results": items}))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let t = SearchKb;
    println!("schema:\n{:#}", Tool::args_schema(&t).unwrap());
    let out = t
        .execute_typed(SearchArgs {
            query: "rust async".into(),
            limit: Some(2),
        })
        .await?;
    println!("output: {out}");
    Ok(())
}
