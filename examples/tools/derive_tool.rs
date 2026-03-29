//! Derive Tool Example
//!
//! Demonstrates `#[derive(ToolSchema)]` and `#[derive(JsonSchema)]` for
//! auto-generating OpenAPI-compatible JSON schemas from Rust structs,
//! and how to combine them with `BaseTool` to build tools with rich schemas.
//!
//! Run with: `cargo run -p cognis-examples --example derive_tool`

use async_trait::async_trait;
use cognis_core::error::Result;
use cognis_core::tools::types::{ToolInput, ToolOutput};
use cognis_core::tools::BaseTool;
use cognis_macros::{JsonSchema, ToolSchema};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

// ─── Schema-only structs (for nested types) ─────────────────────────

/// Configuration for filtering search results.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct SearchFilter {
    /// Minimum relevance score (0.0 to 1.0)
    min_score: f64,
    /// Maximum number of results
    max_results: i32,
    /// Only include results from these categories
    categories: Option<Vec<String>>,
}

/// Output format for summarization.
#[derive(Debug, Clone, Serialize, Deserialize, ToolSchema)]
enum OutputFormat {
    #[serde(rename = "json")]
    Json,
    #[serde(rename = "markdown")]
    Markdown,
    #[serde(rename = "plain_text")]
    PlainText,
}

// ─── Tool: Calculator ───────────────────────────────────────────────

/// Performs basic arithmetic on two numbers.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct CalculatorArgs {
    /// The first operand
    a: f64,
    /// The second operand
    b: f64,
    /// The operation: add, sub, mul, div
    operation: String,
}

struct CalculatorTool;

#[async_trait]
impl BaseTool for CalculatorTool {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Perform arithmetic on two numbers"
    }

    fn args_schema(&self) -> Option<Value> {
        Some(CalculatorArgs::json_schema())
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let args: CalculatorArgs = match &input {
            ToolInput::Text(s) => serde_json::from_str(s)
                .map_err(|e| cognis_core::error::CognisError::ToolException(e.to_string()))?,
            ToolInput::Structured(map) => serde_json::from_value(json!(map))
                .map_err(|e| cognis_core::error::CognisError::ToolException(e.to_string()))?,
            ToolInput::ToolCall(tc) => serde_json::from_value(json!(tc.args))
                .map_err(|e| cognis_core::error::CognisError::ToolException(e.to_string()))?,
        };

        let result = match args.operation.as_str() {
            "add" => args.a + args.b,
            "sub" => args.a - args.b,
            "mul" => args.a * args.b,
            "div" if args.b != 0.0 => args.a / args.b,
            "div" => {
                return Err(cognis_core::error::CognisError::ToolException(
                    "Division by zero".into(),
                ))
            }
            op => {
                return Err(cognis_core::error::CognisError::ToolException(format!(
                    "Unknown operation: {}",
                    op
                )))
            }
        };

        Ok(ToolOutput::Content(json!({
            "result": result,
            "expression": format!("{} {} {} = {}", args.a, args.operation, args.b, result)
        })))
    }
}

// ─── Tool: Search ───────────────────────────────────────────────────

/// Search for documents matching a query.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct SearchArgs {
    /// The search query string
    query: String,
    /// Minimum relevance score
    min_score: Option<f64>,
    /// Maximum results to return
    max_results: Option<i32>,
}

struct SearchTool;

#[async_trait]
impl BaseTool for SearchTool {
    fn name(&self) -> &str {
        "search"
    }

    fn description(&self) -> &str {
        "Search for documents matching a query"
    }

    fn args_schema(&self) -> Option<Value> {
        Some(SearchArgs::json_schema())
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let query = match &input {
            ToolInput::Text(s) => serde_json::from_str::<Value>(s)
                .ok()
                .and_then(|v| v.get("query").and_then(|q| q.as_str()).map(str::to_string))
                .unwrap_or_else(|| s.clone()),
            ToolInput::Structured(map) => map
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("?")
                .to_string(),
            ToolInput::ToolCall(tc) => tc
                .args
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("?")
                .to_string(),
        };

        Ok(ToolOutput::Content(json!({
            "query": query,
            "results": [
                {"title": "Doc 1", "score": 0.95},
                {"title": "Doc 2", "score": 0.87}
            ],
        })))
    }
}

// ─── Main ───────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== Derive Tool Schema Examples ===\n");

    // 1. Show generated JSON schemas
    println!("--- Generated JSON Schemas ---\n");

    println!(
        "CalculatorArgs:\n{}\n",
        serde_json::to_string_pretty(&CalculatorArgs::json_schema())?
    );

    println!(
        "SearchFilter:\n{}\n",
        serde_json::to_string_pretty(&SearchFilter::json_schema())?
    );

    println!(
        "OutputFormat:\n{}\n",
        serde_json::to_string_pretty(&OutputFormat::json_schema())?
    );

    // 2. Use tools with their schemas
    println!("--- Tool Execution ---\n");

    let calc = CalculatorTool;
    println!("Tool: {} — {}", calc.name(), calc.description());
    println!(
        "Schema: {}",
        serde_json::to_string(&calc.args_schema().unwrap())?
    );

    let result = calc
        ._run(ToolInput::Text(
            r#"{"a": 10, "b": 3, "operation": "mul"}"#.into(),
        ))
        .await?;
    println!("Result: {:?}\n", result);

    let search = SearchTool;
    println!("Tool: {} — {}", search.name(), search.description());
    println!(
        "Schema: {}",
        serde_json::to_string(&search.args_schema().unwrap())?
    );

    let result = search
        ._run(ToolInput::Text(r#"{"query": "rust async"}"#.into()))
        .await?;
    println!("Result: {:?}\n", result);

    // 3. Show tool_call_schema (used by LLM providers)
    println!("--- OpenAI-compatible tool_call_schema ---\n");
    println!(
        "{}",
        serde_json::to_string_pretty(&calc.tool_call_schema())?
    );

    println!("\n=== Done ===");
    Ok(())
}
