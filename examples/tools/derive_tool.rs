//! Derive Tool Example
//!
//! Demonstrates `#[derive(Tool)]` for auto-generating OpenAPI-compatible
//! tool schemas from Rust structs, including nested structs and enums.

use cognis_core::error::Result;
use cognis_core::tools::{BaseTool, ToolInput, ToolJsonSchema, ToolOutput};
use cognis_core::{Tool, ToolSchema};
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Performs basic arithmetic on two numbers.
#[derive(Debug, Clone, Serialize, Deserialize, Tool)]
#[tool(name = "calculator", description = "Perform arithmetic on two numbers")]
struct CalculatorTool {
    /// The first operand
    a: f64,
    /// The second operand
    b: f64,
    /// The operation: add, sub, mul, div
    operation: String,
}

impl CalculatorTool {
    async fn execute(&self) -> Result<ToolOutput> {
        let result = match self.operation.as_str() {
            "add" => self.a + self.b,
            "sub" => self.a - self.b,
            "mul" => self.a * self.b,
            "div" if self.b != 0.0 => self.a / self.b,
            "div" => {
                return Err(cognis_core::error::CognisError::ToolException(
                    "Division by zero".into(),
                ))
            }
            _ => {
                return Err(cognis_core::error::CognisError::ToolException(format!(
                    "Unknown op: {}",
                    self.operation
                )))
            }
        };
        Ok(ToolOutput::Content(
            json!({"result": result, "expression": format!("{} {} {} = {}", self.a, self.operation, self.b, result)}),
        ))
    }
}

/// Configuration for filtering search results.
#[derive(Debug, Clone, Serialize, Deserialize, ToolSchema)]
struct SearchFilter {
    /// Minimum relevance score (0.0 to 1.0)
    min_score: f64,
    /// Maximum number of results
    max_results: i32,
    /// Only include results from these categories
    categories: Option<Vec<String>>,
}

/// Search for documents matching a query.
#[derive(Debug, Clone, Serialize, Deserialize, Tool)]
#[tool(name = "search", description = "Search for documents matching a query")]
struct SearchTool {
    /// The search query string
    query: String,
    /// Filter configuration
    filter: SearchFilter,
    /// Whether to include snippets
    #[serde(default)]
    include_snippets: bool,
}

impl SearchTool {
    async fn execute(&self) -> Result<ToolOutput> {
        Ok(ToolOutput::Content(json!({
            "query": self.query,
            "results": [{"title": "Doc 1", "score": 0.95}, {"title": "Doc 2", "score": 0.87}],
        })))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToolSchema)]
enum OutputFormat {
    #[serde(rename = "json")]
    Json,
    #[serde(rename = "markdown")]
    Markdown,
    #[serde(rename = "plain_text")]
    PlainText,
}

/// Summarize a piece of text.
#[derive(Debug, Clone, Serialize, Deserialize, Tool)]
#[tool(name = "summarize", description = "Summarize text in a given format")]
struct SummarizeTool {
    /// The text to summarize
    text: String,
    /// Desired output format
    format: OutputFormat,
    /// Maximum length in words
    max_words: Option<i32>,
}

impl SummarizeTool {
    async fn execute(&self) -> Result<ToolOutput> {
        let summary = if self.text.len() > 100 {
            format!("{}...", &self.text[..100])
        } else {
            self.text.clone()
        };
        Ok(ToolOutput::Content(
            json!({"summary": summary, "word_count": summary.split_whitespace().count()}),
        ))
    }
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== Derive Tool Examples ===\n");

    let empty = ToolInput::Text(String::new());

    let calc = CalculatorTool {
        a: 10.0,
        b: 3.0,
        operation: "mul".into(),
    };
    println!(
        "{} schema: {}",
        calc.name(),
        serde_json::to_string(&calc.args_schema().unwrap())?
    );
    println!("Result: {:?}\n", calc._run(empty.clone()).await?);

    let search = SearchTool {
        query: "rust async".into(),
        filter: SearchFilter {
            min_score: 0.8,
            max_results: 5,
            categories: Some(vec!["rust".into()]),
        },
        include_snippets: true,
    };
    println!(
        "{}: {:?}\n",
        search.name(),
        search._run(empty.clone()).await?
    );

    let summ = SummarizeTool {
        text: "Rust is a systems language focused on safety and concurrency.".into(),
        format: OutputFormat::Markdown,
        max_words: Some(50),
    };
    println!("{}: {:?}\n", summ.name(), summ._run(empty).await?);

    println!(
        "OutputFormat: {}",
        serde_json::to_string(&OutputFormat::json_schema())?
    );
    println!(
        "SearchFilter: {}",
        serde_json::to_string(&SearchFilter::json_schema())?
    );
    Ok(())
}
