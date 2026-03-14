//! Example demonstrating the `#[derive(Tool)]` macro for auto-generating
//! OpenAPI-compatible tool schemas from Rust structs.

use cognis_core::error::Result;
use cognis_core::tools::{BaseTool, ToolInput, ToolJsonSchema, ToolOutput};
use cognis_core::{Tool, ToolSchema};
use serde::{Deserialize, Serialize};
use serde_json::json;

// ---------------------------------------------------------------------------
// 1. Simple calculator tool
// ---------------------------------------------------------------------------

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
            "div" => {
                if self.b == 0.0 {
                    return Err(cognis_core::error::CognisError::ToolException(
                        "Division by zero".into(),
                    ));
                }
                self.a / self.b
            }
            _ => {
                return Err(cognis_core::error::CognisError::ToolException(format!(
                    "Unknown operation: {}",
                    self.operation
                )))
            }
        };
        Ok(ToolOutput::Content(json!({
            "result": result,
            "expression": format!("{} {} {} = {}", self.a, self.operation, self.b, result),
        })))
    }
}

// ---------------------------------------------------------------------------
// 2. Search tool with nested filter struct
// ---------------------------------------------------------------------------

/// Configuration for filtering search results.
#[derive(Debug, Clone, Serialize, Deserialize, ToolSchema)]
struct SearchFilter {
    /// Minimum relevance score (0.0 to 1.0)
    min_score: f64,
    /// Maximum number of results to return
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
    /// Filter configuration for the search
    filter: SearchFilter,
    /// Whether to include document snippets in results
    #[serde(default)]
    include_snippets: bool,
}

impl SearchTool {
    async fn execute(&self) -> Result<ToolOutput> {
        // Simulated search results
        Ok(ToolOutput::Content(json!({
            "query": self.query,
            "results": [
                {"title": "Document 1", "score": 0.95},
                {"title": "Document 2", "score": 0.87},
            ],
            "total": 2,
        })))
    }
}

// ---------------------------------------------------------------------------
// 3. Tool with enum parameter
// ---------------------------------------------------------------------------

/// The output format for generated content.
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
    /// Maximum length of the summary in words
    max_words: Option<i32>,
}

impl SummarizeTool {
    async fn execute(&self) -> Result<ToolOutput> {
        let summary = if self.text.len() > 100 {
            format!("{}...", &self.text[..100])
        } else {
            self.text.clone()
        };
        Ok(ToolOutput::Content(json!({
            "summary": summary,
            "word_count": summary.split_whitespace().count(),
        })))
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== Derive Tool Examples ===\n");

    // --- Calculator ---
    let calculator = CalculatorTool {
        a: 10.0,
        b: 3.0,
        operation: "mul".into(),
    };

    println!("Tool: {}", calculator.name());
    println!("Description: {}", calculator.description());
    println!(
        "Schema:\n{}\n",
        serde_json::to_string_pretty(&calculator.args_schema().unwrap())?
    );

    let result = calculator._run(ToolInput::Text(String::new())).await?;
    println!("Result: {:?}\n", result);

    // --- Search with nested filter ---
    let search = SearchTool {
        query: "rust async programming".into(),
        filter: SearchFilter {
            min_score: 0.8,
            max_results: 5,
            categories: Some(vec!["programming".into(), "rust".into()]),
        },
        include_snippets: true,
    };

    println!("Tool: {}", search.name());
    println!("Description: {}", search.description());
    println!(
        "Schema:\n{}\n",
        serde_json::to_string_pretty(&search.args_schema().unwrap())?
    );

    let result = search._run(ToolInput::Text(String::new())).await?;
    println!("Result: {:?}\n", result);

    // --- Summarize with enum ---
    let summarize = SummarizeTool {
        text: "Rust is a systems programming language focused on safety, speed, and concurrency."
            .into(),
        format: OutputFormat::Markdown,
        max_words: Some(50),
    };

    println!("Tool: {}", summarize.name());
    println!("Description: {}", summarize.description());
    println!(
        "Schema:\n{}\n",
        serde_json::to_string_pretty(&summarize.args_schema().unwrap())?
    );

    let result = summarize._run(ToolInput::Text(String::new())).await?;
    println!("Result: {:?}\n", result);

    // --- Show enum schema ---
    println!("OutputFormat enum schema:");
    println!(
        "{}\n",
        serde_json::to_string_pretty(&OutputFormat::json_schema())?
    );

    // --- Show nested struct schema ---
    println!("SearchFilter struct schema:");
    println!(
        "{}\n",
        serde_json::to_string_pretty(&SearchFilter::json_schema())?
    );

    Ok(())
}
