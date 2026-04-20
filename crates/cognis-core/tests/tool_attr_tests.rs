//! Integration tests for `#[cognis::tool]` on standalone `async fn`s.
//!
//! Lives in `cognis-core` (not `cognis-macros`) because the macro crate
//! can't dev-dep its own consumer — `cognis-core` → `cognis-macros` is
//! the runtime direction.

use cognis_core::error::Result;
use cognis_core::tool;
use cognis_core::tools::{BaseTool, ToolInput, ToolOutput};
use serde_json::json;
use std::collections::HashMap;

/// Search documents in the index.
#[tool(name = "search")]
async fn search(
    /// Query string
    query: String,
    /// Max results
    #[schema(range(min = 1, max = 50))]
    limit: Option<u32>,
) -> Result<ToolOutput> {
    Ok(ToolOutput::Content(json!({ "q": query, "l": limit })))
}

#[tokio::test]
async fn basic_invocation() {
    let tool = Search;
    assert_eq!(tool.name(), "search");
    assert_eq!(tool.description(), "Search documents in the index.");
    let schema = tool.args_schema().unwrap();
    assert_eq!(schema["properties"]["limit"]["minimum"], json!(1));
    assert_eq!(schema["properties"]["limit"]["maximum"], json!(50));
    assert_eq!(
        schema["properties"]["query"]["type"],
        json!("string"),
        "schema: {schema}"
    );

    let mut m = HashMap::new();
    m.insert("query".to_string(), json!("rust"));
    let out = tool._run(ToolInput::Structured(m)).await.unwrap();
    match out {
        ToolOutput::Content(v) => assert_eq!(v, json!({ "q": "rust", "l": null })),
        _ => panic!("expected Content"),
    }
}

#[tokio::test]
async fn validation_rejects_out_of_range() {
    let tool = Search;
    let mut m = HashMap::new();
    m.insert("query".to_string(), json!("x"));
    m.insert("limit".to_string(), json!(100));
    let err = tool._run(ToolInput::Structured(m)).await.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("limit"), "got {msg}");
    assert!(msg.contains("maximum"), "got {msg}");
}

#[tokio::test]
async fn validation_rejects_missing_required() {
    let tool = Search;
    let err = tool
        ._run(ToolInput::Structured(HashMap::new()))
        .await
        .unwrap_err();
    let msg = err.to_string().to_lowercase();
    assert!(msg.contains("query"), "got {msg}");
}

// ---------------------------------------------------------------------------
// length + pattern + enum_values validators
// ---------------------------------------------------------------------------

/// Route requests to a service by region.
#[tool]
async fn route(
    /// Two-letter region code
    #[schema(enum_values("us", "eu", "asia"))]
    region: String,
    /// Human-readable label
    #[schema(length(min = 1, max = 32))]
    label: String,
    /// Optional slug (lowercase letters only)
    #[schema(pattern("^[a-z]+$"))]
    slug: Option<String>,
) -> Result<ToolOutput> {
    Ok(ToolOutput::Content(json!({
        "region": region,
        "label": label,
        "slug": slug,
    })))
}

#[tokio::test]
async fn enum_values_accepts_listed() {
    let tool = Route;
    let mut m = HashMap::new();
    m.insert("region".to_string(), json!("eu"));
    m.insert("label".to_string(), json!("prod"));
    assert!(tool._run(ToolInput::Structured(m)).await.is_ok());
}

#[tokio::test]
async fn enum_values_rejects_unlisted() {
    let tool = Route;
    let mut m = HashMap::new();
    m.insert("region".to_string(), json!("mars"));
    m.insert("label".to_string(), json!("prod"));
    let err = tool._run(ToolInput::Structured(m)).await.unwrap_err();
    assert!(err.to_string().contains("mars"), "got {err}");
}

#[tokio::test]
async fn length_validator_enforced() {
    let tool = Route;
    let mut m = HashMap::new();
    m.insert("region".to_string(), json!("us"));
    m.insert("label".to_string(), json!("")); // empty → below min
    let err = tool._run(ToolInput::Structured(m)).await.unwrap_err();
    assert!(err.to_string().contains("minimum"), "got {err}");
}

#[tokio::test]
async fn pattern_validator_enforced() {
    let tool = Route;
    let mut m = HashMap::new();
    m.insert("region".to_string(), json!("us"));
    m.insert("label".to_string(), json!("ok"));
    m.insert("slug".to_string(), json!("HAS-CAPS"));
    let err = tool._run(ToolInput::Structured(m)).await.unwrap_err();
    assert!(err.to_string().contains("pattern"), "got {err}");
}

#[tokio::test]
async fn default_name_is_fn_name() {
    let tool = Route;
    assert_eq!(tool.name(), "route");
}

#[tokio::test]
async fn empty_description_when_no_doc_comment() {
    #[tool]
    async fn no_docs(_x: String) -> Result<ToolOutput> {
        Ok(ToolOutput::Content(json!(null)))
    }
    let tool = NoDocs;
    assert_eq!(tool.description(), "");
}

#[tokio::test]
async fn description_override_wins_over_doc() {
    /// This doc comment should be ignored.
    #[tool(description = "overridden desc")]
    async fn with_override(_q: String) -> Result<ToolOutput> {
        Ok(ToolOutput::Content(json!(null)))
    }
    let tool = WithOverride;
    assert_eq!(tool.description(), "overridden desc");
}
