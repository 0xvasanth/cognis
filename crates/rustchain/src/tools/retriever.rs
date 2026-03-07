//! Adapter that wraps any `BaseRetriever` as a `BaseTool`, making retrievers
//! callable as tools by agents.
//!
//! Provides rich document formatting, builder pattern construction, structured
//! JSON input parsing, and multi-retriever routing.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use rustchain_core::documents::Document;
use rustchain_core::error::{Result, RustChainError};
use rustchain_core::retrievers::BaseRetriever;
use rustchain_core::tools::base::BaseTool;
use rustchain_core::tools::types::{ErrorHandler, ResponseFormat, ToolInput, ToolOutput};

/// A shared, thread-safe document formatting function.
type DocumentFormatFn = Arc<dyn Fn(&[Document]) -> String + Send + Sync>;

// ---------------------------------------------------------------------------
// DocumentFormatter
// ---------------------------------------------------------------------------

/// Controls how retrieved documents are rendered into a single text output.
#[derive(Default)]
pub enum DocumentFormatter {
    /// Just page_content, newline separated.
    #[default]
    Plain,
    /// "1. content\n2. content\n..."
    Numbered,
    /// "Source: {source}\nContent: {content}\n---"
    WithSource,
    /// JSON array of `{content, metadata}`.
    Json,
    /// Markdown formatted sections.
    Markdown,
    /// User-supplied formatting function.
    Custom(DocumentFormatFn),
}

impl DocumentFormatter {
    /// Format documents according to the chosen strategy.
    pub fn format(&self, docs: &[Document], include_metadata: bool) -> String {
        match self {
            DocumentFormatter::Plain => docs
                .iter()
                .map(|d| d.page_content.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
            DocumentFormatter::Numbered => docs
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    if include_metadata && !d.metadata.is_empty() {
                        format!(
                            "{}. {}\n   Metadata: {}",
                            i + 1,
                            d.page_content,
                            serde_json::to_string(&d.metadata).unwrap_or_default()
                        )
                    } else {
                        format!("{}. {}", i + 1, d.page_content)
                    }
                })
                .collect::<Vec<_>>()
                .join("\n"),
            DocumentFormatter::WithSource => docs
                .iter()
                .map(|d| {
                    let source = d
                        .metadata
                        .get("source")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    format!("Source: {}\nContent: {}", source, d.page_content)
                })
                .collect::<Vec<_>>()
                .join("\n---\n"),
            DocumentFormatter::Json => {
                let arr: Vec<Value> = docs
                    .iter()
                    .map(|d| {
                        let mut obj = json!({ "content": d.page_content });
                        if include_metadata {
                            obj["metadata"] = json!(d.metadata);
                        }
                        obj
                    })
                    .collect();
                serde_json::to_string(&arr).unwrap_or_else(|_| "[]".to_string())
            }
            DocumentFormatter::Markdown => docs
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    let mut section = format!("## Result {}\n\n{}", i + 1, d.page_content);
                    if include_metadata && !d.metadata.is_empty() {
                        section.push_str("\n\n**Metadata:**\n");
                        for (k, v) in &d.metadata {
                            section.push_str(&format!("- **{}:** {}\n", k, v));
                        }
                    }
                    section
                })
                .collect::<Vec<_>>()
                .join("\n\n"),
            DocumentFormatter::Custom(f) => f(docs),
        }
    }
}

// ---------------------------------------------------------------------------
// RetrieverTool
// ---------------------------------------------------------------------------

/// A tool that wraps a `BaseRetriever`, exposing document retrieval as a tool
/// callable by agents.
///
/// Supports multiple document formatters, result limiting, metadata inclusion,
/// and structured JSON input parsing.
pub struct RetrieverTool {
    retriever: Arc<dyn BaseRetriever>,
    name: String,
    description: String,
    document_formatter: DocumentFormatter,
    max_results: Option<usize>,
    include_metadata: bool,
}

impl RetrieverTool {
    /// Parse the query (and optional parameters) from the tool input.
    ///
    /// Accepts:
    /// - Plain string queries
    /// - JSON `{"query": "...", "k": 5}`
    fn parse_input(input: &ToolInput) -> (String, Option<usize>) {
        match input {
            ToolInput::Text(s) => {
                // Try parsing as JSON first
                if let Ok(v) = serde_json::from_str::<Value>(s) {
                    if let Some(q) = v.get("query").and_then(|q| q.as_str()) {
                        let k = v.get("k").and_then(|k| k.as_u64()).map(|k| k as usize);
                        return (q.to_string(), k);
                    }
                }
                (s.clone(), None)
            }
            ToolInput::Structured(map) => {
                let query = map
                    .get("query")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| {
                        map.values()
                            .find_map(|v| v.as_str().map(|s| s.to_string()))
                            .unwrap_or_default()
                    });
                let k = map.get("k").and_then(|v| v.as_u64()).map(|k| k as usize);
                (query, k)
            }
            ToolInput::ToolCall(tc) => {
                let query = tc
                    .args
                    .get("query")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| {
                        tc.args
                            .values()
                            .find_map(|v| v.as_str().map(|s| s.to_string()))
                            .unwrap_or_default()
                    });
                let k = tc
                    .args
                    .get("k")
                    .and_then(|v| v.as_u64())
                    .map(|k| k as usize);
                (query, k)
            }
        }
    }
}

#[async_trait]
impl BaseTool for RetrieverTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to retrieve relevant documents"
                },
                "k": {
                    "type": "integer",
                    "description": "Maximum number of documents to return"
                }
            },
            "required": ["query"]
        }))
    }

    fn return_direct(&self) -> bool {
        false
    }

    fn handle_tool_error(&self) -> &ErrorHandler {
        &ErrorHandler::Propagate
    }

    fn handle_validation_error(&self) -> &ErrorHandler {
        &ErrorHandler::Propagate
    }

    fn response_format(&self) -> ResponseFormat {
        ResponseFormat::Content
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let (query, input_k) = Self::parse_input(&input);
        let mut docs = self.retriever.get_relevant_documents(&query).await?;

        // Apply max_results limit (input k overrides configured max_results)
        let limit = input_k.or(self.max_results);
        if let Some(max) = limit {
            docs.truncate(max);
        }

        let text = self.document_formatter.format(&docs, self.include_metadata);
        Ok(ToolOutput::Content(Value::String(text)))
    }
}

// ---------------------------------------------------------------------------
// RetrieverToolBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing a `RetrieverTool` with custom configuration.
pub struct RetrieverToolBuilder {
    retriever: Option<Arc<dyn BaseRetriever>>,
    name: Option<String>,
    description: Option<String>,
    document_formatter: DocumentFormatter,
    max_results: Option<usize>,
    include_metadata: bool,
}

impl RetrieverToolBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            retriever: None,
            name: None,
            description: None,
            document_formatter: DocumentFormatter::Plain,
            max_results: None,
            include_metadata: false,
        }
    }

    /// Set the retriever to wrap.
    pub fn retriever(mut self, retriever: impl BaseRetriever + 'static) -> Self {
        self.retriever = Some(Arc::new(retriever));
        self
    }

    /// Set the retriever from an Arc.
    pub fn retriever_arc(mut self, retriever: Arc<dyn BaseRetriever>) -> Self {
        self.retriever = Some(retriever);
        self
    }

    /// Set the tool name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the tool description.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the document formatter.
    pub fn formatter(mut self, formatter: DocumentFormatter) -> Self {
        self.document_formatter = formatter;
        self
    }

    /// Set the maximum number of results to return.
    pub fn max_results(mut self, max: usize) -> Self {
        self.max_results = Some(max);
        self
    }

    /// Set whether to include metadata in the output.
    pub fn include_metadata(mut self, include: bool) -> Self {
        self.include_metadata = include;
        self
    }

    /// Build the `RetrieverTool`.
    ///
    /// # Panics
    ///
    /// Panics if `retriever` or `name` have not been set.
    pub fn build(self) -> RetrieverTool {
        RetrieverTool {
            retriever: self.retriever.expect("retriever is required"),
            name: self.name.expect("name is required"),
            description: self.description.unwrap_or_default(),
            document_formatter: self.document_formatter,
            max_results: self.max_results,
            include_metadata: self.include_metadata,
        }
    }
}

impl Default for RetrieverToolBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Convenience constructor
// ---------------------------------------------------------------------------

/// Create a `RetrieverTool` from a retriever, name, and description.
///
/// Uses `DocumentFormatter::Plain` with no result limit and metadata excluded.
pub fn create_retriever_tool(
    retriever: impl BaseRetriever + 'static,
    name: impl Into<String>,
    description: impl Into<String>,
) -> RetrieverTool {
    RetrieverTool {
        retriever: Arc::new(retriever),
        name: name.into(),
        description: description.into(),
        document_formatter: DocumentFormatter::Plain,
        max_results: None,
        include_metadata: false,
    }
}

// ---------------------------------------------------------------------------
// RoutingStrategy & MultiRetrieverTool
// ---------------------------------------------------------------------------

/// Determines how a `MultiRetrieverTool` routes queries across retrievers.
pub enum RoutingStrategy {
    /// Send the query to all retrievers and merge results.
    All,
    /// Send the query to the first retriever only.
    First,
    /// Rotate through retrievers in round-robin fashion.
    RoundRobin,
    /// Parse a `retriever` field from JSON input to choose which retriever to
    /// query. Falls back to `All` if no prefix/field is present.
    ByPrefix,
}

/// A tool that wraps multiple named retrievers and routes queries according to
/// a `RoutingStrategy`.
pub struct MultiRetrieverTool {
    retrievers: Vec<(String, Arc<dyn BaseRetriever>)>,
    name: String,
    description: String,
    routing: RoutingStrategy,
    document_formatter: DocumentFormatter,
    max_results: Option<usize>,
    include_metadata: bool,
    /// Internal round-robin counter (interior mutability via atomic).
    rr_counter: std::sync::atomic::AtomicUsize,
}

impl MultiRetrieverTool {
    /// Create a new `MultiRetrieverTool`.
    pub fn new(
        retrievers: Vec<(String, Arc<dyn BaseRetriever>)>,
        name: impl Into<String>,
        description: impl Into<String>,
        routing: RoutingStrategy,
    ) -> Self {
        Self {
            retrievers,
            name: name.into(),
            description: description.into(),
            routing,
            document_formatter: DocumentFormatter::Plain,
            max_results: None,
            include_metadata: false,
            rr_counter: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Set the document formatter.
    pub fn with_formatter(mut self, formatter: DocumentFormatter) -> Self {
        self.document_formatter = formatter;
        self
    }

    /// Set the maximum number of results.
    pub fn with_max_results(mut self, max: usize) -> Self {
        self.max_results = Some(max);
        self
    }

    /// Set whether to include metadata.
    pub fn with_include_metadata(mut self, include: bool) -> Self {
        self.include_metadata = include;
        self
    }

    /// Parse the query and optional retriever name from the input.
    fn parse_input(input: &ToolInput) -> (String, Option<String>) {
        match input {
            ToolInput::Text(s) => {
                if let Ok(v) = serde_json::from_str::<Value>(s) {
                    let query = v
                        .get("query")
                        .and_then(|q| q.as_str())
                        .unwrap_or("")
                        .to_string();
                    let retriever = v
                        .get("retriever")
                        .and_then(|r| r.as_str())
                        .map(|s| s.to_string());
                    return (query, retriever);
                }
                (s.clone(), None)
            }
            ToolInput::Structured(map) => {
                let query = map
                    .get("query")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let retriever = map
                    .get("retriever")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                (query, retriever)
            }
            ToolInput::ToolCall(tc) => {
                let query = tc
                    .args
                    .get("query")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let retriever = tc
                    .args
                    .get("retriever")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                (query, retriever)
            }
        }
    }
}

#[async_trait]
impl BaseTool for MultiRetrieverTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn args_schema(&self) -> Option<Value> {
        let retriever_names: Vec<&str> = self.retrievers.iter().map(|(n, _)| n.as_str()).collect();
        Some(json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "retriever": {
                    "type": "string",
                    "description": format!("Which retriever to use. Options: {}", retriever_names.join(", "))
                }
            },
            "required": ["query"]
        }))
    }

    fn return_direct(&self) -> bool {
        false
    }

    fn handle_tool_error(&self) -> &ErrorHandler {
        &ErrorHandler::Propagate
    }

    fn handle_validation_error(&self) -> &ErrorHandler {
        &ErrorHandler::Propagate
    }

    fn response_format(&self) -> ResponseFormat {
        ResponseFormat::Content
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        if self.retrievers.is_empty() {
            return Ok(ToolOutput::Content(Value::String(String::new())));
        }

        let (query, retriever_name) = Self::parse_input(&input);
        let mut all_docs: Vec<Document> = Vec::new();

        match &self.routing {
            RoutingStrategy::All => {
                for (_, retriever) in &self.retrievers {
                    let docs = retriever.get_relevant_documents(&query).await?;
                    all_docs.extend(docs);
                }
            }
            RoutingStrategy::First => {
                if let Some((_, retriever)) = self.retrievers.first() {
                    all_docs = retriever.get_relevant_documents(&query).await?;
                }
            }
            RoutingStrategy::RoundRobin => {
                let idx = self
                    .rr_counter
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                    % self.retrievers.len();
                let (_, retriever) = &self.retrievers[idx];
                all_docs = retriever.get_relevant_documents(&query).await?;
            }
            RoutingStrategy::ByPrefix => {
                if let Some(name) = &retriever_name {
                    if let Some((_, retriever)) = self.retrievers.iter().find(|(n, _)| n == name) {
                        all_docs = retriever.get_relevant_documents(&query).await?;
                    } else {
                        return Err(RustChainError::Other(format!(
                            "Unknown retriever: {}",
                            name
                        )));
                    }
                } else {
                    // Fall back to All
                    for (_, retriever) in &self.retrievers {
                        let docs = retriever.get_relevant_documents(&query).await?;
                        all_docs.extend(docs);
                    }
                }
            }
        }

        if let Some(max) = self.max_results {
            all_docs.truncate(max);
        }

        let text = self
            .document_formatter
            .format(&all_docs, self.include_metadata);
        Ok(ToolOutput::Content(Value::String(text)))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A trivial in-memory retriever for testing.
    struct MockRetriever {
        docs: Vec<Document>,
    }

    #[async_trait]
    impl BaseRetriever for MockRetriever {
        async fn get_relevant_documents(&self, _query: &str) -> Result<Vec<Document>> {
            Ok(self.docs.clone())
        }
    }

    fn sample_docs() -> Vec<Document> {
        vec![
            Document {
                page_content: "Rust is a systems programming language".into(),
                metadata: {
                    let mut m = HashMap::new();
                    m.insert("source".to_string(), json!("docs/rust.md"));
                    m.insert("page".to_string(), json!(1));
                    m
                },
                id: Some("doc1".into()),
                doc_type: None,
            },
            Document {
                page_content: "Python is great for prototyping".into(),
                metadata: {
                    let mut m = HashMap::new();
                    m.insert("source".to_string(), json!("docs/python.md"));
                    m.insert("page".to_string(), json!(2));
                    m
                },
                id: Some("doc2".into()),
                doc_type: None,
            },
            Document {
                page_content: "Go is designed for concurrency".into(),
                metadata: {
                    let mut m = HashMap::new();
                    m.insert("source".to_string(), json!("docs/go.md"));
                    m
                },
                id: Some("doc3".into()),
                doc_type: None,
            },
        ]
    }

    fn mock_retriever(docs: Vec<Document>) -> MockRetriever {
        MockRetriever { docs }
    }

    // -----------------------------------------------------------------------
    // 1. RetrieverTool wraps retriever correctly
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_retriever_tool_wraps_retriever() {
        let docs = sample_docs();
        let tool = create_retriever_tool(mock_retriever(docs.clone()), "search", "Search docs");

        assert_eq!(tool.name(), "search");
        assert_eq!(tool.description(), "Search docs");

        let result = tool._run(ToolInput::Text("query".into())).await.unwrap();
        match result {
            ToolOutput::Content(Value::String(s)) => {
                assert!(s.contains("Rust is a systems programming language"));
                assert!(s.contains("Python is great for prototyping"));
            }
            _ => panic!("Expected string Content output"),
        }
    }

    // -----------------------------------------------------------------------
    // 2. Tool invocation returns formatted documents
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_retriever_tool_invocation_returns_formatted() {
        let tool = RetrieverToolBuilder::new()
            .retriever(mock_retriever(sample_docs()))
            .name("search")
            .description("Search")
            .formatter(DocumentFormatter::Numbered)
            .build();

        let result = tool._run(ToolInput::Text("test".into())).await.unwrap();
        match result {
            ToolOutput::Content(Value::String(s)) => {
                assert!(s.starts_with("1. "));
                assert!(s.contains("2. "));
                assert!(s.contains("3. "));
            }
            _ => panic!("Expected string Content output"),
        }
    }

    // -----------------------------------------------------------------------
    // 3. Plain formatter
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_retriever_tool_plain_formatter() {
        let docs = vec![
            Document::new("Alpha"),
            Document::new("Beta"),
            Document::new("Gamma"),
        ];
        let tool = create_retriever_tool(mock_retriever(docs), "s", "d");

        let result = tool._run(ToolInput::Text("q".into())).await.unwrap();
        match result {
            ToolOutput::Content(Value::String(s)) => {
                assert_eq!(s, "Alpha\nBeta\nGamma");
            }
            _ => panic!("Expected string Content output"),
        }
    }

    // -----------------------------------------------------------------------
    // 4. Numbered formatter
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_retriever_tool_numbered_formatter() {
        let docs = vec![Document::new("First"), Document::new("Second")];
        let formatter = DocumentFormatter::Numbered;
        let output = formatter.format(&docs, false);
        assert_eq!(output, "1. First\n2. Second");
    }

    // -----------------------------------------------------------------------
    // 5. WithSource formatter
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_retriever_tool_with_source_formatter() {
        let docs = sample_docs();
        let formatter = DocumentFormatter::WithSource;
        let output = formatter.format(&docs, false);
        assert!(output.contains("Source: docs/rust.md"));
        assert!(output.contains("Content: Rust is a systems programming language"));
        assert!(output.contains("---"));
        assert!(output.contains("Source: docs/python.md"));
    }

    // -----------------------------------------------------------------------
    // 6. Json formatter
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_retriever_tool_json_formatter() {
        let docs = vec![Document::new("Hello")];
        let formatter = DocumentFormatter::Json;
        let output = formatter.format(&docs, true);
        let parsed: Vec<Value> = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0]["content"], "Hello");
    }

    // -----------------------------------------------------------------------
    // 7. Markdown formatter
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_retriever_tool_markdown_formatter() {
        let docs = sample_docs();
        let formatter = DocumentFormatter::Markdown;
        let output = formatter.format(&docs, true);
        assert!(output.contains("## Result 1"));
        assert!(output.contains("## Result 2"));
        assert!(output.contains("**Metadata:**"));
        assert!(output.contains("**source:**"));
    }

    // -----------------------------------------------------------------------
    // 8. Max results limiting
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_retriever_tool_max_results() {
        let tool = RetrieverToolBuilder::new()
            .retriever(mock_retriever(sample_docs()))
            .name("search")
            .description("Search")
            .max_results(1)
            .build();

        let result = tool._run(ToolInput::Text("q".into())).await.unwrap();
        match result {
            ToolOutput::Content(Value::String(s)) => {
                // Plain formatter, only 1 doc, so no newline
                assert!(!s.contains('\n'));
                assert!(s.contains("Rust is a systems programming language"));
            }
            _ => panic!("Expected string Content output"),
        }
    }

    // -----------------------------------------------------------------------
    // 9. Include metadata toggle
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_retriever_tool_include_metadata() {
        let docs = sample_docs();

        // With metadata
        let formatter = DocumentFormatter::Numbered;
        let with_meta = formatter.format(&docs, true);
        assert!(with_meta.contains("Metadata:"));

        // Without metadata
        let formatter2 = DocumentFormatter::Numbered;
        let without_meta = formatter2.format(&docs, false);
        assert!(!without_meta.contains("Metadata:"));
    }

    // -----------------------------------------------------------------------
    // 10. Builder pattern
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_retriever_tool_builder() {
        let tool = RetrieverToolBuilder::new()
            .retriever(mock_retriever(sample_docs()))
            .name("my_search")
            .description("Search the knowledge base")
            .formatter(DocumentFormatter::WithSource)
            .max_results(2)
            .include_metadata(true)
            .build();

        assert_eq!(tool.name(), "my_search");
        assert_eq!(tool.description(), "Search the knowledge base");

        let result = tool._run(ToolInput::Text("q".into())).await.unwrap();
        match result {
            ToolOutput::Content(Value::String(s)) => {
                assert!(s.contains("Source:"));
                // Only 2 docs due to max_results
                assert!(s.contains("docs/rust.md"));
                assert!(s.contains("docs/python.md"));
                assert!(!s.contains("docs/go.md"));
            }
            _ => panic!("Expected string Content output"),
        }
    }

    // -----------------------------------------------------------------------
    // 11. create_retriever_tool convenience
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_retriever_tool_create_convenience() {
        let tool = create_retriever_tool(
            mock_retriever(vec![Document::new("doc content")]),
            "quick_search",
            "A quick search tool",
        );
        assert_eq!(tool.name(), "quick_search");
        assert_eq!(tool.description(), "A quick search tool");

        let result = tool._run(ToolInput::Text("q".into())).await.unwrap();
        match result {
            ToolOutput::Content(Value::String(s)) => {
                assert_eq!(s, "doc content");
            }
            _ => panic!("Expected string Content output"),
        }
    }

    // -----------------------------------------------------------------------
    // 12. MultiRetrieverTool with routing
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_retriever_tool_multi_retriever_routing() {
        let r1 =
            Arc::new(mock_retriever(vec![Document::new("from docs")])) as Arc<dyn BaseRetriever>;
        let r2 =
            Arc::new(mock_retriever(vec![Document::new("from code")])) as Arc<dyn BaseRetriever>;

        let tool = MultiRetrieverTool::new(
            vec![("docs".to_string(), r1), ("code".to_string(), r2)],
            "multi_search",
            "Search multiple sources",
            RoutingStrategy::ByPrefix,
        );

        // Route to "docs"
        let input_json = r#"{"query": "test", "retriever": "docs"}"#;
        let result = tool._run(ToolInput::Text(input_json.into())).await.unwrap();
        match result {
            ToolOutput::Content(Value::String(s)) => {
                assert_eq!(s, "from docs");
            }
            _ => panic!("Expected string Content output"),
        }

        // Route to "code"
        let input_json = r#"{"query": "test", "retriever": "code"}"#;
        let result = tool._run(ToolInput::Text(input_json.into())).await.unwrap();
        match result {
            ToolOutput::Content(Value::String(s)) => {
                assert_eq!(s, "from code");
            }
            _ => panic!("Expected string Content output"),
        }

        // All (no retriever specified)
        let tool_all = MultiRetrieverTool::new(
            vec![
                (
                    "docs".to_string(),
                    Arc::new(mock_retriever(vec![Document::new("from docs")]))
                        as Arc<dyn BaseRetriever>,
                ),
                (
                    "code".to_string(),
                    Arc::new(mock_retriever(vec![Document::new("from code")]))
                        as Arc<dyn BaseRetriever>,
                ),
            ],
            "multi_search",
            "Search multiple sources",
            RoutingStrategy::All,
        );
        let result = tool_all._run(ToolInput::Text("test".into())).await.unwrap();
        match result {
            ToolOutput::Content(Value::String(s)) => {
                assert!(s.contains("from docs"));
                assert!(s.contains("from code"));
            }
            _ => panic!("Expected string Content output"),
        }
    }

    // -----------------------------------------------------------------------
    // 13. JSON input parsing
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_retriever_tool_json_input_parsing() {
        let tool = RetrieverToolBuilder::new()
            .retriever(mock_retriever(sample_docs()))
            .name("search")
            .description("Search")
            .max_results(10) // high limit
            .build();

        // JSON with k override
        let input_json = r#"{"query": "test", "k": 1}"#;
        let result = tool._run(ToolInput::Text(input_json.into())).await.unwrap();
        match result {
            ToolOutput::Content(Value::String(s)) => {
                // Should only have 1 result
                assert_eq!(s, "Rust is a systems programming language");
            }
            _ => panic!("Expected string Content output"),
        }
    }

    // -----------------------------------------------------------------------
    // 14. Empty results formatting
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_retriever_tool_empty_results() {
        let tool = create_retriever_tool(mock_retriever(vec![]), "search", "Search");

        let result = tool._run(ToolInput::Text("q".into())).await.unwrap();
        match result {
            ToolOutput::Content(Value::String(s)) => {
                assert_eq!(s, "");
            }
            _ => panic!("Expected string Content output"),
        }
    }

    // -----------------------------------------------------------------------
    // 15. Tool schema includes description
    // -----------------------------------------------------------------------
    #[test]
    fn test_retriever_tool_schema_includes_description() {
        let tool = create_retriever_tool(
            mock_retriever(vec![]),
            "my_tool",
            "A detailed description of the tool",
        );
        assert_eq!(tool.description(), "A detailed description of the tool");

        let schema = tool.args_schema().unwrap();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["query"].is_object());
        assert_eq!(schema["required"][0], "query");
    }
}
