use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use super::base::BaseTool;
use super::types::{ErrorHandler, ResponseFormat, ToolInput, ToolOutput};
use crate::documents::Document;
use crate::error::Result;
use crate::retrievers::BaseRetriever;

/// A tool that wraps a `BaseRetriever`, exposing document retrieval as a tool
/// callable by agents.
///
/// Mirrors Python's `langchain_core.tools.retriever.create_retriever_tool`.
///
/// On invocation the tool extracts a query string from the input, calls
/// `retriever.get_relevant_documents(query)`, and returns the documents
/// formatted as concatenated text.
///
/// # Example
///
/// ```ignore
/// use cognis_core::tools::retriever::create_retriever_tool;
///
/// let tool = create_retriever_tool(
///     my_retriever,
///     "search_docs",
///     "Search the knowledge base for relevant documents",
/// );
/// let result = tool.run_str("what is Rust?").await?;
/// ```
pub struct RetrieverTool {
    retriever: Arc<dyn BaseRetriever>,
    name: String,
    description: String,
    document_separator: String,
}

impl RetrieverTool {
    /// Set a custom separator used between documents in the output.
    /// Defaults to `"\n\n"`.
    pub fn with_document_separator(mut self, sep: impl Into<String>) -> Self {
        self.document_separator = sep.into();
        self
    }

    /// Extract the query string from the tool input.
    fn extract_query(input: &ToolInput) -> String {
        match input {
            ToolInput::Text(s) => s.clone(),
            ToolInput::ToolCall(tc) => {
                // Prefer an explicit "query" argument, fall back to first string arg.
                if let Some(Value::String(q)) = tc.args.get("query") {
                    q.clone()
                } else {
                    // Use the first string value found, or serialize the whole map.
                    tc.args
                        .values()
                        .find_map(|v| v.as_str().map(|s| s.to_string()))
                        .unwrap_or_else(|| serde_json::to_string(&tc.args).unwrap_or_default())
                }
            }
            ToolInput::Structured(map) => {
                if let Some(Value::String(q)) = map.get("query") {
                    q.clone()
                } else {
                    map.values()
                        .find_map(|v| v.as_str().map(|s| s.to_string()))
                        .unwrap_or_else(|| serde_json::to_string(map).unwrap_or_default())
                }
            }
        }
    }

    /// Format retrieved documents into a single text string.
    fn format_documents(&self, docs: &[Document]) -> String {
        docs.iter()
            .map(|doc| doc.page_content.as_str())
            .collect::<Vec<_>>()
            .join(&self.document_separator)
    }
}

/// Create a `RetrieverTool` from a retriever, name, and description.
///
/// This is the primary constructor, mirroring Python's
/// `create_retriever_tool(retriever, name, description)`.
pub fn create_retriever_tool(
    retriever: impl BaseRetriever + 'static,
    name: impl Into<String>,
    description: impl Into<String>,
) -> RetrieverTool {
    RetrieverTool {
        retriever: Arc::new(retriever),
        name: name.into(),
        description: description.into(),
        document_separator: "\n\n".to_string(),
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
        let query = Self::extract_query(&input);
        let docs = self.retriever.get_relevant_documents(&query).await?;
        let text = self.format_documents(&docs);
        Ok(ToolOutput::Content(Value::String(text)))
    }
}

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

    fn make_mock_retriever(contents: Vec<&str>) -> MockRetriever {
        MockRetriever {
            docs: contents.into_iter().map(|c| Document::new(c)).collect(),
        }
    }

    #[tokio::test]
    async fn test_retriever_tool_text_input() {
        let retriever = make_mock_retriever(vec!["Hello world", "Rust is great"]);
        let tool = create_retriever_tool(retriever, "search", "Search docs");

        let result = tool._run(ToolInput::Text("test".into())).await.unwrap();
        match result {
            ToolOutput::Content(Value::String(s)) => {
                assert_eq!(s, "Hello world\n\nRust is great");
            }
            _ => panic!("Expected string Content output"),
        }
    }

    #[tokio::test]
    async fn test_retriever_tool_structured_input() {
        let retriever = make_mock_retriever(vec!["Doc 1"]);
        let tool = create_retriever_tool(retriever, "search", "Search docs");

        let mut args = HashMap::new();
        args.insert("query".to_string(), Value::String("my query".into()));
        let result = tool._run(ToolInput::Structured(args)).await.unwrap();
        match result {
            ToolOutput::Content(Value::String(s)) => {
                assert_eq!(s, "Doc 1");
            }
            _ => panic!("Expected string Content output"),
        }
    }

    #[tokio::test]
    async fn test_custom_separator() {
        let retriever = make_mock_retriever(vec!["A", "B", "C"]);
        let tool = create_retriever_tool(retriever, "s", "d").with_document_separator(" | ");

        let result = tool._run(ToolInput::Text("q".into())).await.unwrap();
        match result {
            ToolOutput::Content(Value::String(s)) => {
                assert_eq!(s, "A | B | C");
            }
            _ => panic!("Expected string Content output"),
        }
    }

    #[test]
    fn test_name_description_schema() {
        let retriever = make_mock_retriever(vec![]);
        let tool = create_retriever_tool(retriever, "my_search", "Find stuff");
        assert_eq!(tool.name(), "my_search");
        assert_eq!(tool.description(), "Find stuff");
        let schema = tool.args_schema().unwrap();
        assert_eq!(schema["properties"]["query"]["type"], "string");
    }
}
