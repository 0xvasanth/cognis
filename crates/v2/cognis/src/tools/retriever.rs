//! Wrap any `Runnable<String, Vec<Document>>` as a [`Tool`].
//!
//! Lets agents invoke retrievers through the standard tool dispatch path:
//! the LLM emits a `tool_call` with a `query` argument, the tool runs the
//! retriever, and the matched documents come back as the tool result.

use std::sync::Arc;

use async_trait::async_trait;
use cognis2_core::schemars::{self, JsonSchema};
use serde::Deserialize;

use cognis2_core::{CognisError, Result, Runnable, RunnableConfig};
use cognis2_llm::tools::{Tool, ToolInput, ToolOutput};
use cognis2_rag::Document;

#[derive(Debug, Deserialize, JsonSchema)]
struct RetrieverInput {
    /// The query string to retrieve documents for.
    query: String,
}

/// Adapts any `Runnable<String, Vec<Document>>` into a `Tool`.
///
/// Output: `[{id, content, metadata}, ...]` as JSON. Useful when you want
/// the LLM to be able to ask "what does the corpus say about X?" without
/// hand-rolling a tool around your retriever.
pub struct RetrieverTool {
    name: String,
    description: String,
    inner: Arc<dyn Runnable<String, Vec<Document>>>,
}

impl RetrieverTool {
    /// Build a retriever tool.
    ///
    /// - `name`: how the agent addresses the tool. Use a verb-y name like
    ///   `"search_kb"`.
    /// - `description`: tells the LLM when to call it.
    /// - `inner`: the wrapped retriever.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        inner: Arc<dyn Runnable<String, Vec<Document>>>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            inner,
        }
    }
}

#[async_trait]
impl Tool for RetrieverTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn args_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::to_value(schemars::schema_for!(RetrieverInput)).unwrap_or_default())
    }
    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let parsed: RetrieverInput = serde_json::from_value(input.into_json())
            .map_err(|e| CognisError::ToolValidationError(format!("retriever_tool: {e}")))?;
        let docs = self
            .inner
            .invoke(parsed.query, RunnableConfig::default())
            .await?;
        Ok(ToolOutput::Content(serde_json::json!(docs)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StaticRetriever(Vec<Document>);
    #[async_trait]
    impl Runnable<String, Vec<Document>> for StaticRetriever {
        async fn invoke(&self, _: String, _: RunnableConfig) -> Result<Vec<Document>> {
            Ok(self.0.clone())
        }
    }

    #[tokio::test]
    async fn dispatches_query_through_retriever() {
        let r: Arc<dyn Runnable<String, Vec<Document>>> = Arc::new(StaticRetriever(vec![
            Document::new("doc1").with_id("1"),
            Document::new("doc2").with_id("2"),
        ]));
        let t = RetrieverTool::new("kb", "search KB", r);
        let mut a = std::collections::HashMap::new();
        a.insert("query".into(), serde_json::json!("hello"));
        let out = t._run(ToolInput::Structured(a)).await.unwrap();
        let v: serde_json::Value = match out {
            ToolOutput::Content(v) => v,
            _ => panic!(),
        };
        assert_eq!(v.as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn validates_input() {
        let r: Arc<dyn Runnable<String, Vec<Document>>> = Arc::new(StaticRetriever(vec![]));
        let t = RetrieverTool::new("kb", "search KB", r);
        // No `query` field.
        let a = std::collections::HashMap::new();
        let err = t._run(ToolInput::Structured(a)).await.unwrap_err();
        assert!(matches!(err, CognisError::ToolValidationError(_)));
    }
}
