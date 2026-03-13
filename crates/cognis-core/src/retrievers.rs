//! **Retriever** trait returns `Document` objects given a text query.
//!
//! Mirrors Python `langchain_core.retrievers`.

use async_trait::async_trait;
use serde_json::Value;

use crate::documents::Document;
use crate::error::{Result, CognisError};
use crate::runnables::base::Runnable;
use crate::runnables::config::RunnableConfig;

/// Abstract base class for a document retrieval system.
///
/// A retriever takes a string query and returns relevant documents.
/// It implements `Runnable<String, Vec<Document>>` for use in LCEL chains.
#[async_trait]
pub trait BaseRetriever: Send + Sync {
    /// Retrieve relevant documents for a query.
    async fn get_relevant_documents(&self, query: &str) -> Result<Vec<Document>>;
}

/// Wrapper that makes any `BaseRetriever` usable as a `Runnable`.
///
/// Input: `Value::String` (the query)
/// Output: `Value::Array` of serialized `Document` objects
pub struct RetrieverRunnable<R: BaseRetriever> {
    retriever: R,
}

impl<R: BaseRetriever> RetrieverRunnable<R> {
    pub fn new(retriever: R) -> Self {
        Self { retriever }
    }
}

#[async_trait]
impl<R: BaseRetriever + 'static> Runnable for RetrieverRunnable<R> {
    fn name(&self) -> &str {
        "Retriever"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let query = input.as_str().ok_or_else(|| CognisError::TypeMismatch {
            expected: "String".into(),
            got: format!("{}", input),
        })?;
        let docs = self.retriever.get_relevant_documents(query).await?;
        serde_json::to_value(&docs).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct MockRetriever {
        docs: Vec<Document>,
    }

    #[async_trait]
    impl BaseRetriever for MockRetriever {
        async fn get_relevant_documents(&self, _query: &str) -> Result<Vec<Document>> {
            Ok(self.docs.clone())
        }
    }

    #[tokio::test]
    async fn test_retriever_get_documents() {
        let retriever = MockRetriever {
            docs: vec![Document {
                page_content: "Hello world".into(),
                metadata: HashMap::new(),
                id: None,
                doc_type: None,
            }],
        };
        let docs = retriever.get_relevant_documents("test").await.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "Hello world");
    }

    #[tokio::test]
    async fn test_retriever_runnable() {
        let retriever = MockRetriever {
            docs: vec![Document {
                page_content: "Result doc".into(),
                metadata: HashMap::new(),
                id: None,
                doc_type: None,
            }],
        };
        let runnable = RetrieverRunnable::new(retriever);
        let result = runnable
            .invoke(Value::String("query".into()), None)
            .await
            .unwrap();
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["page_content"], "Result doc");
    }

    #[tokio::test]
    async fn test_retriever_runnable_type_error() {
        let retriever = MockRetriever { docs: vec![] };
        let runnable = RetrieverRunnable::new(retriever);
        let result = runnable.invoke(Value::Number(42.into()), None).await;
        assert!(result.is_err());
    }
}
