//! Translate the query before retrieval.
//!
//! Common shape: take a free-text query, run it through a translator
//! (rephraser, language-converter, structurer), then retrieve. Already
//! expressible via `Runnable::pipe`, but a named type aids discovery.

use std::sync::Arc;

use async_trait::async_trait;

use cognis2_core::{Result, Runnable, RunnableConfig};

use crate::document::Document;

/// `query → translator → retriever`.
pub struct QueryTranslatorRetriever {
    translator: Arc<dyn Runnable<String, String>>,
    inner: Arc<dyn Runnable<String, Vec<Document>>>,
}

impl QueryTranslatorRetriever {
    /// Build with a translator and an inner retriever.
    pub fn new(
        translator: Arc<dyn Runnable<String, String>>,
        inner: Arc<dyn Runnable<String, Vec<Document>>>,
    ) -> Self {
        Self { translator, inner }
    }
}

#[async_trait]
impl Runnable<String, Vec<Document>> for QueryTranslatorRetriever {
    async fn invoke(&self, query: String, config: RunnableConfig) -> Result<Vec<Document>> {
        let translated = self.translator.invoke(query, config.clone()).await?;
        self.inner.invoke(translated, config).await
    }
    fn name(&self) -> &str {
        "QueryTranslatorRetriever"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct UpperCase;
    #[async_trait]
    impl Runnable<String, String> for UpperCase {
        async fn invoke(&self, q: String, _: RunnableConfig) -> Result<String> {
            Ok(q.to_uppercase())
        }
    }

    struct EchoRetriever;
    #[async_trait]
    impl Runnable<String, Vec<Document>> for EchoRetriever {
        async fn invoke(&self, q: String, _: RunnableConfig) -> Result<Vec<Document>> {
            Ok(vec![Document::new(q)])
        }
    }

    #[tokio::test]
    async fn translates_then_retrieves() {
        let r = QueryTranslatorRetriever::new(Arc::new(UpperCase), Arc::new(EchoRetriever));
        let out = r
            .invoke("hello".into(), RunnableConfig::default())
            .await
            .unwrap();
        assert_eq!(out[0].content, "HELLO");
    }
}
