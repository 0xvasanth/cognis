//! `SelfQueryRetriever` — uses an LLM to parse a free-text query into a
//! `(semantic_query, filter)` pair, then runs the filtered search.
//!
//! Rust-native take: instead of V1's bespoke "structured query AST +
//! parser + visitor", we lean on V2's existing
//! `Client::with_structured_output<T>` to deserialize directly into a
//! [`SearchSpec`]. The LLM emits JSON that matches the schema; serde does
//! the rest.

use std::sync::Arc;

use async_trait::async_trait;
use cognis_core::schemars::{self, JsonSchema};
use serde::Deserialize;
use tokio::sync::RwLock;

use cognis_core::{Message, Result, Runnable, RunnableConfig};
use cognis_llm::Client;
use cognis_rag::{Document, Filter, VectorStore};

/// Structured search spec the LLM emits.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchSpec {
    /// The semantic part of the query — what to embed and similarity-search.
    pub semantic_query: String,
    /// Metadata equality filters (`{"key": "value"}` pairs).
    #[serde(default)]
    pub filter_equals: std::collections::HashMap<String, serde_json::Value>,
}

const DEFAULT_PROMPT: &str =
    "You translate a user query into a JSON search spec. Output ONLY JSON \
     matching the schema. The `semantic_query` is what to search for; \
     `filter_equals` carries any metadata constraints the user expressed \
     (e.g. \"papers from 2024 about rust\" → \
     `filter_equals: {\"year\": 2024}`). When no metadata is implied, \
     leave `filter_equals` empty.\n\n\
     User query: {query}";

/// Retriever that uses an LLM to extract a metadata filter from the query
/// before running similarity search.
pub struct SelfQueryRetriever {
    store: Arc<RwLock<dyn VectorStore>>,
    client: Client,
    k: usize,
    prompt: String,
}

impl SelfQueryRetriever {
    /// Build with a vector store, an LLM client (for query parsing), and a `k`.
    pub fn new(store: Arc<RwLock<dyn VectorStore>>, client: Client, k: usize) -> Self {
        Self {
            store,
            client,
            k,
            prompt: DEFAULT_PROMPT.to_string(),
        }
    }

    /// Override the parsing prompt. Use `{query}` as the placeholder.
    pub fn with_prompt(mut self, p: impl Into<String>) -> Self {
        self.prompt = p.into();
        self
    }

    async fn parse(&self, query: &str) -> Result<SearchSpec> {
        let prompt = self.prompt.replace("{query}", query);
        // Use the structured-output adapter — it injects schema instructions
        // and parses the reply.
        let parser = self.client.clone().with_structured_output::<SearchSpec>();
        let cfg = RunnableConfig::default();
        parser.invoke(vec![Message::human(prompt)], cfg).await
    }
}

#[async_trait]
impl Runnable<String, Vec<Document>> for SelfQueryRetriever {
    async fn invoke(&self, query: String, _config: RunnableConfig) -> Result<Vec<Document>> {
        let spec = self.parse(&query).await?;
        let mut filter = Filter::new();
        for (k, v) in spec.filter_equals {
            filter = filter.equals(k, v);
        }
        let hits = self
            .store
            .read()
            .await
            .similarity_search_with_filter(&spec.semantic_query, self.k, &filter)
            .await?;
        Ok(hits
            .into_iter()
            .map(|h| Document {
                id: Some(h.id),
                content: h.text,
                metadata: h.metadata,
            })
            .collect())
    }
    fn name(&self) -> &str {
        "SelfQueryRetriever"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use async_trait::async_trait;
    use cognis_core::{Message, Result, RunnableStream};
    use cognis_llm::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk, Usage};
    use cognis_llm::provider::{LLMProvider, Provider};
    use cognis_rag::{FakeEmbeddings, InMemoryVectorStore};

    /// Provider that always returns the same canned spec JSON.
    struct StaticSpec(String);
    #[async_trait]
    impl LLMProvider for StaticSpec {
        fn name(&self) -> &str {
            "static-spec"
        }
        fn provider_type(&self) -> Provider {
            Provider::Ollama
        }
        async fn chat_completion(&self, _: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message::ai(self.0.clone()),
                usage: Some(Usage::default()),
                finish_reason: "stop".into(),
                model: "static-spec".into(),
            })
        }
        async fn chat_completion_stream(
            &self,
            _: Vec<Message>,
            _: ChatOptions,
        ) -> Result<RunnableStream<StreamChunk>> {
            unimplemented!()
        }
        async fn health_check(&self) -> Result<HealthStatus> {
            Ok(HealthStatus::Healthy { latency_ms: 0 })
        }
    }

    #[tokio::test]
    async fn parses_filter_and_applies_it() {
        let mut store = InMemoryVectorStore::new(Arc::new(FakeEmbeddings::new(8)));
        let mut m_a: HashMap<String, serde_json::Value> = HashMap::new();
        m_a.insert("year".into(), serde_json::json!(2023));
        let mut m_b: HashMap<String, serde_json::Value> = HashMap::new();
        m_b.insert("year".into(), serde_json::json!(2024));
        store
            .add_texts(
                vec!["rust paper one".into(), "rust paper two".into()],
                Some(vec![m_a, m_b]),
            )
            .await
            .unwrap();
        let store_arc: Arc<RwLock<dyn VectorStore>> = Arc::new(RwLock::new(store));

        let spec_json = r#"{"semantic_query":"rust","filter_equals":{"year":2024}}"#;
        let client = Client::new(Arc::new(StaticSpec(spec_json.into())));
        let r = SelfQueryRetriever::new(store_arc, client, 5);

        let out = r
            .invoke("rust papers from 2024".into(), RunnableConfig::default())
            .await
            .unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].content.contains("two"));
    }
}
