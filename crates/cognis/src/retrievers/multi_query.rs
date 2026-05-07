//! `MultiQueryRetriever` — ask the LLM to rephrase the query N ways, run
//! the inner retriever for each rephrasing, and union (dedupe by id, keep
//! best rank).

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use cognis_core::{Message, Result, Runnable, RunnableConfig};
use cognis_llm::chat::ChatOptions;
use cognis_llm::Client;
use cognis_rag::Document;

const DEFAULT_PROMPT: &str =
    "You are a query rephraser. Rephrase the user's question {n} different \
     ways that surface different relevant aspects. Output ONLY the rephrased \
     queries, one per line. No numbering, no commentary. Original: {query}";

/// Rephrases queries via an LLM, runs the inner retriever per rephrasing,
/// and merges the results.
pub struct MultiQueryRetriever {
    inner: Arc<dyn Runnable<String, Vec<Document>>>,
    client: Client,
    n: usize,
    prompt: String,
}

impl MultiQueryRetriever {
    /// Wrap a retriever with a multi-query expander.
    pub fn new(inner: Arc<dyn Runnable<String, Vec<Document>>>, client: Client, n: usize) -> Self {
        Self {
            inner,
            client,
            n,
            prompt: DEFAULT_PROMPT.to_string(),
        }
    }

    /// Override the prompt template. Available placeholders: `{n}`, `{query}`.
    pub fn with_prompt(mut self, p: impl Into<String>) -> Self {
        self.prompt = p.into();
        self
    }

    fn render_prompt(&self, query: &str) -> String {
        self.prompt
            .replace("{n}", &self.n.to_string())
            .replace("{query}", query)
    }

    async fn rephrase(&self, query: &str) -> Result<Vec<String>> {
        let prompt = self.render_prompt(query);
        let resp = self
            .client
            .chat(vec![Message::human(prompt)], ChatOptions::default())
            .await?;
        let text = resp.message.content().to_string();
        let queries: Vec<String> = text
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .take(self.n)
            .collect();
        // Always include the original.
        let mut out = vec![query.to_string()];
        out.extend(queries);
        Ok(out)
    }
}

#[async_trait]
impl Runnable<String, Vec<Document>> for MultiQueryRetriever {
    async fn invoke(&self, query: String, config: RunnableConfig) -> Result<Vec<Document>> {
        let queries = self.rephrase(&query).await?;
        let mut seen: HashMap<String, Document> = HashMap::new();
        let mut order: Vec<String> = Vec::new();
        for q in queries {
            let docs = self.inner.invoke(q, config.clone()).await?;
            for d in docs {
                let key = d.id.clone().unwrap_or_else(|| d.content.clone());
                seen.entry(key.clone()).or_insert_with(|| {
                    order.push(key.clone());
                    d
                });
            }
        }
        Ok(order.into_iter().filter_map(|k| seen.remove(&k)).collect())
    }

    fn name(&self) -> &str {
        "MultiQueryRetriever"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Mutex;

    use cognis_core::{Message, Result, RunnableStream};
    use cognis_llm::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk, Usage};
    use cognis_llm::provider::{LLMProvider, Provider};

    struct StaticInner {
        per_query: Mutex<HashMap<String, Vec<Document>>>,
    }
    #[async_trait]
    impl Runnable<String, Vec<Document>> for StaticInner {
        async fn invoke(&self, q: String, _: RunnableConfig) -> Result<Vec<Document>> {
            Ok(self
                .per_query
                .lock()
                .unwrap()
                .get(&q)
                .cloned()
                .unwrap_or_default())
        }
    }

    struct LinesProvider(String);
    #[async_trait]
    impl LLMProvider for LinesProvider {
        fn name(&self) -> &str {
            "lines"
        }
        fn provider_type(&self) -> Provider {
            Provider::Ollama
        }
        async fn chat_completion(
            &self,
            _messages: Vec<Message>,
            _opts: ChatOptions,
        ) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message::ai(self.0.clone()),
                usage: Some(Usage::default()),
                finish_reason: "stop".into(),
                model: "lines".into(),
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
    async fn unions_results_across_rephrasings() {
        let mut per_query = HashMap::new();
        per_query.insert(
            "rust ownership".to_string(),
            vec![Document::new("a").with_id("a")],
        );
        per_query.insert(
            "memory safety in rust".to_string(),
            vec![
                Document::new("b").with_id("b"),
                Document::new("a").with_id("a"),
            ],
        );
        per_query.insert(
            "borrow checker".to_string(),
            vec![Document::new("c").with_id("c")],
        );
        let inner: Arc<dyn Runnable<String, Vec<Document>>> = Arc::new(StaticInner {
            per_query: Mutex::new(per_query),
        });

        let provider = Arc::new(LinesProvider(
            "memory safety in rust\nborrow checker".to_string(),
        ));
        let client = Client::new(provider);
        let mq = MultiQueryRetriever::new(inner, client, 2);
        let docs = mq
            .invoke("rust ownership".to_string(), RunnableConfig::default())
            .await
            .unwrap();
        let ids: Vec<_> = docs.iter().filter_map(|d| d.id.clone()).collect();
        // Original query yields `a` first, then dedupe across `b`, `c`.
        assert_eq!(ids, vec!["a", "b", "c"]);
    }
}
