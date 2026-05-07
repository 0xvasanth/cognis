//! `RerankingRetriever` — LLM-driven cross-encoder rerank of an inner
//! retriever's hits.
//!
//! Asks the LLM to score each candidate against the query and returns
//! them sorted by score, top-k. Useful when the inner retriever is fast
//! but coarse (BM25, naive vector) and you want a more expensive but
//! more accurate ranker on top.

use std::sync::Arc;

use async_trait::async_trait;
use futures::future::join_all;

use cognis_core::{Message, Result, Runnable, RunnableConfig};
use cognis_llm::chat::ChatOptions;
use cognis_llm::Client;
use cognis_rag::Document;

const DEFAULT_PROMPT: &str =
    "Rate how relevant the passage is to the query on a scale 0-10. Reply \
     with ONLY a single integer, no commentary.\n\nQuery: {query}\n\nPassage: {passage}";

/// Rerank an inner retriever's hits by LLM-judged relevance score.
pub struct RerankingRetriever {
    inner: Arc<dyn Runnable<String, Vec<Document>>>,
    client: Client,
    top_k: usize,
    prompt: String,
}

impl RerankingRetriever {
    /// Wrap an inner retriever. `top_k` is the post-rerank cut.
    pub fn new(
        inner: Arc<dyn Runnable<String, Vec<Document>>>,
        client: Client,
        top_k: usize,
    ) -> Self {
        Self {
            inner,
            client,
            top_k,
            prompt: DEFAULT_PROMPT.to_string(),
        }
    }

    /// Override the scoring prompt. Placeholders: `{query}`, `{passage}`.
    pub fn with_prompt(mut self, p: impl Into<String>) -> Self {
        self.prompt = p.into();
        self
    }

    async fn score(&self, query: &str, passage: &str) -> Result<f32> {
        let prompt = self
            .prompt
            .replace("{query}", query)
            .replace("{passage}", passage);
        let resp = self
            .client
            .chat(vec![Message::human(prompt)], ChatOptions::default())
            .await?;
        let text = resp.message.content().trim().to_string();
        // Pull the first integer or float from the reply — robust to
        // models that prepend "Score: " or similar.
        let mut num = String::new();
        for c in text.chars() {
            if c.is_ascii_digit() || c == '.' || (c == '-' && num.is_empty()) {
                num.push(c);
            } else if !num.is_empty() {
                break;
            }
        }
        Ok(num.parse::<f32>().unwrap_or(0.0))
    }
}

#[async_trait]
impl Runnable<String, Vec<Document>> for RerankingRetriever {
    async fn invoke(&self, query: String, config: RunnableConfig) -> Result<Vec<Document>> {
        let docs = self.inner.invoke(query.clone(), config).await?;
        let scoring = docs.iter().map(|d| self.score(&query, &d.content));
        let scores: Vec<f32> = join_all(scoring)
            .await
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let mut paired: Vec<(f32, Document)> = scores.into_iter().zip(docs).collect();
        paired.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(paired
            .into_iter()
            .take(self.top_k)
            .map(|(_, d)| d)
            .collect())
    }
    fn name(&self) -> &str {
        "RerankingRetriever"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use cognis_core::{Message, Result, RunnableStream};
    use cognis_llm::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk, Usage};
    use cognis_llm::provider::{LLMProvider, Provider};

    struct StaticInner(Vec<Document>);
    #[async_trait]
    impl Runnable<String, Vec<Document>> for StaticInner {
        async fn invoke(&self, _: String, _: RunnableConfig) -> Result<Vec<Document>> {
            Ok(self.0.clone())
        }
    }

    /// Returns a score derived from how many times "rust" appears.
    struct CountScorer;
    #[async_trait]
    impl LLMProvider for CountScorer {
        fn name(&self) -> &str {
            "scorer"
        }
        fn provider_type(&self) -> Provider {
            Provider::Ollama
        }
        async fn chat_completion(
            &self,
            messages: Vec<Message>,
            _: ChatOptions,
        ) -> Result<ChatResponse> {
            let prompt = messages.last().unwrap().content().to_lowercase();
            let passage = prompt
                .split_once("passage:")
                .map(|(_, r)| r.to_string())
                .unwrap_or_default();
            let n = passage.matches("rust").count();
            Ok(ChatResponse {
                message: Message::ai(n.to_string()),
                usage: Some(Usage::default()),
                finish_reason: "stop".into(),
                model: "scorer".into(),
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
    async fn reranks_by_score() {
        let inner: Arc<dyn Runnable<String, Vec<Document>>> = Arc::new(StaticInner(vec![
            Document::new("apple pie").with_id("a"),
            Document::new("rust rust rust crab").with_id("b"),
            Document::new("rust mention").with_id("c"),
        ]));
        let client = Client::new(Arc::new(CountScorer));
        let r = RerankingRetriever::new(inner, client, 2);
        let out = r
            .invoke("rust ranking".into(), RunnableConfig::default())
            .await
            .unwrap();
        let ids: Vec<_> = out.iter().filter_map(|d| d.id.clone()).collect();
        // `b` has 3 mentions, `c` has 1, `a` has 0 → top-2 = [b, c].
        assert_eq!(ids, vec!["b", "c"]);
    }
}
