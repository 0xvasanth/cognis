//! `ContextualCompressionRetriever` — score each retrieved doc with the LLM
//! and drop low-relevance ones.
//!
//! For each candidate doc the inner retriever returned, ask the LLM:
//! "Given the query, is this passage relevant? Reply yes or no." Drop the
//! `no`s. Order is preserved among the kept documents.

use std::sync::Arc;

use async_trait::async_trait;
use futures::future::join_all;

use cognis2_core::{Message, Result, Runnable, RunnableConfig};
use cognis2_llm::chat::ChatOptions;
use cognis2_llm::Client;
use cognis2_rag::Document;

const DEFAULT_PROMPT: &str =
    "Decide if the passage is relevant to the query. Reply with exactly one \
     word: `yes` or `no`. No explanation.\n\nQuery: {query}\n\nPassage: {passage}";

/// Filters retrieved documents by an LLM relevance check.
pub struct ContextualCompressionRetriever {
    inner: Arc<dyn Runnable<String, Vec<Document>>>,
    client: Client,
    prompt: String,
}

impl ContextualCompressionRetriever {
    /// Wrap an inner retriever with an LLM-driven filter.
    pub fn new(inner: Arc<dyn Runnable<String, Vec<Document>>>, client: Client) -> Self {
        Self {
            inner,
            client,
            prompt: DEFAULT_PROMPT.to_string(),
        }
    }

    /// Override the relevance prompt. Placeholders: `{query}`, `{passage}`.
    pub fn with_prompt(mut self, p: impl Into<String>) -> Self {
        self.prompt = p.into();
        self
    }

    fn render(&self, query: &str, passage: &str) -> String {
        self.prompt
            .replace("{query}", query)
            .replace("{passage}", passage)
    }

    async fn keep(&self, query: &str, passage: &str) -> Result<bool> {
        let prompt = self.render(query, passage);
        let resp = self
            .client
            .chat(vec![Message::human(prompt)], ChatOptions::default())
            .await?;
        let answer = resp.message.content().trim().to_lowercase();
        Ok(matches!(answer.as_str(), "yes" | "y" | "true"))
    }
}

#[async_trait]
impl Runnable<String, Vec<Document>> for ContextualCompressionRetriever {
    async fn invoke(&self, query: String, config: RunnableConfig) -> Result<Vec<Document>> {
        let docs = self.inner.invoke(query.clone(), config).await?;
        let checks = docs.iter().map(|d| self.keep(&query, &d.content));
        let verdicts: Vec<bool> = join_all(checks)
            .await
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        Ok(docs
            .into_iter()
            .zip(verdicts)
            .filter_map(|(d, keep)| if keep { Some(d) } else { None })
            .collect())
    }

    fn name(&self) -> &str {
        "ContextualCompressionRetriever"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use cognis2_core::{Message, Result, RunnableStream};
    use cognis2_llm::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk, Usage};
    use cognis2_llm::provider::{LLMProvider, Provider};

    struct StaticInner(Vec<Document>);
    #[async_trait]
    impl Runnable<String, Vec<Document>> for StaticInner {
        async fn invoke(&self, _q: String, _: RunnableConfig) -> Result<Vec<Document>> {
            Ok(self.0.clone())
        }
    }

    /// Per-substring scoring provider — returns "yes" for any passage whose
    /// content contains a configured trigger; "no" otherwise.
    struct KeywordJudge {
        keep_if_contains: Vec<String>,
    }
    #[async_trait]
    impl LLMProvider for KeywordJudge {
        fn name(&self) -> &str {
            "judge"
        }
        fn provider_type(&self) -> Provider {
            Provider::Ollama
        }
        async fn chat_completion(
            &self,
            messages: Vec<Message>,
            _opts: ChatOptions,
        ) -> Result<ChatResponse> {
            // Score against just the passage text; the prompt body always
            // contains the query so we'd otherwise match every prompt.
            let prompt = messages
                .last()
                .map(|m| m.content().to_string())
                .unwrap_or_default();
            let passage = prompt
                .split_once("Passage:")
                .map(|(_, rest)| rest.to_lowercase())
                .unwrap_or_else(|| prompt.to_lowercase());
            let answer = if self
                .keep_if_contains
                .iter()
                .any(|kw| passage.contains(&kw.to_lowercase()))
            {
                "yes"
            } else {
                "no"
            };
            Ok(ChatResponse {
                message: Message::ai(answer),
                usage: Some(Usage::default()),
                finish_reason: "stop".into(),
                model: "judge".into(),
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
    async fn drops_irrelevant_docs() {
        let inner: Arc<dyn Runnable<String, Vec<Document>>> = Arc::new(StaticInner(vec![
            Document::new("rust ownership rules").with_id("a"),
            Document::new("a recipe for sourdough").with_id("b"),
            Document::new("rust borrow checker").with_id("c"),
        ]));
        let client = Client::new(Arc::new(KeywordJudge {
            keep_if_contains: vec!["rust".to_string()],
        }));
        let r = ContextualCompressionRetriever::new(inner, client);
        let out = r
            .invoke("rust safety".to_string(), RunnableConfig::default())
            .await
            .unwrap();
        let ids: Vec<_> = out.iter().filter_map(|d| d.id.clone()).collect();
        assert_eq!(ids, vec!["a", "c"]);
    }
}
