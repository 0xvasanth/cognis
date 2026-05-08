//! What you'll learn:
//!   Two-stage retrieval: a fast vector recall over many candidates,
//!   then a slower `CrossEncoder` that asks the LLM to score each
//!   (query, doc) pair and keeps only the top few.
//!
//! Why this matters:
//!   Vector similarity catches "near-enough" matches but routinely
//!   misranks them. A cross-encoder pass — even a small one — fixes
//!   the ordering on a tractable subset, which is the standard
//!   recipe for production-grade retrieval quality.
//!
//! Scenario:
//!   A small product-docs corpus. The user asks "how do I make my
//!   Cognis chain run faster?". The vector retriever surfaces ten
//!   plausible candidates (some only loosely related); the LLM-judged
//!   reranker keeps the top 3 — the ones a human reviewer would also
//!   pick.
//!
//! Run with:
//!   COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.1 \
//!     cargo run -p cognis-examples --example retrieval_reranking
//!
//! Sample output (against ollama / llama3.1):
//!   === top 3 reranked ===
//!     1. Use `with_max_concurrency` on ToolOrchestrator to fan out independent tool calls.
//!     2. Streaming mode reduces perceived latency for long replies.
//!     3. Rate-limit middleware prevents runaway LLM cost.

use std::sync::Arc;

use async_trait::async_trait;
use cognis::prelude::*;
use cognis_rag::{
    CrossEncoder, CrossEncoderReranker, Document, Embeddings, FakeEmbeddings, InMemoryVectorStore,
    VectorRetriever, VectorStore,
};
use tokio::sync::RwLock;

/// LLM-judged cross-encoder: scores each candidate by asking the
/// model to rate its relevance on a 0-10 scale. In production you'd
/// batch the calls or use a dedicated scorer model — this shape is
/// the simplest version that demonstrates the pattern.
struct LlmJudge {
    client: Client,
}

#[async_trait]
impl CrossEncoder for LlmJudge {
    async fn score(&self, query: &str, docs: &[Document]) -> Result<Vec<f32>> {
        let mut scores = Vec::with_capacity(docs.len());
        for d in docs {
            let prompt = format!(
                "On a scale of 0.0 to 10.0, how well does this snippet \
                 answer the user's question? Reply with just the number.\n\n\
                 Question: {query}\nSnippet: {}",
                d.content
            );
            let resp = self.client.invoke(vec![Message::human(prompt)]).await?;
            let s = resp
                .content()
                .split_whitespace()
                .next()
                .and_then(|w| w.parse::<f32>().ok())
                .unwrap_or(0.0);
            scores.push(s);
        }
        Ok(scores)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = Client::from_env()?;

    let emb: Arc<dyn Embeddings> = Arc::new(FakeEmbeddings::new(32));
    let store = Arc::new(RwLock::new(InMemoryVectorStore::new(emb)));
    {
        let mut s = store.write().await;
        s.add_texts(
            vec![
                "Use `with_max_concurrency` on ToolOrchestrator to fan out independent tool calls."
                    .into(),
                "Cognis chains are typed; the compiler verifies stage I/O.".into(),
                "Streaming mode reduces perceived latency for long replies.".into(),
                "Caching retriever memoises identical queries — drops embed cost on re-asks."
                    .into(),
                "Window memory caps history at N turns; cheaper than Buffer.".into(),
                "Pre-warm your provider with a health_check before traffic spikes.".into(),
                "Rate-limit middleware prevents runaway LLM cost.".into(),
                "Choose a smaller model for short prompts via RoutingProvider.".into(),
                "Use Calculator tool instead of asking the LLM to do arithmetic.".into(),
                "Index incrementally — only re-embed changed docs.".into(),
            ],
            None,
        )
        .await?;
    }

    // Stage 1: vector recall pulls 10 plausible candidates.
    let recall: Arc<dyn Runnable<String, Vec<Document>>> =
        Arc::new(VectorRetriever::new(store, 10));

    // Stage 2: the LLM judges each (query, candidate) pair, keep top 3.
    let encoder: Arc<dyn CrossEncoder> = Arc::new(LlmJudge { client });
    let reranker = CrossEncoderReranker::new(recall, encoder, 3);

    let docs = reranker
        .invoke(
            "how do I make my Cognis chain run faster?".into(),
            Default::default(),
        )
        .await?;

    println!("=== top 3 reranked ===");
    for (i, d) in docs.iter().enumerate() {
        println!("  {}. {}", i + 1, d.content);
    }
    Ok(())
}
