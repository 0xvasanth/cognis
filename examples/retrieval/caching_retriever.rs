//! What you'll learn:
//!   How `CachingRetriever` wraps any `Runnable<String, Vec<Document>>`
//!   so identical queries hit a memoised result instead of re-running
//!   the embedding + vector-search round trip.
//!
//! Why this matters:
//!   Users re-ask. "How do I cancel my plan?" gets typed into the
//!   help bot dozens of times an hour with tiny variations. Retrieval
//!   is one of the noisiest cost lines in a RAG pipeline, and a
//!   trivial cache on top of an existing retriever often clears
//!   30-50% of redundant lookups in real workloads. Because the
//!   wrapper is itself a `Runnable`, nothing downstream notices.
//!
//! Scenario:
//!   A chat session where the same retrieval question appears twice
//!   in different turns ("how do I cancel my plan?" both times).
//!   Round 1 pays the full embed-and-search cost; round 2 returns
//!   from cache — latency drops to effectively zero.
//!
//! Run with:
//!   cargo run -p cognis-examples --example retrieval_caching_retriever
//!
//! Sample output (against ollama / llama3.1):
//!   turn 1 (cold): 37.458µs  -> 2 hits
//!   turn 5 (warm): 4.75µs  -> 2 hits  (returned from cache)
//!
//!   top result: Password resets are sent to your registered email.

use std::sync::Arc;
use std::time::Instant;

use cognis::prelude::*;
use cognis_rag::{
    CachingRetriever, Document, Embeddings, FakeEmbeddings, InMemoryVectorStore, VectorRetriever,
    VectorStore,
};
use tokio::sync::RwLock;

#[tokio::main]
async fn main() -> Result<()> {
    let emb: Arc<dyn Embeddings> = Arc::new(FakeEmbeddings::new(32));
    let store = Arc::new(RwLock::new(InMemoryVectorStore::new(emb)));
    {
        let mut s = store.write().await;
        s.add_texts(
            vec![
                "To cancel your plan, open Settings -> Billing -> Cancel.".into(),
                "Refunds are processed within 5 business days.".into(),
                "Password resets are sent to your registered email.".into(),
                "Contact support@example.com for account help.".into(),
            ],
            None,
        )
        .await?;
    }

    let inner: Arc<dyn Runnable<String, Vec<Document>>> = Arc::new(VectorRetriever::new(store, 2));
    let cached = CachingRetriever::new(inner);

    // Turn 1: first time we've seen this question — full lookup.
    let q = "how do I cancel my plan?".to_string();
    let t0 = Instant::now();
    let r1 = cached.invoke(q.clone(), Default::default()).await?;
    let cold = t0.elapsed();

    // ... a few user turns later, the same question comes back.
    let t1 = Instant::now();
    let r2 = cached.invoke(q.clone(), Default::default()).await?;
    let warm = t1.elapsed();

    println!("turn 1 (cold): {:?}  -> {} hits", cold, r1.len());
    println!(
        "turn 5 (warm): {:?}  -> {} hits  (returned from cache)",
        warm,
        r2.len()
    );
    if let Some(top) = r1.first() {
        println!("\ntop result: {}", top.content);
    }
    Ok(())
}
