//! What you'll learn:
//!   The full RAG flow end-to-end: split documents into chunks, embed
//!   them, store in a vector index, retrieve the top-k for a query,
//!   then feed the context into an LLM call.
//!
//! Why this matters:
//!   This is the canonical RAG pattern every Cognis user will reach
//!   for. The pieces — splitter, embeddings, vector store, retriever,
//!   client — are all swappable behind their traits, so swapping
//!   `FakeEmbeddings` for `OllamaEmbeddings` or in-memory for sqlite
//!   is a one-line change.
//!
//! Scenario:
//!   Three short docs describe Cognis. We chunk, embed, and index them,
//!   then ask "What does cognis-rag include?". The retriever finds the
//!   matching chunk and the LLM answers grounded in only that context.
//!
//! Run with:
//!   COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.1 \
//!     cargo run -p cognis-examples --example retrieval_rag_pipeline
//!
//! Sample output (against ollama / llama3.1):
//!   --- context ---
//!   - Cognis is a Rust LLM framework.
//!   - cognisgraph offers a Pregel-style stateful graph engine.
//!   --- answer ---
//!   cognis-rag includes a relational algebra layer.

use std::sync::Arc;

use cognis::prelude::*;
use cognis_rag::{
    Document, Embeddings, FakeEmbeddings, InMemoryVectorStore, RecursiveCharSplitter, TextSplitter,
    VectorStore,
};

#[tokio::main]
async fn main() -> Result<()> {
    let docs = vec![
        Document::new("Cognis is a Rust LLM framework."),
        Document::new("cognisgraph offers a Pregel-style stateful graph engine."),
        Document::new("cognis-rag bundles embeddings, vector stores, and retrievers."),
    ];
    let chunks = RecursiveCharSplitter::new()
        .with_chunk_size(120)
        .split_all(&docs);

    let emb: Arc<dyn Embeddings> = Arc::new(FakeEmbeddings::new(32));
    let mut store = InMemoryVectorStore::new(emb);
    let texts: Vec<_> = chunks.iter().map(|c| c.content.clone()).collect();
    store.add_texts(texts, None).await?;

    let q = "What does cognis-rag include?";
    let hits = store.similarity_search(q, 2).await?;
    let context: String = hits
        .iter()
        .map(|h| format!("- {}", h.text))
        .collect::<Vec<_>>()
        .join("\n");

    let client = Client::from_env()?;
    let prompt = format!("Answer using only:\n{context}\n\nQ: {q}\nA:");
    let resp = client.invoke(vec![Message::human(prompt)]).await?;
    println!(
        "--- context ---\n{context}\n--- answer ---\n{}",
        resp.content()
    );
    Ok(())
}
