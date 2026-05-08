//! What you'll learn:
//!   The full RAG flow end-to-end: split documents into chunks, embed
//!   them, store in a vector index, retrieve the top-k for a query,
//!   then feed the context into an LLM call.
//!
//! Why this matters:
//!   This is the canonical RAG pattern every Cognis user will reach
//!   for. The pieces — splitter, embeddings, vector store, retriever,
//!   client — are all swappable behind their traits. Swap
//!   `OllamaEmbeddings` for `OpenAIEmbeddings`, or in-memory for
//!   FAISS / Qdrant / Pinecone — one-line changes.
//!
//! Scenario:
//!   Three short docs describe Cognis. We chunk, embed (with real
//!   `nomic-embed-text` so similarity is meaningful), and index them,
//!   then ask "What does cognis-rag include?". The retriever finds the
//!   matching chunk and the LLM answers grounded in only that context.
//!
//! Run with:
//!   COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.1 \
//!     cargo run -p cognis-examples --example retrieval_rag_pipeline
//!
//!   Requires `ollama pull nomic-embed-text` for the embedder.
//!
//! Sample output (against ollama / llama3.1 + nomic-embed-text):
//!   --- context ---
//!   - cognis-rag bundles embeddings, vector stores, and retrievers.
//!   --- answer ---
//!   Cognis-RAG includes:
//!   1. Embeddings — vector representations of each query or prompt.
//!   2. Vector stores — databases optimized for storing dense vectors.
//!   3. Retrievers — algorithms that use the store to fetch the most
//!      relevant documents based on similarity scores.

use std::sync::Arc;

use cognis::prelude::*;
use cognis_rag::{
    Document, Embeddings, InMemoryVectorStore, OllamaEmbeddings, RecursiveCharSplitter,
    TextSplitter, VectorStore,
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

    // Real semantic embeddings — `nomic-embed-text` is a small (~270 MB)
    // local model that does the job for short docs. Swap to OpenAI /
    // Voyage for production quality at higher latency + cost.
    let emb: Arc<dyn Embeddings> = Arc::new(OllamaEmbeddings::new("nomic-embed-text"));
    let mut store = InMemoryVectorStore::new(emb);
    let texts: Vec<_> = chunks.iter().map(|c| c.content.clone()).collect();
    store.add_texts(texts, None).await?;

    let q = "What does cognis-rag include?";
    // Top 1 — semantic search should pick the cognis-rag doc first.
    let hits = store.similarity_search(q, 1).await?;
    let context: String = hits
        .iter()
        .map(|h| format!("- {}", h.text))
        .collect::<Vec<_>>()
        .join("\n");

    // Ground the LLM in retrieved context only — no general knowledge.
    let client = Client::from_env()?;
    let prompt = format!("Answer using only:\n{context}\n\nQ: {q}\nA:");
    let resp = client.invoke(vec![Message::human(prompt)]).await?;
    println!(
        "--- context ---\n{context}\n--- answer ---\n{}",
        resp.content()
    );
    Ok(())
}
