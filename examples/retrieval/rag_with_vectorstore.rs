//! End-to-end RAG: split → embed → store → retrieve → answer.

use std::sync::Arc;

use cognis::prelude::*;
use cognis_llm::Client;
use cognis_rag::{
    Document, Embeddings, FakeEmbeddings, InMemoryVectorStore, RecursiveCharSplitter, TextSplitter,
    VectorStore,
};

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::var("COGNIS_PROVIDER").is_err() {
        std::env::set_var("COGNIS_PROVIDER", "ollama");
    }
    let docs = vec![
        Document::new("Cognis is a Rust LLM framework."),
        Document::new("cognisgraph offers a StateGraph engine inspired by LangGraph."),
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
