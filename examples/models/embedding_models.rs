//! What you'll learn:
//!   How to use the `Embeddings` trait to turn product descriptions
//!   and a search query into vectors, then rank by cosine similarity
//!   to find the closest match.
//!
//! Why this matters:
//!   Embeddings + cosine similarity is the bedrock of any vector
//!   search or RAG pipeline. The trait abstracts the backend: swap
//!   `OllamaEmbeddings` for `OpenAIEmbeddings` or `VoyageEmbeddings`
//!   later and the ranking code never changes.
//!
//! Scenario:
//!   You have three product descriptions in a tiny catalogue:
//!   waterproof hiking boots, a ceramic coffee mug, and a wireless
//!   ergonomic keyboard. A shopper searches for "something for typing
//!   all day" — the keyboard should rank top on semantic similarity.
//!
//! Run with:
//!   COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.1 \
//!     cargo run -p cognis-examples --example models_embedding
//!
//!   Requires `ollama pull nomic-embed-text` for the embedder.
//!
//! Sample output (against ollama / nomic-embed-text):
//!   query: "something for typing all day"
//!     1. score=0.705  Wireless ergonomic keyboard — split layout, mechanical switches, designed for all-day typing.
//!     2. score=0.356  Waterproof hiking boots — full-grain leather, vibram sole, ankle support for rough trails.
//!     3. score=0.334  Ceramic coffee mug — 12oz, dishwasher safe, glossy navy glaze.

use std::sync::Arc;

use cognis::prelude::*;
use cognis_rag::{Embeddings, OllamaEmbeddings};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Real Ollama embeddings. `nomic-embed-text` is small (~270 MB) and
    // semantically ranks similar text correctly. The Ollama daemon must
    // be running with that model pulled.
    let emb: Arc<dyn Embeddings> = Arc::new(OllamaEmbeddings::new("nomic-embed-text"));

    let products = [
        "Waterproof hiking boots — full-grain leather, vibram sole, ankle support for rough trails.",
        "Ceramic coffee mug — 12oz, dishwasher safe, glossy navy glaze.",
        "Wireless ergonomic keyboard — split layout, mechanical switches, designed for all-day typing.",
    ];
    let query = "something for typing all day";

    // One batch call for the catalogue (cheaper than per-doc roundtrips
    // — this is why production code keeps a list and calls
    // `embed_documents` once).
    let doc_vecs = emb
        .embed_documents(products.iter().map(|s| s.to_string()).collect())
        .await?;
    let q_vec = emb.embed_query(query.to_string()).await?;

    let mut scored: Vec<(usize, f32)> = doc_vecs
        .iter()
        .enumerate()
        .map(|(i, v)| (i, cosine(&q_vec, v)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("query: {query:?}");
    for (rank, (idx, score)) in scored.iter().enumerate() {
        println!("  {}. score={:.3}  {}", rank + 1, score, products[*idx]);
    }
    Ok(())
}
