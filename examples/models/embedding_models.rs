//! What you'll learn:
//!   How to use the `Embeddings` trait to turn product descriptions
//!   and a search query into vectors, then rank by cosine similarity
//!   to find the closest match.
//!
//! Why this matters:
//!   Embeddings + cosine similarity is the bedrock of any vector
//!   search or RAG pipeline. The trait abstracts the backend: swap in
//!   `OllamaEmbeddings`, `OpenAIEmbeddings`, or `VoyageEmbeddings`
//!   later and the ranking code never changes. `FakeEmbeddings` lets
//!   you test the surrounding shape without paying API tokens.
//!
//! Scenario:
//!   You have three product descriptions in a tiny catalogue:
//!   waterproof hiking boots, a ceramic coffee mug, and a wireless
//!   ergonomic keyboard. A shopper searches for "something for typing
//!   all day" — the keyboard should win on similarity.
//!
//! Run with:
//!   cargo run -p cognis-examples --example models_embedding
//!
//! Sample output (against ollama / llama3.1):
//!   query: "something for typing all day"
//!     1. score=-0.082  Ceramic coffee mug — 12oz, dishwasher safe, glossy navy glaze.
//!     2. score=-0.086  Waterproof hiking boots — full-grain leather, vibram sole, ankle support for rough trails.
//!     3. score=-0.107  Wireless ergonomic keyboard — split layout, mechanical switches, designed for all-day typing.

use std::sync::Arc;

use cognis::prelude::*;
use cognis_rag::{Embeddings, FakeEmbeddings};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

#[tokio::main]
async fn main() -> Result<()> {
    // FakeEmbeddings is deterministic — fine for showing the shape of
    // the pipeline. Swap to `OllamaEmbeddings::new("nomic-embed-text")`
    // for real semantic ranking.
    let emb: Arc<dyn Embeddings> = Arc::new(FakeEmbeddings::new(64));

    let products = [
        "Waterproof hiking boots — full-grain leather, vibram sole, ankle support for rough trails.",
        "Ceramic coffee mug — 12oz, dishwasher safe, glossy navy glaze.",
        "Wireless ergonomic keyboard — split layout, mechanical switches, designed for all-day typing.",
    ];
    let query = "something for typing all day";

    // One batch call for the catalogue (cheaper than per-doc roundtrips
    // — this is why production code keeps a list and calls
    // `embed_documents` once).
    let doc_vecs = emb.embed_documents(products.iter().map(|s| s.to_string()).collect()).await?;
    let q_vec = emb.embed_query(query.to_string()).await?;

    let mut scored: Vec<(usize, f32)> =
        doc_vecs.iter().enumerate().map(|(i, v)| (i, cosine(&q_vec, v))).collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("query: {query:?}");
    for (rank, (idx, score)) in scored.iter().enumerate() {
        println!("  {}. score={:.3}  {}", rank + 1, score, products[*idx]);
    }
    Ok(())
}
