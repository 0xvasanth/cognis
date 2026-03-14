//! Cross-Encoder Example
//!
//! Demonstrates cross-encoder scoring, threshold filtering, reranking,
//! caching, normalization, and LLM-generated query reranking.

#[path = "../shared.rs"]
mod shared;

use cognis_core::cross_encoders::{
    CachedCrossEncoder, CrossEncoder, CrossEncoderReranker, FakeCrossEncoder,
    NormalizedCrossEncoder, ThresholdCrossEncoder,
};
use cognis_core::documents::Document;
use cognis_core::messages::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. FakeCrossEncoder — scoring pairs
    let encoder = FakeCrossEncoder::new();
    let pairs = vec![
        (
            "rust programming language".into(),
            "rust programming language".into(),
        ),
        (
            "rust programming language".into(),
            "python scripting language".into(),
        ),
        (
            "rust programming language".into(),
            "completely unrelated xyz".into(),
        ),
    ];
    let scores = encoder.score_pairs(&pairs).await?;
    for ((a, b), score) in pairs.iter().zip(&scores) {
        println!("Score: {:.4}  ({} | {})", score, a, b);
    }

    // 2. ThresholdCrossEncoder — filtering below 0.6
    let threshold = ThresholdCrossEncoder::new(FakeCrossEncoder::new(), 0.6);
    let pairs = vec![
        ("machine learning".into(), "machine learning".into()),
        ("machine learning".into(), "deep learning models".into()),
        ("machine learning".into(), "quantum physics theory".into()),
    ];
    let scores = threshold.score_pairs(&pairs).await?;
    for ((_, b), score) in pairs.iter().zip(&scores) {
        let status = if *score > 0.0 { "PASS" } else { "FILTERED" };
        println!("Threshold: {:.4} [{}]  {}", score, status, b);
    }

    // 3. CrossEncoderReranker — reranking documents
    let reranker = CrossEncoderReranker::new(FakeCrossEncoder::new()).with_top_k(3);
    let documents = vec![
        Document::new("quantum computing and physics"),
        Document::new("rust programming language guide"),
        Document::new("introduction to machine learning"),
        Document::new("rust systems programming with cargo"),
        Document::new("web development with javascript"),
    ];
    let results = reranker.rerank("rust programming", &documents).await?;
    println!("Reranked top 3 for 'rust programming':");
    for r in &results {
        println!("  [{:.4}] {}", r.score, documents[r.index].page_content);
    }

    // 4. CachedCrossEncoder — LRU caching
    let cached = CachedCrossEncoder::new(FakeCrossEncoder::new(), 5);
    let pairs = vec![("hello world".into(), "hello world".into())];
    let s1 = cached.score_pairs(&pairs).await?;
    let s2 = cached.score_pairs(&pairs).await?;
    println!(
        "Cached: scores match={}, cache_len={}",
        s1 == s2,
        cached.cache_len()
    );

    // 5. NormalizedCrossEncoder
    let normalized = NormalizedCrossEncoder::new(FakeCrossEncoder::new());
    let pairs = vec![
        ("rust programming".into(), "rust programming".into()),
        ("rust programming".into(), "rust language".into()),
        ("rust programming".into(), "totally different xyz".into()),
    ];
    let raw = FakeCrossEncoder::new().score_pairs(&pairs).await?;
    let norm = normalized.score_pairs(&pairs).await?;
    for ((_, b), (r, n)) in pairs.iter().zip(raw.iter().zip(&norm)) {
        println!("Normalized: raw={:.4} norm={:.4}  {}", r, n, b);
    }

    // 6. LLM-generated query with reranking
    let model = shared::get_chat_model(vec![
        "What are the best practices for writing efficient Rust code?".into(),
    ]);
    let messages = vec![
        Message::system("Respond with a single search query."),
        Message::human("Generate a search query about Rust performance optimization."),
    ];
    let response = model.invoke_messages(&messages, None).await?;
    let query = response.base.content.text();
    println!("LLM query: \"{}\"", query);

    let kb = vec![
        Document::new("Rust zero-cost abstractions enable high performance"),
        Document::new("Python is popular for data science"),
        Document::new("Optimizing Rust with cargo bench and profiling"),
        Document::new("Memory safety in Rust improves efficiency"),
    ];
    let results = CrossEncoderReranker::new(FakeCrossEncoder::new())
        .with_top_k(3)
        .rerank(&query, &kb)
        .await?;
    for r in &results {
        println!("  [{:.4}] {}", r.score, kb[r.index].page_content);
    }

    Ok(())
}
