//! Cross-Encoder Example
//!
//! Demonstrates the cross-encoder interface for scoring text pairs and
//! reranking search results by semantic similarity.
//!
//! Features shown:
//! - FakeCrossEncoder (deterministic character-overlap scoring)
//! - ThresholdCrossEncoder (filtering below a score threshold)
//! - CrossEncoderReranker (reranking documents by query relevance)
//! - CachedCrossEncoder (LRU-cached scoring)
//! - NormalizedCrossEncoder (min-max normalization to [0, 1])
//! - LLM-generated query with cross-encoder reranking
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example cross_encoder`

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
    println!("=== Cross-Encoder Example ===\n");

    // -----------------------------------------------------------------------
    // 1. FakeCrossEncoder — scoring pairs
    // -----------------------------------------------------------------------
    println!("--- 1. FakeCrossEncoder: scoring text pairs ---");
    let encoder = FakeCrossEncoder::new();

    let pairs = vec![
        ("rust programming language".into(), "rust programming language".into()),
        ("rust programming language".into(), "python scripting language".into()),
        ("rust programming language".into(), "completely unrelated xyz".into()),
        ("rust programming language".into(), "rust is a great language".into()),
    ];

    let scores = encoder.score_pairs(&pairs).await?;
    for (i, ((a, b), score)) in pairs.iter().zip(scores.iter()).enumerate() {
        println!("  Pair {}: ({:40} | {:40}) => score: {:.4}", i, a, b, score);
    }
    println!();

    // -----------------------------------------------------------------------
    // 2. ThresholdCrossEncoder — filtering low-scoring pairs
    // -----------------------------------------------------------------------
    println!("--- 2. ThresholdCrossEncoder: filtering below 0.6 ---");
    let threshold_encoder = ThresholdCrossEncoder::new(FakeCrossEncoder::new(), 0.6);
    println!("  Threshold: {}", threshold_encoder.threshold());

    let pairs = vec![
        ("machine learning".into(), "machine learning".into()),       // high overlap
        ("machine learning".into(), "deep learning models".into()),   // partial overlap
        ("machine learning".into(), "quantum physics theory".into()), // low overlap
    ];

    let scores = threshold_encoder.score_pairs(&pairs).await?;
    for ((_, b), score) in pairs.iter().zip(scores.iter()) {
        let status = if *score > 0.0 { "PASS" } else { "FILTERED" };
        println!("  \"{}\" => {:.4} [{}]", b, score, status);
    }
    println!();

    // -----------------------------------------------------------------------
    // 3. CrossEncoderReranker — reranking search results
    // -----------------------------------------------------------------------
    println!("--- 3. CrossEncoderReranker: reranking documents ---");
    let reranker = CrossEncoderReranker::new(FakeCrossEncoder::new()).with_top_k(3);

    let documents = vec![
        Document::new("quantum computing and physics"),
        Document::new("rust programming language guide"),
        Document::new("introduction to machine learning"),
        Document::new("rust systems programming with cargo"),
        Document::new("web development with javascript"),
    ];

    let query = "rust programming";
    println!("  Query: \"{}\"", query);
    println!("  Documents (original order):");
    for (i, doc) in documents.iter().enumerate() {
        println!("    [{}] {}", i, doc.page_content);
    }

    let results = reranker.rerank(query, &documents).await?;
    println!("  Reranked results (top 3):");
    for result in &results {
        println!(
            "    [orig_idx={}] \"{}\" => score: {:.4}",
            result.index, documents[result.index].page_content, result.score
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // 4. CachedCrossEncoder — LRU caching
    // -----------------------------------------------------------------------
    println!("--- 4. CachedCrossEncoder: LRU caching ---");
    let cached = CachedCrossEncoder::new(FakeCrossEncoder::new(), 5);

    let pairs = vec![
        ("hello world".into(), "hello world".into()),
        ("foo bar".into(), "baz qux".into()),
    ];

    println!("  Cache size before scoring: {}", cached.cache_len());
    let scores = cached.score_pairs(&pairs).await?;
    println!("  Cache size after scoring {} pairs: {}", pairs.len(), cached.cache_len());
    for (pair, score) in pairs.iter().zip(scores.iter()) {
        println!("    ({}, {}) => {:.4}", pair.0, pair.1, score);
    }

    // Score same pairs again — served from cache
    let scores2 = cached.score_pairs(&pairs).await?;
    println!("  Cache size after re-scoring (cache hit): {}", cached.cache_len());
    assert_eq!(scores, scores2);
    println!("  Scores match on re-score: true");

    // Add more to trigger eviction (capacity = 5)
    for i in 0..4 {
        let extra = vec![(format!("key{}", i), format!("val{}", i))];
        cached.score_pairs(&extra).await?;
    }
    println!("  Cache size after adding 4 more entries (capacity=5): {}", cached.cache_len());
    println!();

    // -----------------------------------------------------------------------
    // 5. NormalizedCrossEncoder — score normalization
    // -----------------------------------------------------------------------
    println!("--- 5. NormalizedCrossEncoder: normalizing scores to [0, 1] ---");
    let normalized = NormalizedCrossEncoder::new(FakeCrossEncoder::new());

    let pairs = vec![
        ("rust programming".into(), "rust programming".into()),  // highest
        ("rust programming".into(), "rust language".into()),      // mid
        ("rust programming".into(), "totally different xyz".into()), // lowest
    ];

    let raw_scores = FakeCrossEncoder::new().score_pairs(&pairs).await?;
    let norm_scores = normalized.score_pairs(&pairs).await?;

    println!("  {:40} | {:>8} | {:>10}", "Text B", "Raw", "Normalized");
    println!("  {:-<40}-+-{:-<8}-+-{:-<10}", "", "", "");
    for ((_, b), (raw, norm)) in pairs.iter().zip(raw_scores.iter().zip(norm_scores.iter())) {
        println!("  {:40} | {:>8.4} | {:>10.4}", b, raw, norm);
    }
    println!();

    // -----------------------------------------------------------------------
    // 6. LLM + Cross-Encoder — generate a query, then rerank
    // -----------------------------------------------------------------------
    println!("--- 6. LLM-generated query with cross-encoder reranking ---");
    let model = shared::get_chat_model(vec![
        "What are the best practices for writing efficient Rust code?".to_string(),
    ]);

    let messages = vec![
        Message::system("You are a helpful assistant. Respond with a single search query."),
        Message::human("Generate a search query about Rust performance optimization."),
    ];

    let ai_response = model.invoke_messages(&messages, None).await?;
    let generated_query = ai_response.base.content.text();
    println!("  LLM-generated query: \"{}\"", generated_query);

    let knowledge_base = vec![
        Document::new("Rust zero-cost abstractions enable high performance code"),
        Document::new("Python is a popular language for data science"),
        Document::new("Optimizing Rust code with cargo bench and profiling tools"),
        Document::new("JavaScript frameworks for building web applications"),
        Document::new("Memory safety in Rust prevents common bugs and improves efficiency"),
        Document::new("Database indexing strategies for PostgreSQL"),
    ];

    let llm_reranker = CrossEncoderReranker::new(FakeCrossEncoder::new()).with_top_k(3);
    let results = llm_reranker.rerank(&generated_query, &knowledge_base).await?;

    println!("  Top 3 documents for LLM-generated query:");
    for result in &results {
        println!(
            "    [score={:.4}] {}",
            result.score, knowledge_base[result.index].page_content
        );
    }
    println!();

    println!("=== Cross-Encoder Example Complete ===");
    Ok(())
}
