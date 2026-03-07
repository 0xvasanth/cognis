//! Reranking Retriever Example
//!
//! Demonstrates the reranking retriever system that takes initial retrieval
//! results and reorders them by relevance using various scoring strategies.
//!
//! Features shown:
//! - KeywordReranker (keyword overlap scoring)
//! - TfIdfReranker (TF-IDF scoring)
//! - CascadeReranker (weighted combination of multiple rerankers)
//! - RerankingRetriever (end-to-end retrieval with reranking)
//! - RerankerPipeline (sequential multi-stage reranking)
//! - Score thresholds and top-k filtering
//!
//! No API keys required.
//!
//! Run with: `cargo run -p rustchain-examples --example reranking_retriever`

use std::collections::HashMap;

use rustchain::retrievers::reranking::{
    CascadeReranker, KeywordReranker, LengthReranker, MetadataReranker, RerankerPipeline,
    RerankingRetriever, Reranker, TfIdfReranker,
};
use rustchain_core::documents::Document;
use serde_json::json;

fn make_doc(content: &str) -> Document {
    Document::new(content)
}

fn make_doc_with_meta(content: &str, key: &str, value: f64) -> Document {
    let mut meta = HashMap::new();
    meta.insert(key.to_string(), json!(value));
    Document::new(content).with_metadata(meta)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Reranking Retriever Example ===\n");

    // Sample document corpus
    let documents = vec![
        make_doc("Rust is a systems programming language focused on safety and performance"),
        make_doc("Python is widely used for machine learning and data science applications"),
        make_doc("Rust provides memory safety without garbage collection through ownership"),
        make_doc("JavaScript runs in the browser and powers modern web applications"),
        make_doc("Rust and Python can interoperate through PyO3 bindings for performance"),
        make_doc("Go is a statically typed language designed for simplicity and concurrency"),
    ];

    // -----------------------------------------------------------------------
    // 1. KeywordReranker — scores by keyword overlap
    // -----------------------------------------------------------------------
    println!("--- 1. KeywordReranker ---");
    println!("Scores documents by the fraction of query terms found in each document.\n");

    let keyword_reranker = KeywordReranker::new();
    let query = "Rust memory safety performance";
    println!("Query: \"{}\"\n", query);

    let results = keyword_reranker.rerank(query, &documents)?;
    println!("Ranked results:");
    for (i, (doc, score)) in results.iter().enumerate() {
        println!("  {}. [score: {:.3}] {}", i + 1, score, doc.page_content);
    }

    // -----------------------------------------------------------------------
    // 2. TfIdfReranker — scores using TF-IDF
    // -----------------------------------------------------------------------
    println!("\n--- 2. TfIdfReranker ---");
    println!("Scores using term frequency * inverse document frequency.\n");
    println!("Rare terms in the corpus get higher weight.\n");

    let tfidf_reranker = TfIdfReranker::new();
    let query = "ownership garbage collection";
    println!("Query: \"{}\"\n", query);

    let results = tfidf_reranker.rerank(query, &documents)?;
    println!("Ranked results:");
    for (i, (doc, score)) in results.iter().enumerate() {
        println!("  {}. [score: {:.4}] {}", i + 1, score, doc.page_content);
    }

    // -----------------------------------------------------------------------
    // 3. CascadeReranker — weighted combination
    // -----------------------------------------------------------------------
    println!("\n--- 3. CascadeReranker ---");
    println!("Combines multiple rerankers with configurable weights.\n");

    // Documents with metadata boost scores
    let docs_with_meta = vec![
        make_doc_with_meta(
            "Rust is a systems programming language focused on safety and performance",
            "relevance",
            0.9,
        ),
        make_doc_with_meta(
            "Python is widely used for machine learning and data science applications",
            "relevance",
            0.7,
        ),
        make_doc_with_meta(
            "Rust provides memory safety without garbage collection through ownership",
            "relevance",
            0.95,
        ),
        make_doc_with_meta(
            "JavaScript runs in the browser and powers modern web applications",
            "relevance",
            0.3,
        ),
        make_doc_with_meta(
            "Rust and Python can interoperate through PyO3 bindings for performance",
            "relevance",
            0.8,
        ),
    ];

    let cascade = CascadeReranker::new(vec![
        (Box::new(KeywordReranker::new()), 0.6),
        (Box::new(MetadataReranker::new("relevance")), 0.4),
    ]);

    let query = "Rust safety";
    println!("Query: \"{}\"", query);
    println!("Weights: KeywordReranker=0.6, MetadataReranker(relevance)=0.4\n");

    let results = cascade.rerank(query, &docs_with_meta)?;
    println!("Ranked results (combined score):");
    for (i, (doc, score)) in results.iter().enumerate() {
        let meta_score = doc
            .metadata
            .get("relevance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        println!(
            "  {}. [combined: {:.3}, meta: {:.2}] {}",
            i + 1,
            score,
            meta_score,
            doc.page_content
        );
    }

    // -----------------------------------------------------------------------
    // 4. RerankingRetriever — end-to-end retrieval with top-k and min_score
    // -----------------------------------------------------------------------
    println!("\n--- 4. RerankingRetriever ---");
    println!("Wraps a document set with a reranker for end-to-end retrieval.\n");

    let retriever = RerankingRetriever::new(documents.clone(), Box::new(KeywordReranker::new()))
        .with_top_k(3)
        .with_min_score(0.2);

    let query = "Rust programming performance";
    println!("Query: \"{}\"", query);
    println!("Config: top_k=3, min_score=0.2\n");

    let results = retriever.retrieve_with_scores(query, 3)?;
    println!("Retrieved documents (top-3 with score >= 0.2):");
    for (i, (doc, score)) in results.iter().enumerate() {
        println!("  {}. [score: {:.3}] {}", i + 1, score, doc.page_content);
    }

    // Show filtering effect with a query that has low overlap
    let query_low = "database SQL NoSQL";
    println!("\nQuery: \"{}\"", query_low);
    let results_low = retriever.retrieve_with_scores(query_low, 3)?;
    println!(
        "Retrieved: {} documents (low overlap filtered by min_score)",
        results_low.len()
    );

    // -----------------------------------------------------------------------
    // 5. RerankerPipeline — sequential multi-stage reranking
    // -----------------------------------------------------------------------
    println!("\n--- 5. RerankerPipeline ---");
    println!("Applies rerankers sequentially, narrowing results at each stage.\n");

    let pipeline = RerankerPipeline::new(vec![
        // Stage 1: Keyword relevance, keep top 4
        (Box::new(KeywordReranker::new()), 4),
        // Stage 2: Prefer medium-length docs (~60 chars), keep top 2
        (Box::new(LengthReranker::new(60)), 2),
    ]);

    let query = "Rust programming";
    println!("Query: \"{}\"", query);
    println!("Pipeline: KeywordReranker(top 4) -> LengthReranker(ideal=60, top 2)\n");

    let results = pipeline.run(query, &documents)?;
    println!("Final results after pipeline:");
    for (i, (doc, score)) in results.iter().enumerate() {
        println!(
            "  {}. [score: {:.4}, len: {}] {}",
            i + 1,
            score,
            doc.page_content.len(),
            doc.page_content
        );
    }

    // -----------------------------------------------------------------------
    // 6. Comparing Rerankers on the Same Query
    // -----------------------------------------------------------------------
    println!("\n--- 6. Comparison ---");
    println!("Same query ranked by different rerankers.\n");

    let query = "machine learning Python";
    println!("Query: \"{}\"\n", query);

    let keyword_results = KeywordReranker::new().rerank(query, &documents)?;
    let tfidf_results = TfIdfReranker::new().rerank(query, &documents)?;

    println!("KeywordReranker top-3:");
    for (i, (doc, score)) in keyword_results.iter().take(3).enumerate() {
        println!("  {}. [score: {:.3}] {}", i + 1, score, doc.page_content);
    }

    println!("\nTfIdfReranker top-3:");
    for (i, (doc, score)) in tfidf_results.iter().take(3).enumerate() {
        println!("  {}. [score: {:.4}] {}", i + 1, score, doc.page_content);
    }

    println!("\n=== Reranking Retriever Example Complete ===");
    Ok(())
}
