//! Reranking Retriever Example
//!
//! Demonstrates reranking strategies: KeywordReranker, TfIdfReranker,
//! CascadeReranker, RerankingRetriever, and RerankerPipeline.
//!
//! Run with: cargo run -p cognis-examples --example reranking_retriever

#[path = "../shared.rs"]
mod shared;

use std::collections::HashMap;

use cognis::retrievers::reranking::{
    CascadeReranker, KeywordReranker, LengthReranker, MetadataReranker, Reranker, RerankerPipeline,
    RerankingRetriever, TfIdfReranker,
};
use cognis_core::documents::Document;
use serde_json::json;

fn doc(content: &str) -> Document {
    Document::new(content)
}

fn doc_meta(content: &str, key: &str, value: f64) -> Document {
    let mut meta = HashMap::new();
    meta.insert(key.to_string(), json!(value));
    Document::new(content).with_metadata(meta)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let documents = vec![
        doc("Rust is a systems programming language focused on safety and performance"),
        doc("Python is widely used for machine learning and data science applications"),
        doc("Rust provides memory safety without garbage collection through ownership"),
        doc("JavaScript runs in the browser and powers modern web applications"),
        doc("Rust and Python can interoperate through PyO3 bindings for performance"),
        doc("Go is a statically typed language designed for simplicity and concurrency"),
    ];

    // 1. KeywordReranker
    println!("--- KeywordReranker ---");
    let results = KeywordReranker::new().rerank("Rust memory safety performance", &documents)?;
    for (i, (d, score)) in results.iter().take(3).enumerate() {
        println!("  {}. [{:.3}] {}", i + 1, score, d.page_content);
    }

    // 2. TfIdfReranker
    println!("\n--- TfIdfReranker ---");
    let results = TfIdfReranker::new().rerank("ownership garbage collection", &documents)?;
    for (i, (d, score)) in results.iter().take(3).enumerate() {
        println!("  {}. [{:.4}] {}", i + 1, score, d.page_content);
    }

    // 3. CascadeReranker (keyword + metadata weights)
    println!("\n--- CascadeReranker ---");
    let docs_meta = vec![
        doc_meta(
            "Rust is a systems programming language focused on safety and performance",
            "relevance",
            0.9,
        ),
        doc_meta(
            "Python is widely used for machine learning and data science applications",
            "relevance",
            0.7,
        ),
        doc_meta(
            "Rust provides memory safety without garbage collection through ownership",
            "relevance",
            0.95,
        ),
        doc_meta(
            "JavaScript runs in the browser and powers modern web applications",
            "relevance",
            0.3,
        ),
    ];
    let cascade = CascadeReranker::new(vec![
        (Box::new(KeywordReranker::new()), 0.6),
        (Box::new(MetadataReranker::new("relevance")), 0.4),
    ]);
    let results = cascade.rerank("Rust safety", &docs_meta)?;
    for (i, (d, score)) in results.iter().enumerate() {
        println!("  {}. [{:.3}] {}", i + 1, score, d.page_content);
    }

    // 4. RerankingRetriever with top-k and min_score
    println!("\n--- RerankingRetriever ---");
    let retriever = RerankingRetriever::new(documents.clone(), Box::new(KeywordReranker::new()))
        .with_top_k(3)
        .with_min_score(0.2);
    let results = retriever.retrieve_with_scores("Rust programming performance", 3)?;
    for (i, (d, score)) in results.iter().enumerate() {
        println!("  {}. [{:.3}] {}", i + 1, score, d.page_content);
    }

    // 5. RerankerPipeline (multi-stage)
    println!("\n--- RerankerPipeline ---");
    let pipeline = RerankerPipeline::new(vec![
        (Box::new(KeywordReranker::new()), 4),
        (Box::new(LengthReranker::new(60)), 2),
    ]);
    let results = pipeline.run("Rust programming", &documents)?;
    for (i, (d, score)) in results.iter().enumerate() {
        println!("  {}. [{:.4}] {}", i + 1, score, d.page_content);
    }

    // 6. LLM Q&A using top reranked document as context
    println!("\n--- LLM with Reranked Context ---");
    let top = KeywordReranker::new().rerank("Rust memory safety", &documents)?;
    let context = top
        .first()
        .map(|(d, _)| d.page_content.as_str())
        .unwrap_or("");

    let model = shared::get_chat_model(vec![
        "Rust achieves memory safety through ownership and the borrow checker at compile time."
            .into(),
    ]);
    let messages = vec![cognis_core::messages::Message::human(&format!(
        "Based on: '{context}'\n\nHow does Rust achieve memory safety?"
    ))];
    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        println!("  Answer: {}", gen.message.content().text());
    }

    Ok(())
}
