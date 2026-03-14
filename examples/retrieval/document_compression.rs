//! Document Compression Example
//!
//! Demonstrates LengthCompressor, SentenceExtractor, RedundancyFilter,
//! RelevanceScorer, CompressorPipeline, and ContextualCompressionRetriever.

#[path = "../shared.rs"]
mod shared;
use std::collections::HashMap;

use cognis::retrievers::compression::{
    CompressorPipeline, ContextualCompressionRetriever, DocumentCompressor, LengthCompressor,
    RedundancyFilter, RelevanceScorer, SentenceExtractor,
};
use cognis_core::documents::Document;
use serde_json::json;

fn main() {
    let documents = vec![
        Document::new(
            "Rust is a systems programming language. It provides memory safety without garbage collection. \
             The borrow checker enforces ownership rules at compile time. Rust is widely used for CLI tools and web servers."
        ),
        Document::new(
            "Python is a high-level programming language. It is popular for data science and machine learning. \
             Python uses dynamic typing and has a large ecosystem of libraries."
        ),
        Document::new(
            "Rust is a systems programming language. It provides memory safety without garbage collection. \
             The borrow checker enforces ownership rules at compile time. Rust is widely used for CLI tools and web servers."
        ), // duplicate of first
        Document::new(
            "Go is a statically typed language designed at Google. It compiles quickly and has built-in concurrency \
             via goroutines. Go is often used for cloud infrastructure and microservices."
        ),
    ];

    // 1. LengthCompressor
    let truncated = LengthCompressor::new(80)
        .compress(&documents, "any")
        .unwrap();
    println!("LengthCompressor (max 80 chars): {} docs", truncated.len());
    for doc in &truncated {
        println!(
            "  ({} chars) \"{}\"",
            doc.page_content.len(),
            doc.page_content
        );
    }

    // 2. SentenceExtractor
    let extracted = SentenceExtractor::new()
        .with_min_sentences(1)
        .compress(&documents, "borrow checker memory")
        .unwrap();
    println!(
        "SentenceExtractor ('borrow checker memory'): {} docs",
        extracted.len()
    );
    for doc in &extracted {
        println!("  \"{}\"", doc.page_content);
    }

    // 3. RedundancyFilter
    let deduped = RedundancyFilter::new(0.8)
        .compress(&documents, "any")
        .unwrap();
    println!(
        "RedundancyFilter: {} -> {} docs (removed duplicates)",
        documents.len(),
        deduped.len()
    );

    // 4. RelevanceScorer
    let relevant = RelevanceScorer::new(0.5)
        .compress(&documents, "rust memory safety")
        .unwrap();
    println!(
        "RelevanceScorer: {} of {} passed threshold",
        relevant.len(),
        documents.len()
    );

    // 5. CompressorPipeline
    let pipeline = CompressorPipeline::new()
        .add(Box::new(RedundancyFilter::new(0.8)))
        .add(Box::new(SentenceExtractor::new().with_min_sentences(1)))
        .add(Box::new(LengthCompressor::new(100)));

    let result = pipeline.compress(&documents, "borrow checker").unwrap();
    println!(
        "Pipeline ({} stages): {} docs",
        pipeline.len(),
        result.len()
    );
    for doc in &result {
        println!(
            "  ({} chars) \"{}\"",
            doc.page_content.len(),
            doc.page_content
        );
    }

    // 6. ContextualCompressionRetriever
    let mut metadata = HashMap::new();
    metadata.insert("source".to_string(), json!("rust-book"));
    let docs_with_meta = vec![
        Document::new("The ownership system is Rust's most unique feature. It enables memory safety without a GC.")
            .with_metadata(metadata.clone()),
        Document::new("Cargo is the Rust package manager. It handles deps and publishing.")
            .with_metadata(metadata),
        Document::new("Cooking pasta requires boiling water. Add salt. Cook 8-10 minutes."),
    ];

    let retriever_pipeline = CompressorPipeline::new()
        .add(Box::new(RelevanceScorer::new(0.3)))
        .add(Box::new(LengthCompressor::new(120)));
    let retriever =
        ContextualCompressionRetriever::new(docs_with_meta, Box::new(retriever_pipeline));

    let results = retriever.retrieve("rust ownership memory", 2).unwrap();
    println!(
        "ContextualCompressionRetriever: {} docs for 'rust ownership memory'",
        results.len()
    );
    for doc in &results {
        println!("  \"{}\"", doc.page_content);
    }

    // 7. LLM demo — summarize compressed documents
    let model = shared::get_chat_model(vec![
        "The documents discuss Rust's ownership system and memory safety via the borrow checker."
            .into(),
    ]);
    let text: String = result
        .iter()
        .map(|d| d.page_content.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let messages = vec![
        cognis_core::messages::Message::system("Summarize the given documents briefly."),
        cognis_core::messages::Message::human(&format!("Summarize: {}", text)),
    ];
    let rt = tokio::runtime::Runtime::new().unwrap();
    match rt.block_on(model.invoke_messages(&messages, None)) {
        Ok(r) => println!("LLM summary: {}", r.base.content.text()),
        Err(e) => println!("LLM error: {}", e),
    }
}
