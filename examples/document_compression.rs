//! Document Compression Example
//!
//! Demonstrates the document compression retriever system:
//!
//! - `LengthCompressor` — truncate documents to a max character count
//! - `SentenceExtractor` — extract sentences matching query keywords
//! - `RedundancyFilter` — remove near-duplicate documents via Jaccard similarity
//! - `CompressorPipeline` — chain compressors in sequence
//! - `ContextualCompressionRetriever` — end-to-end compressed retrieval
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example document_compression`

use std::collections::HashMap;

use cognis::retrievers::compression::{
    CompressorPipeline, ContextualCompressionRetriever, DocumentCompressor, LengthCompressor,
    RedundancyFilter, RelevanceScorer, SentenceExtractor,
};
use cognis_core::documents::Document;
use serde_json::json;

fn main() {
    println!("=== Document Compression Example ===\n");

    // -----------------------------------------------------------------------
    // 1. Create sample documents
    // -----------------------------------------------------------------------
    println!("--- 1. Sample documents ---");

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
        ), // exact duplicate of first
        Document::new(
            "Go is a statically typed language designed at Google. It compiles quickly and has built-in concurrency \
             via goroutines. Go is often used for cloud infrastructure and microservices."
        ),
    ];

    for (i, doc) in documents.iter().enumerate() {
        println!(
            "  Doc {}: \"{}...\" ({} chars)",
            i,
            &doc.page_content[..doc.page_content.len().min(60)],
            doc.page_content.len()
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // 2. LengthCompressor
    // -----------------------------------------------------------------------
    println!("--- 2. LengthCompressor (max 80 chars) ---");

    let length_compressor = LengthCompressor::new(80);
    let truncated = length_compressor
        .compress(&documents, "any query")
        .expect("compression should succeed");

    for (i, doc) in truncated.iter().enumerate() {
        println!(
            "  Doc {}: \"{}\" ({} chars)",
            i,
            doc.page_content,
            doc.page_content.len()
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // 3. SentenceExtractor
    // -----------------------------------------------------------------------
    println!("--- 3. SentenceExtractor (query: 'borrow checker memory') ---");

    let extractor = SentenceExtractor::new().with_min_sentences(1);
    let extracted = extractor
        .compress(&documents, "borrow checker memory")
        .expect("extraction should succeed");

    for (i, doc) in extracted.iter().enumerate() {
        println!("  Doc {}: \"{}\"", i, doc.page_content);
    }
    println!();

    // -----------------------------------------------------------------------
    // 4. RedundancyFilter
    // -----------------------------------------------------------------------
    println!("--- 4. RedundancyFilter (threshold 0.8) ---");
    println!(
        "  Input: {} documents (doc 0 and doc 2 are duplicates)",
        documents.len()
    );

    let redundancy_filter = RedundancyFilter::new(0.8);
    let deduped = redundancy_filter
        .compress(&documents, "any")
        .expect("filter should succeed");

    println!("  Output: {} documents after deduplication", deduped.len());
    for (i, doc) in deduped.iter().enumerate() {
        println!(
            "  Doc {}: \"{}...\"",
            i,
            &doc.page_content[..doc.page_content.len().min(50)]
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // 5. RelevanceScorer
    // -----------------------------------------------------------------------
    println!("--- 5. RelevanceScorer (min_score 0.5, query: 'rust memory safety') ---");

    let scorer = RelevanceScorer::new(0.5);
    let relevant = scorer
        .compress(&documents, "rust memory safety")
        .expect("scoring should succeed");

    println!(
        "  {} of {} documents passed relevance threshold",
        relevant.len(),
        documents.len()
    );
    for doc in &relevant {
        println!(
            "  \"{}...\"",
            &doc.page_content[..doc.page_content.len().min(60)]
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // 6. CompressorPipeline
    // -----------------------------------------------------------------------
    println!("--- 6. CompressorPipeline (deduplicate -> extract sentences -> truncate) ---");

    let pipeline = CompressorPipeline::new()
        .add(Box::new(RedundancyFilter::new(0.8)))
        .add(Box::new(SentenceExtractor::new().with_min_sentences(1)))
        .add(Box::new(LengthCompressor::new(100)));

    println!("  Pipeline stages: {}", pipeline.len());

    let pipeline_result = pipeline
        .compress(&documents, "borrow checker")
        .expect("pipeline should succeed");

    println!("  Results ({} documents):", pipeline_result.len());
    for (i, doc) in pipeline_result.iter().enumerate() {
        println!(
            "  Doc {}: \"{}\" ({} chars)",
            i,
            doc.page_content,
            doc.page_content.len()
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // 7. ContextualCompressionRetriever (end-to-end)
    // -----------------------------------------------------------------------
    println!("--- 7. ContextualCompressionRetriever ---");

    let mut metadata = HashMap::new();
    metadata.insert("source".to_string(), json!("rust-book"));
    let docs_with_meta = vec![
        Document::new(
            "The ownership system is Rust's most unique feature. It enables memory safety guarantees \
             without a garbage collector. Each value has exactly one owner.",
        )
        .with_metadata(metadata.clone()),
        Document::new(
            "Cargo is the Rust package manager. It handles downloading dependencies, compiling packages, \
             and publishing to crates.io.",
        )
        .with_metadata(metadata),
        Document::new(
            "Cooking pasta requires boiling water. Add salt for flavor. Cook for 8-10 minutes.",
        ),
    ];

    let retriever_pipeline = CompressorPipeline::new()
        .add(Box::new(RelevanceScorer::new(0.3)))
        .add(Box::new(LengthCompressor::new(120)));

    let retriever =
        ContextualCompressionRetriever::new(docs_with_meta, Box::new(retriever_pipeline));

    let query = "rust ownership memory";
    let results = retriever
        .retrieve(query, 2)
        .expect("retrieval should succeed");

    println!("  Query: \"{}\"", query);
    println!("  Retrieved {} documents (max 2):", results.len());
    for (i, doc) in results.iter().enumerate() {
        println!("  [{}] \"{}\"", i, doc.page_content);
        if !doc.metadata.is_empty() {
            println!("      metadata: {:?}", doc.metadata);
        }
    }
    println!();

    // -----------------------------------------------------------------------
    // 8. Retrieve all with a different query
    // -----------------------------------------------------------------------
    println!("--- 8. Retrieve all for query 'cargo package' ---");

    let docs2 = vec![
        Document::new("Cargo manages Rust project dependencies and builds."),
        Document::new("The package ecosystem is hosted on crates.io."),
        Document::new("Unrelated content about weather patterns."),
    ];
    let simple_pipeline = CompressorPipeline::new().add(Box::new(RelevanceScorer::new(0.5)));
    let retriever2 = ContextualCompressionRetriever::new(docs2, Box::new(simple_pipeline));

    let all_results = retriever2
        .retrieve_all("cargo package")
        .expect("retrieval should succeed");

    println!("  {} documents matched:", all_results.len());
    for doc in &all_results {
        println!("  - \"{}\"", doc.page_content);
    }
    println!();

    println!("=== Document Compression Example Complete ===");
}
