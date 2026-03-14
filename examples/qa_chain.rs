//! QA Chain Example
//!
//! Demonstrates the QA (Question Answering) chain with mock documents.
//! Shows how to format documents into a QA prompt, use different chain types,
//! extract citations, and use the Runnable interface.
//!
//! No API keys required -- uses document formatting only (no LLM calls).

mod shared;
use std::collections::HashMap;

use serde_json::json;

use cognis::chains::{create_qa_chain, CitedAnswer, QAChain, QAChainType, QAConfig, QAResult};
use cognis_core::documents::Document;
use cognis_core::messages::Message;
use cognis_core::runnables::Runnable;

/// Create a sample document with source metadata.
fn make_doc(content: &str, source: &str) -> Document {
    let mut metadata = HashMap::new();
    metadata.insert("source".to_string(), json!(source));
    Document::new(content).with_metadata(metadata)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== QA Chain Example ===\n");

    // Create sample documents (simulating retrieved documents from a knowledge base)
    let documents = vec![
        make_doc(
            "Rust is a multi-paradigm, general-purpose programming language that emphasizes \
             performance, type safety, and concurrency. It enforces memory safety without a \
             garbage collector.",
            "rust_overview.md",
        ),
        make_doc(
            "The Rust compiler uses a borrow checker to enforce ownership rules at compile time. \
             This prevents data races and ensures memory safety without runtime overhead.",
            "rust_borrow_checker.md",
        ),
        make_doc(
            "Cargo is the Rust package manager and build system. It handles downloading \
             dependencies, compiling packages, and publishing to crates.io.",
            "cargo_guide.md",
        ),
        make_doc(
            "Ferris is the unofficial mascot of the Rust programming language. Ferris is a \
             red crab, chosen because crabs are related to the concept of 'Rustaceans'.",
            "rust_mascot.md",
        ),
    ];

    // =========================================================================
    // 1. Basic QA with default config
    // =========================================================================
    println!("--- 1. Basic QA (Stuff strategy, default config) ---\n");

    let chain = create_qa_chain(QAConfig::default());

    let result = chain.answer("What is Rust?", &documents)?;
    println!("  Question: What is Rust?");
    println!("  Chain type: {}", result.chain_type);
    println!("  Source documents used: {}", result.source_documents.len());
    println!("  Answer (formatted prompt):");
    for line in result.answer.lines().take(8) {
        println!("    {}", line);
    }
    println!("    ...");

    // =========================================================================
    // 2. Limiting source documents
    // =========================================================================
    println!("\n--- 2. Limiting source documents (max_source_docs=2) ---\n");

    let config = QAConfig::builder().max_source_docs(2).build();
    let chain = create_qa_chain(config);

    let result = chain.answer("Tell me about Cargo", &documents)?;
    println!("  Question: Tell me about Cargo");
    println!(
        "  Source documents used: {} (limited to 2)",
        result.source_documents.len()
    );
    for (i, doc) in result.source_documents.iter().enumerate() {
        let source = doc
            .metadata
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        println!(
            "    Doc {}: source={}, content={}...",
            i + 1,
            source,
            &doc.page_content[..doc.page_content.len().min(50)]
        );
    }

    // =========================================================================
    // 3. Without source documents in result
    // =========================================================================
    println!("\n--- 3. Hiding source documents in result ---\n");

    let config = QAConfig::builder().return_source_documents(false).build();
    let chain = create_qa_chain(config);

    let result = chain.answer("What is the borrow checker?", &documents)?;
    println!("  return_source_documents=false");
    println!(
        "  Source docs in result: {} (empty as configured)",
        result.source_documents.len()
    );
    println!(
        "  Answer still contains document content: {}",
        result.answer.contains("borrow checker")
    );

    // =========================================================================
    // 4. Custom prompt templates
    // =========================================================================
    println!("\n--- 4. Custom prompt templates ---\n");

    let chain = QAChain::new(QAConfig::default())
        .with_document_prompt("Source [{doc_index}] ({metadata.source}): {page_content}")
        .with_qa_prompt(
            "Based on these sources:\n\n{context}\n\n---\nQuestion: {question}\nProvide a concise answer with citations:",
        );

    let result = chain.answer("Who is Ferris?", &documents)?;
    println!("  Custom-formatted prompt (first 300 chars):");
    let preview = &result.answer[..result.answer.len().min(300)];
    for line in preview.lines() {
        println!("    {}", line);
    }
    println!("    ...");

    // =========================================================================
    // 5. Citation extraction
    // =========================================================================
    println!("\n--- 5. Citation extraction ---\n");

    // Simulate an LLM answer that references documents by index
    let simulated_answer = "Rust emphasizes performance and memory safety [1]. \
        The borrow checker enforces ownership rules at compile time [2]. \
        Cargo handles dependency management [3].";

    let cited = CitedAnswer::from_answer_and_docs(simulated_answer, &documents);
    println!("  Simulated answer: {}", cited.answer);
    println!("  Citations found: {}", cited.citations.len());
    for citation in &cited.citations {
        println!(
            "    [{}] source={}, snippet={}...",
            citation.doc_index + 1,
            citation.source,
            &citation.page_content_snippet[..citation.page_content_snippet.len().min(60)]
        );
    }

    // Also via the chain method
    let chain = create_qa_chain(QAConfig::default());
    let cited2 = chain.extract_citations(simulated_answer, &documents);
    println!(
        "\n  Via chain.extract_citations(): {} citations",
        cited2.citations.len()
    );

    // =========================================================================
    // 6. Different chain types
    // =========================================================================
    println!("\n--- 6. Different chain type configurations ---\n");

    for chain_type in [
        QAChainType::Stuff,
        QAChainType::MapReduce,
        QAChainType::Refine,
    ] {
        let config = QAConfig::builder().chain_type(chain_type).build();
        let chain = create_qa_chain(config);
        let result = chain.answer("What is Rust?", &documents)?;
        println!(
            "  chain_type={}: result.chain_type={}",
            chain_type, result.chain_type
        );
    }

    // =========================================================================
    // 7. Runnable interface (invoke with JSON)
    // =========================================================================
    println!("\n--- 7. Runnable interface ---\n");

    let chain = create_qa_chain(QAConfig::default());
    println!("  Chain name: {}", chain.name());

    let input = json!({
        "question": "What does Cargo do?",
        "documents": [
            { "page_content": "Cargo is the Rust package manager." },
            { "page_content": "Cargo compiles Rust projects and manages dependencies." }
        ]
    });

    let result = chain.invoke(input, None).await?;
    let qa_result: QAResult = serde_json::from_value(result)?;
    println!("  Question: What does Cargo do?");
    println!(
        "  Source docs in result: {}",
        qa_result.source_documents.len()
    );
    println!("  Chain type: {}", qa_result.chain_type);
    println!(
        "  Answer contains 'Cargo': {}",
        qa_result.answer.contains("Cargo")
    );

    // =========================================================================
    // 8. answer_with_context: pre-formatted context
    // =========================================================================
    println!("\n--- 8. Pre-formatted context ---\n");

    let chain = create_qa_chain(QAConfig::default());
    let context = "Rust was created by Graydon Hoare at Mozilla Research. \
                   The first stable release (1.0) was in May 2015.";
    let prompt = chain.answer_with_context("When was Rust 1.0 released?", context)?;
    println!("  Generated prompt (first 200 chars):");
    for line in prompt[..prompt.len().min(200)].lines() {
        println!("    {}", line);
    }

    // =========================================================================
    // 9. Error handling
    // =========================================================================
    println!("\n--- 9. Error handling ---\n");

    let chain = create_qa_chain(QAConfig::default());
    let result = chain.answer("", &documents);
    match result {
        Err(e) => println!("  Empty question error: {}", e),
        Ok(_) => println!("  Empty question: (unexpected success)"),
    }

    // --- Real LLM Demo ---
    println!("\n--- Real LLM Demo ---\n");
    println!("Using an LLM to answer a question based on retrieved documents.\n");

    let chain = create_qa_chain(QAConfig::default());
    let qa_prompt = chain.answer("What is Rust's mascot?", &documents)?;

    let model = shared::get_chat_model(vec![
        "Based on the provided context, Ferris is the unofficial mascot of Rust. Ferris is a red crab, chosen because Rust developers are called 'Rustaceans'.".into(),
    ]);
    let messages = vec![
        Message::system("Answer the question using only the provided context."),
        Message::human(&qa_prompt.answer),
    ];
    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        println!("Question: What is Rust's mascot?");
        println!("LLM Answer: {}", gen.message.content().text());
    }

    println!("\n=== QA Chain Example Complete ===");
    Ok(())
}
