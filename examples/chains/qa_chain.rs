//! QA Chain Example
//!
//! Demonstrates the QA chain with mock documents: formatting documents into
//! prompts, chain type selection, citation extraction, and the Runnable interface.

#[path = "../shared.rs"]
mod shared;
use std::collections::HashMap;

use serde_json::json;

use cognis::chains::{create_qa_chain, CitedAnswer, QAChain, QAChainType, QAConfig, QAResult};
use cognis_core::documents::Document;
use cognis_core::messages::Message;
use cognis_core::runnables::Runnable;

fn make_doc(content: &str, source: &str) -> Document {
    let mut metadata = HashMap::new();
    metadata.insert("source".to_string(), json!(source));
    Document::new(content).with_metadata(metadata)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let documents = vec![
        make_doc("Rust emphasizes performance, type safety, and concurrency. It enforces memory safety without a garbage collector.", "rust_overview.md"),
        make_doc("The Rust compiler uses a borrow checker to enforce ownership rules at compile time.", "rust_borrow_checker.md"),
        make_doc("Cargo is the Rust package manager and build system.", "cargo_guide.md"),
        make_doc("Ferris is the unofficial mascot of Rust, a red crab.", "rust_mascot.md"),
    ];

    // 1. Basic QA with default config
    let chain = create_qa_chain(QAConfig::default());
    let result = chain.answer("What is Rust?", &documents)?;
    println!(
        "Basic QA: {} source docs, chain_type={}",
        result.source_documents.len(),
        result.chain_type
    );

    // 2. Limiting source documents
    let config = QAConfig::builder().max_source_docs(2).build();
    let chain = create_qa_chain(config);
    let result = chain.answer("Tell me about Cargo", &documents)?;
    println!(
        "Limited: {} source docs (max 2)",
        result.source_documents.len()
    );

    // 3. Custom prompt templates
    let chain = QAChain::new(QAConfig::default())
        .with_document_prompt("Source [{doc_index}] ({metadata.source}): {page_content}")
        .with_qa_prompt("Sources:\n{context}\n---\nQ: {question}\nAnswer with citations:");
    let result = chain.answer("Who is Ferris?", &documents)?;
    println!(
        "Custom prompt: answer contains 'Ferris' = {}",
        result.answer.contains("Ferris")
    );

    // 4. Citation extraction
    let simulated = "Rust emphasizes performance [1]. The borrow checker enforces ownership [2]. Cargo handles deps [3].";
    let cited = CitedAnswer::from_answer_and_docs(simulated, &documents);
    println!("Citations found: {}", cited.citations.len());
    for c in &cited.citations {
        println!("  [{}] source={}", c.doc_index + 1, c.source);
    }

    // 5. Different chain types
    for chain_type in [
        QAChainType::Stuff,
        QAChainType::MapReduce,
        QAChainType::Refine,
    ] {
        let config = QAConfig::builder().chain_type(chain_type).build();
        let chain = create_qa_chain(config);
        let result = chain.answer("What is Rust?", &documents)?;
        println!("chain_type={}: {}", chain_type, result.chain_type);
    }

    // 6. Runnable interface
    let chain = create_qa_chain(QAConfig::default());
    let input = json!({
        "question": "What does Cargo do?",
        "documents": [
            { "page_content": "Cargo is the Rust package manager." },
            { "page_content": "Cargo compiles Rust projects and manages dependencies." }
        ]
    });
    let result: QAResult = serde_json::from_value(chain.invoke(input, None).await?)?;
    println!(
        "Runnable: {} source docs, contains 'Cargo' = {}",
        result.source_documents.len(),
        result.answer.contains("Cargo")
    );

    // 7. LLM demo with retrieved documents
    let chain = create_qa_chain(QAConfig::default());
    let qa_prompt = chain.answer("What is Rust's mascot?", &documents)?;
    let model = shared::get_chat_model(vec![
        "Ferris is the unofficial mascot of Rust. Ferris is a red crab.".into(),
    ]);
    let messages = vec![
        Message::system("Answer using only the provided context."),
        Message::human(&qa_prompt.answer),
    ];
    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        println!("LLM answer: {}", gen.message.content().text());
    }

    Ok(())
}
