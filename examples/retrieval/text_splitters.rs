//! Text Splitters Example
//!
//! Demonstrates chunking documents with RecursiveCharacterTextSplitter and
//! MarkdownTextSplitter, including document splitting with metadata preservation
//! and a quick LLM summarization demo.
//!
//! Run with: `cargo run -p cognis-examples --example text_splitters`

#[path = "../shared.rs"]
mod shared;

use std::collections::HashMap;

use cognis::text_splitters::{MarkdownTextSplitter, RecursiveCharacterTextSplitter, TextSplitter};
use cognis_core::documents::Document;
use cognis_core::messages::Message;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Text Splitters Example ===\n");

    // -- 1. RecursiveCharacterTextSplitter ------------------------------------
    println!("--- RecursiveCharacterTextSplitter ---");

    let article = "\
The Rust programming language is fast and safe.\n\n\
Ownership and borrowing are key features. They prevent data races \
at compile time and eliminate the need for a garbage collector.\n\n\
Cargo is the Rust package manager. It handles dependencies, building, \
testing, and publishing crates to crates.io.\n\n\
The community is welcoming and helpful. The Rust subreddit, Discord, \
and forums are great places to ask questions.";

    let splitter = RecursiveCharacterTextSplitter::new()
        .with_chunk_size(80)
        .with_chunk_overlap(10);
    let chunks = splitter.split_text(article);

    println!(
        "Input: {} chars -> {} chunks (size=80, overlap=10)",
        article.len(),
        chunks.len()
    );
    for (i, chunk) in chunks.iter().enumerate() {
        println!(
            "  [{}] ({} chars) {:?}",
            i,
            chunk.len(),
            truncate(chunk, 80)
        );
    }

    // -- 2. MarkdownTextSplitter ----------------------------------------------
    println!("\n--- MarkdownTextSplitter ---");

    let markdown = "\
# Getting Started\n\n\
Welcome to the project. This guide walks you through setup.\n\n\
## Installation\n\n\
Install with cargo:\n\n\
```\ncargo add cognis\n```\n\n\
## Configuration\n\n\
Create a config file in your project root.\n\n\
### Environment Variables\n\n\
Set API keys via environment variables for security.\n\n\
## Usage\n\n\
Import the crate and create your first chain.";

    let md_splitter = MarkdownTextSplitter::new()
        .with_chunk_size(80)
        .with_chunk_overlap(0);
    let chunks = md_splitter.split_text(markdown);

    println!("{} chunks (size=80):", chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        println!(
            "  [{}] ({} chars) {:?}",
            i,
            chunk.len(),
            truncate(chunk, 80)
        );
    }

    // -- 3. Document splitting with metadata ----------------------------------
    println!("\n--- Document Splitting with Metadata ---");

    let docs = vec![
        Document::new(
            "Rust provides memory safety without garbage collection. \
             The borrow checker enforces ownership rules at compile time.",
        )
        .with_metadata(HashMap::from([("source".into(), json!("chapter1.txt"))])),
        Document::new(
            "Tokio is an async runtime for Rust. \
             It powers many production web services and networking tools.",
        )
        .with_metadata(HashMap::from([("source".into(), json!("chapter2.txt"))])),
    ];

    let doc_splitter = RecursiveCharacterTextSplitter::new()
        .with_chunk_size(50)
        .with_chunk_overlap(0);
    let split_docs = doc_splitter.split_documents(docs);

    println!("2 documents -> {} chunks (size=50):", split_docs.len());
    for (i, doc) in split_docs.iter().enumerate() {
        let source = doc
            .metadata
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        println!(
            "  [{}] source={:?} -> {:?}",
            i,
            source,
            truncate(&doc.page_content, 60)
        );
    }

    // -- 4. LLM summarization of a chunk --------------------------------------
    println!("\n--- LLM Summarization Demo ---");

    let model = shared::get_chat_model(vec![
        "This chunk introduces Rust as a fast and safe systems programming language.".into(),
    ]);

    let demo_chunks = RecursiveCharacterTextSplitter::new()
        .with_chunk_size(80)
        .with_chunk_overlap(10)
        .split_text(article);

    if let Some(first) = demo_chunks.first() {
        let messages = vec![
            Message::system("Summarize the following text chunk in one sentence."),
            Message::human(first),
        ];
        let result = model._generate(&messages, None).await?;
        if let Some(gen) = result.generations.first() {
            println!("Chunk:   {:?}", truncate(first, 70));
            println!("Summary: {}", gen.message.content().text());
        }
    }

    println!("\n=== Done ===");
    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}...", &s[..max - 3])
    } else {
        s.to_string()
    }
}
