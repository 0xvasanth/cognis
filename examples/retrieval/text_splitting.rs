//! Text Splitting Example
//!
//! Demonstrates RecursiveCharacterTextSplitter (default, Markdown, Python presets)
//! and SentenceTextSplitter with abbreviation handling.
//!
//! Run with: cargo run -p cognis-examples --example text_splitting

#[path = "../shared.rs"]
mod shared;

use cognis::text_splitter::{
    Language, RecursiveCharacterTextSplitter, SentenceTextSplitter, TextSplitter,
};
use cognis_core::messages::Message;

const SAMPLE_TEXT: &str = "Rust is a multi-paradigm, general-purpose programming language that \
    emphasizes performance, type safety, and concurrency. It enforces memory safety without a \
    garbage collector.\n\nThe language grew out of a personal project begun in 2006. Mozilla \
    began sponsoring the project in 2009. The first stable release was Rust 1.0 on May 15, 2015.\n\n\
    Key features include zero-cost abstractions, move semantics, guaranteed memory safety, \
    threads without data races, trait-based generics, and pattern matching.";

const MARKDOWN_TEXT: &str = "# Rust Guide\n\n## Getting Started\n\nInstall via rustup.\n\n\
    ### First Project\n\n```bash\ncargo new hello_world\ncargo run\n```\n\n\
    ## Ownership\n\nEvery value has exactly one owner. When the owner goes out of scope, \
    the value is dropped.\n\n## Error Handling\n\nRust uses Result<T, E> for recoverable errors \
    and panic! for unrecoverable ones.";

const PYTHON_CODE: &str = r#"
import os
from typing import List

class DocumentLoader:
    """Base class for document loaders."""
    def __init__(self, source: str):
        self.source = source

    def load(self) -> List[str]:
        raise NotImplementedError

class FileLoader(DocumentLoader):
    def load(self) -> List[str]:
        with open(self.source, "r") as f:
            return [f.read()]
"#;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Recursive character splitting.
    println!("--- RecursiveCharacterTextSplitter ---");
    let splitter = RecursiveCharacterTextSplitter::new()
        .with_chunk_size(200)
        .with_chunk_overlap(30);
    let chunks = splitter.split_text(SAMPLE_TEXT);
    println!(
        "Input: {} chars -> {} chunks (size=200, overlap=30)",
        SAMPLE_TEXT.len(),
        chunks.len()
    );
    for (i, chunk) in chunks.iter().enumerate() {
        println!("  Chunk {}: {} chars", i + 1, chunk.len());
    }

    // 2. Markdown preset.
    println!("\n--- Markdown Preset ---");
    let md_chunks = RecursiveCharacterTextSplitter::for_language(Language::Markdown)
        .with_chunk_size(300)
        .with_chunk_overlap(20)
        .split_text(MARKDOWN_TEXT);
    println!(
        "{} chars -> {} chunks",
        MARKDOWN_TEXT.len(),
        md_chunks.len()
    );
    for (i, chunk) in md_chunks.iter().enumerate() {
        let first_line = chunk.lines().next().unwrap_or("(empty)");
        println!(
            "  Chunk {}: {} chars, starts: \"{}\"",
            i + 1,
            chunk.len(),
            first_line
        );
    }

    // 3. Python preset.
    println!("\n--- Python Preset ---");
    let py_chunks = RecursiveCharacterTextSplitter::for_language(Language::Python)
        .with_chunk_size(300)
        .with_chunk_overlap(0)
        .split_text(PYTHON_CODE);
    println!("{} chars -> {} chunks", PYTHON_CODE.len(), py_chunks.len());
    for (i, chunk) in py_chunks.iter().enumerate() {
        let first_line = chunk.lines().next().unwrap_or("(empty)").trim();
        println!(
            "  Chunk {}: {} chars, starts: \"{}\"",
            i + 1,
            chunk.len(),
            first_line
        );
    }

    // 4. Sentence splitter with abbreviation handling.
    println!("\n--- SentenceTextSplitter ---");
    let abbrev_text = "Dr. Smith arrived at 3 p.m. to meet Prof. Johnson. \
        They discussed the U.S. patent for the new A.I. system. \
        Mr. Brown from the U.K. joined via video call.";
    let sentence_splitter = SentenceTextSplitter::builder()
        .chunk_size(120)
        .chunk_overlap(0)
        .strip_whitespace(true)
        .build();
    let sentences = sentence_splitter.split_into_sentences(abbrev_text);
    println!("Detected {} sentences", sentences.len());
    let sent_chunks = sentence_splitter.split_text(abbrev_text);
    println!("Chunked into {} chunks (max 120 chars)", sent_chunks.len());

    // 5. LLM summarization of a chunk.
    println!("\n--- LLM Summarization ---");
    let model = shared::get_chat_model(vec![
        "Rust is a performance-focused, memory-safe language created at Mozilla.".into(),
    ]);
    if let Some(first_chunk) = chunks.first() {
        let messages = vec![
            Message::system("Summarize in one sentence."),
            Message::human(first_chunk),
        ];
        let result = model._generate(&messages, None).await?;
        if let Some(gen) = result.generations.first() {
            println!("  Summary: {}", gen.message.content().text());
        }
    }

    Ok(())
}
