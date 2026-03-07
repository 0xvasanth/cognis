//! Text Splitting Example
//!
//! Demonstrates various text splitter strategies:
//! - RecursiveCharacterTextSplitter with default separators
//! - Splitting a large text into chunks with configurable size and overlap
//! - Language-specific splitting (Markdown and Python presets)
//! - SentenceTextSplitter with abbreviation handling
//!
//! No API keys required.
//!
//! Run with: cargo run -p rustchain-examples --example text_splitting

use rustchain::text_splitter::{
    Language, RecursiveCharacterTextSplitter, SentenceTextSplitter, TextSplitter,
};
use rustchain_core::documents::Document;

/// Sample long-form text for demonstrating recursive splitting.
const SAMPLE_TEXT: &str = r#"Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, type safety, and concurrency. It enforces memory safety without a garbage collector. Rust was originally designed by Graydon Hoare at Mozilla Research, with contributions from Dave Herman and others. The designers refined the language while working on the experimental Servo browser engine and the Rust compiler itself.

The language grew out of a personal project begun in 2006 by Hoare. Mozilla began sponsoring the project in 2009 and announced it in 2010. The first stable release, Rust 1.0, was made on May 15, 2015. Since then, new stable releases have been delivered every six weeks.

Rust has been adopted by major technology companies. Microsoft uses Rust in parts of Windows, Amazon Web Services uses it in Firecracker, and Google uses it in parts of Android and Chromium. The language consistently tops developer surveys as one of the most loved programming languages.

Key features of Rust include zero-cost abstractions, move semantics, guaranteed memory safety, threads without data races, trait-based generics, pattern matching, type inference, minimal runtime, and efficient C bindings. These features make Rust suitable for systems programming, embedded development, and web assembly applications."#;

/// Sample Markdown content for language-specific splitting.
const MARKDOWN_TEXT: &str = r#"# Rust Programming Guide

## Getting Started

Rust is installed via rustup, the Rust toolchain manager. It handles
downloading and managing multiple Rust versions and associated tools.

### Installation

Run the following command to install rustup:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### First Project

Create a new project with Cargo:

```bash
cargo new hello_world
cd hello_world
cargo run
```

## Ownership and Borrowing

Ownership is Rust's most distinctive feature. Every value has exactly
one owner, and when the owner goes out of scope, the value is dropped.

### Borrowing Rules

1. You can have either one mutable reference or many immutable references.
2. References must always be valid (no dangling pointers).

## Error Handling

Rust uses the Result type for recoverable errors and the panic! macro
for unrecoverable errors. The ? operator enables ergonomic error propagation."#;

/// Sample Python code for language-specific splitting.
const PYTHON_CODE: &str = r#"
import os
from typing import List, Optional

class DocumentLoader:
    """Base class for document loaders."""

    def __init__(self, source: str):
        self.source = source
        self.documents: List[str] = []

    def load(self) -> List[str]:
        """Load documents from the source."""
        raise NotImplementedError

    def lazy_load(self):
        """Lazily load documents one at a time."""
        for doc in self.load():
            yield doc

class FileLoader(DocumentLoader):
    """Load documents from a file."""

    def __init__(self, file_path: str, encoding: str = "utf-8"):
        super().__init__(file_path)
        self.encoding = encoding

    def load(self) -> List[str]:
        """Read and return the file contents."""
        with open(self.source, "r", encoding=self.encoding) as f:
            content = f.read()
        self.documents = [content]
        return self.documents

def split_text(text: str, chunk_size: int = 1000) -> List[str]:
    """Split text into chunks of specified size."""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks
"#;

/// Sample text with abbreviations for sentence splitting.
const ABBREVIATION_TEXT: &str = "Dr. Smith arrived at 3 p.m. to meet with Prof. Johnson. \
They discussed the U.S. patent filing for the new A.I. system. \
The meeting lasted approx. 2 hrs. and covered topics including \
machine learning, natural language processing, and computer vision. \
Mr. Brown from the U.K. office joined via video call at 4 p.m. \
The next meeting is scheduled for Jan. 15th at the N.Y. headquarters.";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Text Splitting Example ===\n");

    // -------------------------------------------------------------------------
    // Part 1: RecursiveCharacterTextSplitter with default separators
    // -------------------------------------------------------------------------
    println!("--- Part 1: RecursiveCharacterTextSplitter ---\n");

    let splitter = RecursiveCharacterTextSplitter::new()
        .with_chunk_size(200)
        .with_chunk_overlap(30);

    println!(
        "Configuration: chunk_size={}, chunk_overlap={}",
        splitter.chunk_size(),
        splitter.chunk_overlap()
    );
    println!("Input text length: {} chars\n", SAMPLE_TEXT.len());

    let chunks = splitter.split_text(SAMPLE_TEXT);

    println!("Split into {} chunks:", chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        let preview = chunk.replace('\n', " ");
        let preview = if preview.len() > 70 {
            format!("{}...", &preview[..70])
        } else {
            preview
        };
        println!("  Chunk {}: {} chars - {}", i + 1, chunk.len(), preview);
    }
    println!();

    // Show overlap between adjacent chunks.
    if chunks.len() >= 2 {
        println!("Overlap demonstration (last 30 chars of chunk 1 vs first 30 chars of chunk 2):");
        let end_of_first = &chunks[0][chunks[0].len().saturating_sub(30)..];
        let start_of_second = &chunks[1][..chunks[1].len().min(30)];
        println!("  End of chunk 1:   \"{}\"", end_of_first);
        println!("  Start of chunk 2: \"{}\"", start_of_second);
        println!();
    }

    // -------------------------------------------------------------------------
    // Part 2: Split documents (preserves metadata)
    // -------------------------------------------------------------------------
    println!("--- Part 2: Splitting documents with metadata ---\n");

    let doc = Document::new(SAMPLE_TEXT.trim());
    let doc_chunks = splitter.split_documents(&[doc]);

    println!("Split 1 document into {} chunk documents", doc_chunks.len());
    for (i, chunk) in doc_chunks.iter().enumerate() {
        println!(
            "  Doc chunk {}: {} chars, metadata keys: {:?}",
            i + 1,
            chunk.page_content.len(),
            chunk.metadata.keys().collect::<Vec<_>>()
        );
    }
    println!();

    // -------------------------------------------------------------------------
    // Part 3: Language preset - Markdown
    // -------------------------------------------------------------------------
    println!("--- Part 3: Markdown language preset ---\n");

    let md_splitter = RecursiveCharacterTextSplitter::for_language(Language::Markdown)
        .with_chunk_size(300)
        .with_chunk_overlap(20);

    let md_chunks = md_splitter.split_text(MARKDOWN_TEXT);

    println!(
        "Markdown text ({} chars) split into {} chunks:",
        MARKDOWN_TEXT.len(),
        md_chunks.len()
    );
    for (i, chunk) in md_chunks.iter().enumerate() {
        let first_line = chunk.lines().next().unwrap_or("(empty)");
        println!(
            "  Chunk {}: {} chars - starts with: \"{}\"",
            i + 1,
            chunk.len(),
            first_line
        );
    }
    println!();

    // -------------------------------------------------------------------------
    // Part 4: Language preset - Python
    // -------------------------------------------------------------------------
    println!("--- Part 4: Python language preset ---\n");

    let py_splitter = RecursiveCharacterTextSplitter::for_language(Language::Python)
        .with_chunk_size(300)
        .with_chunk_overlap(0);

    let py_chunks = py_splitter.split_text(PYTHON_CODE);

    println!(
        "Python code ({} chars) split into {} chunks:",
        PYTHON_CODE.len(),
        py_chunks.len()
    );
    for (i, chunk) in py_chunks.iter().enumerate() {
        let first_line = chunk.lines().next().unwrap_or("(empty)");
        let last_line = chunk.lines().last().unwrap_or("(empty)");
        println!(
            "  Chunk {}: {} chars | first: \"{}\" | last: \"{}\"",
            i + 1,
            chunk.len(),
            first_line.trim(),
            last_line.trim()
        );
    }
    println!();

    // -------------------------------------------------------------------------
    // Part 5: SentenceTextSplitter with abbreviation handling
    // -------------------------------------------------------------------------
    println!("--- Part 5: SentenceTextSplitter ---\n");

    let sentence_splitter = SentenceTextSplitter::builder()
        .chunk_size(150)
        .chunk_overlap(0)
        .strip_whitespace(true)
        .build();

    println!("Input: \"{}\"", ABBREVIATION_TEXT);
    println!();

    // First show sentence detection.
    let sentences = sentence_splitter.split_into_sentences(ABBREVIATION_TEXT);
    println!("Detected {} sentences:", sentences.len());
    for (i, sentence) in sentences.iter().enumerate() {
        println!("  {}: \"{}\"", i + 1, sentence.trim());
    }
    println!();

    // Then show chunking.
    let sentence_chunks = sentence_splitter.split_text(ABBREVIATION_TEXT);
    println!(
        "Chunked into {} chunks (max 150 chars each):",
        sentence_chunks.len()
    );
    for (i, chunk) in sentence_chunks.iter().enumerate() {
        println!("  Chunk {}: {} chars - \"{}\"", i + 1, chunk.len(), chunk);
    }

    println!("\nDone!");
    Ok(())
}
