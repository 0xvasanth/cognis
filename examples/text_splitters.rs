//! Text Splitters Example
//!
//! Demonstrates the various text splitting strategies available in Cognis
//! for chunking documents before embedding or retrieval. Each splitter targets
//! a different content type (plain text, markdown, code) and supports
//! configurable chunk size, overlap, and whitespace handling.
//!
//! Features shown:
//! - SplitConfig customization (chunk_size, chunk_overlap, strip_whitespace)
//! - CharacterTextSplitter with default and custom separators
//! - RecursiveCharacterTextSplitter with multiple separator levels
//! - TokenTextSplitter for word-based splitting
//! - MarkdownTextSplitter for splitting by headers
//! - CodeTextSplitter for Rust and Python code
//! - Document splitting with metadata preservation
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example text_splitters`

mod shared;
use std::collections::HashMap;

use cognis::text_splitters::{
    CharacterTextSplitter, CodeLanguage, CodeTextSplitter, MarkdownTextSplitter,
    RecursiveCharacterTextSplitter, SplitConfig, TextSplitter, TokenTextSplitter,
};
use cognis_core::documents::Document;
use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::Message;
use serde_json::json;

fn print_chunks(chunks: &[String]) {
    for (i, chunk) in chunks.iter().enumerate() {
        let preview = if chunk.len() > 80 {
            format!("{}...", &chunk[..77])
        } else {
            chunk.clone()
        };
        println!("  [{}] ({} chars) {:?}", i, chunk.len(), preview);
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Text Splitters Example ===\n");

    // -------------------------------------------------------------------
    // 1. SplitConfig Customization
    // -------------------------------------------------------------------
    println!("--- 1. SplitConfig Customization ---");
    println!("SplitConfig controls chunk_size, chunk_overlap, strip_whitespace,");
    println!("keep_separator, and the length function used for measurement.\n");

    let default_config = SplitConfig::new();
    println!("Default config:");
    println!("  chunk_size:       {}", default_config.chunk_size);
    println!("  chunk_overlap:    {}", default_config.chunk_overlap);
    println!("  strip_whitespace: {}", default_config.strip_whitespace);
    println!("  keep_separator:   {}", default_config.keep_separator);

    let custom_config = SplitConfig::new()
        .with_chunk_size(100)
        .with_chunk_overlap(20)
        .with_strip_whitespace(false)
        .with_keep_separator(true);

    println!("\nCustom config:");
    println!("  chunk_size:       {}", custom_config.chunk_size);
    println!("  chunk_overlap:    {}", custom_config.chunk_overlap);
    println!("  strip_whitespace: {}", custom_config.strip_whitespace);
    println!("  keep_separator:   {}", custom_config.keep_separator);

    // -------------------------------------------------------------------
    // 2. CharacterTextSplitter
    // -------------------------------------------------------------------
    println!("\n--- 2. CharacterTextSplitter ---");
    println!("Splits text on a single separator string.\n");

    let text = "First paragraph of the document.\n\n\
                Second paragraph with more detail.\n\n\
                Third paragraph wrapping up the content.\n\n\
                Fourth paragraph as an epilogue.";

    // Default separator: "\n\n"
    let splitter = CharacterTextSplitter::new()
        .with_chunk_size(60)
        .with_chunk_overlap(0);
    let chunks = splitter.split_text(text);
    println!("Default separator (double newline), chunk_size=60:");
    print_chunks(&chunks);

    // Custom separator: ". "
    let csv_text = "apple|banana|cherry|date|elderberry|fig|grape|honeydew";
    let splitter = CharacterTextSplitter::new()
        .with_separator("|")
        .with_chunk_size(25)
        .with_chunk_overlap(0);
    let chunks = splitter.split_text(csv_text);
    println!("\nCustom separator '|', chunk_size=25:");
    print_chunks(&chunks);

    // With overlap
    let splitter = CharacterTextSplitter::new()
        .with_separator(" ")
        .with_chunk_size(20)
        .with_chunk_overlap(8);
    let overlap_text = "one two three four five six seven eight";
    let chunks = splitter.split_text(overlap_text);
    println!("\nSeparator=' ', chunk_size=20, chunk_overlap=8:");
    print_chunks(&chunks);

    // -------------------------------------------------------------------
    // 3. RecursiveCharacterTextSplitter
    // -------------------------------------------------------------------
    println!("\n--- 3. RecursiveCharacterTextSplitter ---");
    println!("Tries separators in order, recursing on oversized chunks.");
    println!("Default separators: [\"\\n\\n\", \"\\n\", \" \", \"\"]\n");

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
    println!("chunk_size=80, chunk_overlap=10:");
    print_chunks(&chunks);

    // Custom separators
    let splitter = RecursiveCharacterTextSplitter::new()
        .with_separators(vec!["---", "\n", " "])
        .with_chunk_size(40)
        .with_chunk_overlap(0);
    let sectioned = "Introduction to Rust---Ownership model explained---Lifetimes and borrowing";
    let chunks = splitter.split_text(sectioned);
    println!("\nCustom separators [\"---\", \"\\n\", \" \"], chunk_size=40:");
    print_chunks(&chunks);

    // -------------------------------------------------------------------
    // 4. TokenTextSplitter
    // -------------------------------------------------------------------
    println!("\n--- 4. TokenTextSplitter ---");
    println!("Splits by word count (whitespace-delimited tokens).\n");

    let prose = "The quick brown fox jumps over the lazy dog. \
                 Pack my box with five dozen liquor jugs. \
                 How vexingly quick daft zebras jump.";

    let splitter = TokenTextSplitter::new()
        .with_chunk_size(6)
        .with_chunk_overlap(0);
    let chunks = splitter.split_text(prose);
    println!("chunk_size=6 tokens, no overlap:");
    print_chunks(&chunks);

    let splitter = TokenTextSplitter::new()
        .with_chunk_size(6)
        .with_chunk_overlap(2);
    let chunks = splitter.split_text(prose);
    println!("\nchunk_size=6 tokens, chunk_overlap=2:");
    print_chunks(&chunks);

    // -------------------------------------------------------------------
    // 5. MarkdownTextSplitter
    // -------------------------------------------------------------------
    println!("\n--- 5. MarkdownTextSplitter ---");
    println!("Splits markdown by header levels, then falls back to recursive splitting.\n");

    let markdown = "\
# Getting Started

Welcome to the project. This guide walks you through setup.

## Installation

Install with cargo:

```
cargo add cognis
```

## Configuration

Create a config file in your project root.

### Environment Variables

Set API keys via environment variables for security.

### Config File

Alternatively, use a TOML configuration file.

## Usage

Import the crate and create your first chain.";

    let splitter = MarkdownTextSplitter::new()
        .with_chunk_size(80)
        .with_chunk_overlap(0);
    let chunks = splitter.split_text(markdown);
    println!("chunk_size=80, chunk_overlap=0:");
    print_chunks(&chunks);

    // -------------------------------------------------------------------
    // 6. CodeTextSplitter
    // -------------------------------------------------------------------
    println!("\n--- 6. CodeTextSplitter ---");
    println!("Splits code using language-specific separators.\n");

    // Rust code
    let rust_code = r#"
pub struct Config {
    pub name: String,
    pub version: u32,
}

impl Config {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            version: 1,
        }
    }
}

fn process(config: &Config) {
    println!("Processing {}", config.name);
}

fn validate(config: &Config) -> bool {
    !config.name.is_empty()
}"#;

    let splitter = CodeTextSplitter::new()
        .with_language(CodeLanguage::Rust)
        .with_chunk_size(80)
        .with_chunk_overlap(0);
    let chunks = splitter.split_text(rust_code);
    println!("Rust code, chunk_size=80:");
    print_chunks(&chunks);

    // Python code
    let python_code = r#"
class Calculator:
    def __init__(self):
        self.result = 0

    def add(self, x):
        self.result += x
        return self

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b"#;

    let splitter = CodeTextSplitter::new()
        .with_language(CodeLanguage::Python)
        .with_chunk_size(80)
        .with_chunk_overlap(0);
    let chunks = splitter.split_text(python_code);
    println!("\nPython code, chunk_size=80:");
    print_chunks(&chunks);

    // -------------------------------------------------------------------
    // 7. Document Splitting with Metadata Preservation
    // -------------------------------------------------------------------
    println!("\n--- 7. Document Splitting with Metadata Preservation ---");
    println!("All splitters implement split_documents(), which preserves metadata.\n");

    let mut meta1 = HashMap::new();
    meta1.insert("source".to_string(), json!("chapter1.txt"));
    meta1.insert("author".to_string(), json!("Alice"));

    let mut meta2 = HashMap::new();
    meta2.insert("source".to_string(), json!("chapter2.txt"));
    meta2.insert("author".to_string(), json!("Bob"));

    let docs = vec![
        Document::new(
            "Rust provides memory safety without garbage collection. \
             The borrow checker enforces ownership rules at compile time.",
        )
        .with_metadata(meta1),
        Document::new(
            "Tokio is an async runtime for Rust. \
             It powers many production web services and networking tools.",
        )
        .with_metadata(meta2),
    ];

    let splitter = RecursiveCharacterTextSplitter::new()
        .with_chunk_size(50)
        .with_chunk_overlap(0);
    let split_docs = splitter.split_documents(docs);

    println!(
        "Split 2 documents into {} chunks (chunk_size=50):\n",
        split_docs.len()
    );
    for (i, doc) in split_docs.iter().enumerate() {
        let source = doc
            .metadata
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let author = doc
            .metadata
            .get("author")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let preview = if doc.page_content.len() > 60 {
            format!("{}...", &doc.page_content[..57])
        } else {
            doc.page_content.clone()
        };
        println!(
            "  [{}] source={:?}, author={:?} -> {:?}",
            i, source, author, preview
        );
    }

    // -------------------------------------------------------------------
    // 8. Real LLM Demo — Summarize split text chunks
    // -------------------------------------------------------------------
    println!("\n--- 8. Real LLM Demo ---");
    println!("Ask an LLM to summarize the first text chunk.\n");

    let model = shared::get_chat_model(vec![
        "This chunk introduces Rust as a fast and safe systems programming language with key features like ownership, borrowing, and the Cargo package manager.".into(),
    ]);

    let demo_splitter = RecursiveCharacterTextSplitter::new()
        .with_chunk_size(80)
        .with_chunk_overlap(10);
    let demo_chunks = demo_splitter.split_text(article);
    if let Some(first_chunk) = demo_chunks.first() {
        let messages = vec![
            Message::system("Summarize the following text chunk in one sentence."),
            Message::human(first_chunk),
        ];
        let result = model._generate(&messages, None).await?;
        if let Some(gen) = result.generations.first() {
            println!("Chunk: {:?}", first_chunk);
            println!("LLM summary: {}", gen.message.content().text());
        }
    }

    println!("\n=== Text Splitters Example Complete ===");
    Ok(())
}
