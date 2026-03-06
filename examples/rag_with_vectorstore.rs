//! Full RAG Pipeline: Load -> Split -> Embed -> Store -> Retrieve -> Answer
//!
//! Demonstrates a complete Retrieval-Augmented Generation workflow using:
//! - Embedded sample text (no files needed)
//! - RecursiveCharacterTextSplitter for chunking
//! - DeterministicFakeEmbedding (no API keys required)
//! - InMemoryVectorStore for vector storage
//! - RetrievalQAChain with FakeListChatModel for answering
//!
//! Run with: cargo run -p rustchain-examples --example rag_with_vectorstore

use std::sync::Arc;

use rustchain::chains::RetrievalQAChain;
use rustchain::text_splitter::{RecursiveCharacterTextSplitter, TextSplitter};
use rustchain_core::documents::Document;
use rustchain_core::embeddings::Embeddings;
use rustchain_core::embeddings_fake::DeterministicFakeEmbedding;
use rustchain_core::language_models::chat_model::BaseChatModel;
use rustchain_core::language_models::FakeListChatModel;
use rustchain_core::retrievers::BaseRetriever;
use rustchain_core::vectorstores::base::{SearchType, VectorStore};
use rustchain_core::vectorstores::in_memory::InMemoryVectorStore;

/// Sample knowledge base about Rust programming.
const SAMPLE_TEXT: &str = r#"
Rust is a systems programming language focused on safety, speed, and concurrency.
It achieves memory safety without garbage collection through its ownership system.
The borrow checker enforces strict rules about references at compile time.

The ownership system has three rules: each value has exactly one owner, there can
only be one owner at a time, and when the owner goes out of scope the value is
dropped. This eliminates use-after-free bugs and double-free errors.

Cargo is Rust's build system and package manager. It handles downloading
dependencies, compiling packages, running tests, and generating documentation.
The Cargo.toml file defines project metadata and dependencies.

Traits in Rust are similar to interfaces in other languages. They define shared
behavior that types can implement. Trait objects enable dynamic dispatch when
the concrete type is not known at compile time.

Async programming in Rust uses the async/await syntax. The tokio runtime is the
most popular async executor. Futures in Rust are lazy and only make progress
when polled by a runtime.

Error handling in Rust uses the Result<T, E> type. The ? operator provides
ergonomic error propagation. Libraries like thiserror and anyhow simplify
custom error type definitions.
"#;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Full RAG Pipeline Example ===\n");

    // Step 1: Create documents from the sample text.
    let doc = Document::new(SAMPLE_TEXT.trim());
    println!(
        "Step 1: Created source document ({} chars)\n",
        doc.page_content.len()
    );

    // Step 2: Split into chunks using RecursiveCharacterTextSplitter.
    let splitter = RecursiveCharacterTextSplitter::new()
        .with_chunk_size(200)
        .with_chunk_overlap(30);

    let chunks = splitter.split_documents(&[doc]);
    println!(
        "Step 2: Split into {} chunks (chunk_size=200, overlap=30)",
        chunks.len()
    );
    for (i, chunk) in chunks.iter().enumerate() {
        let preview = chunk.page_content.replace('\n', " ");
        let preview = if preview.len() > 70 {
            format!("{}...", &preview[..70])
        } else {
            preview
        };
        println!(
            "  Chunk {}: {} chars - {}",
            i + 1,
            chunk.page_content.len(),
            preview
        );
    }
    println!();

    // Step 3: Create embeddings and vector store.
    //
    // DeterministicFakeEmbedding produces hash-based vectors that are consistent
    // per input text. In production, use OpenAI, Ollama, or another provider.
    let embedding: Arc<dyn Embeddings> = Arc::new(DeterministicFakeEmbedding::new(64));
    let store = Arc::new(InMemoryVectorStore::new(embedding));

    // Add chunks to the vector store.
    let ids = store.add_documents(chunks, None).await?;
    println!(
        "Step 3: Stored {} chunks in InMemoryVectorStore\n",
        ids.len()
    );

    // Step 4: Create a retriever from the vector store.
    let retriever: Arc<dyn BaseRetriever> =
        Arc::new(store.as_retriever_with(SearchType::Similarity, 3));

    // Step 5: Create a RetrievalQAChain with a fake LLM.
    //
    // The fake model returns predefined answers. In production, replace with
    // a real chat model (e.g., ChatAnthropic, ChatOpenAI).
    let llm: Arc<dyn BaseChatModel> = Arc::new(FakeListChatModel::new(vec![
        "Rust achieves memory safety through its ownership system, which has three rules: each value has exactly one owner, there can only be one owner at a time, and values are dropped when the owner goes out of scope. The borrow checker enforces reference rules at compile time.".into(),
        "Cargo is Rust's build system and package manager. It handles downloading dependencies, compiling packages, running tests, and generating documentation. Projects are configured via Cargo.toml.".into(),
        "Async programming in Rust uses async/await syntax with the tokio runtime as the most popular executor. Futures are lazy and only make progress when polled.".into(),
    ]));

    let chain = RetrievalQAChain::new(retriever, llm).with_k(3);
    println!("Step 5: Built RetrievalQAChain (k=3)\n");

    // Step 6: Query the chain.
    let queries = [
        "How does Rust handle memory safety?",
        "What is Cargo used for?",
        "How does async work in Rust?",
    ];

    println!("Step 6: Querying the RAG chain\n");

    for query in &queries {
        println!("  Q: {query}");
        let result = chain.call_with_sources(query).await?;
        println!("  A: {}", result.answer);
        println!(
            "  Sources: {} document(s) retrieved",
            result.source_documents.len()
        );
        for (i, doc) in result.source_documents.iter().enumerate() {
            let preview = doc.page_content.replace('\n', " ");
            let preview = if preview.len() > 60 {
                format!("{}...", &preview[..60])
            } else {
                preview
            };
            println!("    [{i}] {preview}");
        }
        println!();
    }

    println!("Done!");
    Ok(())
}
