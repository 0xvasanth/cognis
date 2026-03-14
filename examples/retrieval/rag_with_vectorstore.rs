//! RAG with VectorStore Example
//!
//! End-to-end RAG: embed text chunks -> store -> RetrievalQAChain answers questions.
//! No API keys required.
//!
//! Run with: cargo run -p cognis-examples --example rag_with_vectorstore

#[path = "../shared.rs"]
mod shared;

use cognis::chains::RetrievalQAChain;
use cognis::text_splitter::RecursiveCharacterTextSplitter;
use cognis_core::documents::Document;
use cognis_core::embeddings::Embeddings;
use cognis_core::embeddings_fake::DeterministicFakeEmbedding;
use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::retrievers::BaseRetriever;
use cognis_core::vectorstores::base::{SearchType, VectorStore};
use cognis_core::vectorstores::in_memory::InMemoryVectorStore;
use std::sync::Arc;

const SAMPLE_TEXT: &str = "\
Rust is a systems programming language focused on safety, speed, and concurrency.
It achieves memory safety without garbage collection through its ownership system.
The borrow checker enforces strict rules about references at compile time.

Cargo is Rust's build system and package manager. It handles downloading
dependencies, compiling packages, running tests, and generating documentation.

Async programming in Rust uses the async/await syntax. The tokio runtime is the
most popular async executor. Futures are lazy and only make progress when polled.

Error handling in Rust uses the Result<T, E> type. The ? operator provides
ergonomic error propagation.";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Split source text into chunks.
    let chunks = RecursiveCharacterTextSplitter::new()
        .with_chunk_size(200)
        .with_chunk_overlap(30)
        .split_documents(&[Document::new(SAMPLE_TEXT.trim())]);
    println!("Split into {} chunks", chunks.len());

    // Embed and store chunks.
    let embedding: Arc<dyn Embeddings> = Arc::new(DeterministicFakeEmbedding::new(64));
    let store = Arc::new(InMemoryVectorStore::new(embedding));
    store.add_documents(chunks, None).await?;

    // Build RetrievalQAChain.
    let retriever: Arc<dyn BaseRetriever> =
        Arc::new(store.as_retriever_with(SearchType::Similarity, 3));

    let llm: Arc<dyn BaseChatModel> = shared::get_chat_model(vec![
        "Rust achieves memory safety through its ownership system and borrow checker.".into(),
        "Cargo handles dependencies, compilation, testing, and documentation.".into(),
        "Async Rust uses async/await with the tokio runtime; futures are lazy.".into(),
    ]);

    let chain = RetrievalQAChain::new(retriever, llm).with_k(3);

    // Query the chain.
    for query in [
        "How does Rust handle memory safety?",
        "What is Cargo used for?",
        "How does async work in Rust?",
    ] {
        let result = chain.call_with_sources(query).await?;
        println!("\nQ: {query}");
        println!("A: {}", result.answer);
        println!("Sources: {} document(s)", result.source_documents.len());
    }

    Ok(())
}
