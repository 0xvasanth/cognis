//! Indexing RAG Pipeline Example
//!
//! Demonstrates incremental indexing with deduplication, then RAG question answering.
//! Uses InMemoryRecordManager, InMemoryVectorStore, and RetrievalQAChain.
//!
//! Run with: cargo run -p cognis-examples --example indexing_rag

#[path = "../shared.rs"]
mod shared;

use cognis::chains::RetrievalQAChain;
use cognis::indexing::{CleanupMode, InMemoryRecordManager, IndexingPipeline};
use cognis::text_splitter::{RecursiveCharacterTextSplitter, TextSplitter};
use cognis_core::documents::Document;
use cognis_core::embeddings::Embeddings;
use cognis_core::embeddings_fake::DeterministicFakeEmbedding;
use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::retrievers::BaseRetriever;
use cognis_core::vectorstores::base::{SearchType, VectorStore};
use cognis_core::vectorstores::in_memory::InMemoryVectorStore;
use std::sync::Arc;

const DOC_RUST: &str = "Rust is a systems programming language focused on safety, speed, and \
    concurrency. It achieves memory safety without garbage collection through its ownership system.";

const DOC_TOKIO: &str = "Tokio is an asynchronous runtime for Rust. It provides a multithreaded \
    runtime for executing async code and a large ecosystem of libraries.";

const DOC_SERDE: &str = "Serde is a framework for serializing and deserializing Rust data \
    structures efficiently and generically.";

const DOC_CARGO: &str = "Cargo is Rust's build system and package manager. Cargo.toml contains \
    metadata and dependency specifications. Cargo.lock ensures reproducible builds.";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let embedding: Arc<dyn Embeddings> = Arc::new(DeterministicFakeEmbedding::new(64));
    let store = Arc::new(InMemoryVectorStore::new(embedding));

    let splitter = RecursiveCharacterTextSplitter::new()
        .with_chunk_size(150)
        .with_chunk_overlap(20);

    let pipeline = IndexingPipeline::new(store.clone())
        .with_text_splitter(Box::new(splitter) as Box<dyn TextSplitter>)
        .with_record_manager(Box::new(InMemoryRecordManager::new()))
        .with_cleanup_mode(CleanupMode::Incremental);

    // First pass: index three documents.
    let initial_docs = vec![
        Document::new(DOC_RUST),
        Document::new(DOC_TOKIO),
        Document::new(DOC_SERDE),
    ];
    let r1 = pipeline.index(initial_docs).await?;
    println!(
        "Pass 1: added={}, skipped={}, deleted={}",
        r1.num_added, r1.num_skipped, r1.num_deleted
    );

    // Second pass: same docs should be skipped (deduplication).
    let same_docs = vec![
        Document::new(DOC_RUST),
        Document::new(DOC_TOKIO),
        Document::new(DOC_SERDE),
    ];
    let r2 = pipeline.index(same_docs).await?;
    println!(
        "Pass 2: added={}, skipped={}, deleted={}",
        r2.num_added, r2.num_skipped, r2.num_deleted
    );

    // Third pass: replace Serde with Cargo.
    let updated_docs = vec![
        Document::new(DOC_RUST),
        Document::new(DOC_TOKIO),
        Document::new(DOC_CARGO),
    ];
    let r3 = pipeline.index(updated_docs).await?;
    println!(
        "Pass 3: added={}, skipped={}, deleted={}",
        r3.num_added, r3.num_skipped, r3.num_deleted
    );

    // Query with RetrievalQAChain.
    let retriever: Arc<dyn BaseRetriever> =
        Arc::new(store.as_retriever_with(SearchType::Similarity, 3));

    let llm: Arc<dyn BaseChatModel> = shared::get_chat_model(vec![
        "Rust achieves memory safety through its ownership system and borrow checker.".into(),
        "Tokio is an async runtime providing a multithreaded executor and I/O libraries.".into(),
        "Cargo manages dependencies via Cargo.toml and ensures reproducible builds.".into(),
    ]);

    let qa_chain = RetrievalQAChain::new(retriever, llm).with_k(3);

    for query in [
        "How does Rust handle memory?",
        "What is Tokio?",
        "How do I manage dependencies?",
    ] {
        let answer = qa_chain.call_with_sources(query).await?;
        println!("\nQ: {query}");
        println!("A: {}", answer.answer);
    }

    Ok(())
}
