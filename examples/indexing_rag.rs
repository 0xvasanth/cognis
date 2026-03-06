//! Indexing RAG Pipeline Example
//!
//! Demonstrates a full Retrieval-Augmented Generation workflow with incremental
//! indexing: load documents -> split -> index (with deduplication) -> retrieve -> generate.
//! Uses IndexingPipeline with InMemoryRecordManager for tracking indexed content,
//! InMemoryVectorStore for storage, and RetrievalQAChain for question answering.
//!
//! No API keys required -- uses DeterministicFakeEmbedding and FakeListChatModel.
//!
//! Run with: cargo run -p rustchain-examples --example indexing_rag

use std::sync::Arc;

use rustchain::chains::RetrievalQAChain;
use rustchain::indexing::{CleanupMode, InMemoryRecordManager, IndexingPipeline};
use rustchain::text_splitter::{RecursiveCharacterTextSplitter, TextSplitter};
use rustchain_core::documents::Document;
use rustchain_core::embeddings::Embeddings;
use rustchain_core::embeddings_fake::DeterministicFakeEmbedding;
use rustchain_core::language_models::chat_model::BaseChatModel;
use rustchain_core::language_models::FakeListChatModel;
use rustchain_core::retrievers::BaseRetriever;
use rustchain_core::vectorstores::base::{SearchType, VectorStore};
use rustchain_core::vectorstores::in_memory::InMemoryVectorStore;

/// Sample documents about different programming topics.
const DOC_RUST: &str = "\
Rust is a systems programming language focused on safety, speed, and concurrency. \
It achieves memory safety without garbage collection through its ownership system. \
The borrow checker enforces strict rules about references at compile time. \
Each value has exactly one owner, and when the owner goes out of scope the value is dropped.";

const DOC_TOKIO: &str = "\
Tokio is an asynchronous runtime for the Rust programming language. \
It provides building blocks needed for writing network applications. \
Tokio gives the flexibility to target a wide range of systems, from large servers \
with dozens of cores to small embedded devices. At a high level, Tokio provides \
a multithreaded runtime for executing asynchronous code, an async version of the \
standard library, and a large ecosystem of libraries.";

const DOC_SERDE: &str = "\
Serde is a framework for serializing and deserializing Rust data structures \
efficiently and generically. The serde ecosystem consists of data structures \
that know how to serialize and deserialize themselves, and data formats that \
know how to serialize and deserialize other things. Serde provides the layer \
by which these two groups interact with each other.";

const DOC_CARGO: &str = "\
Cargo is Rust's build system and package manager. It handles downloading \
dependencies, compiling packages, making distributable packages, and uploading \
them to crates.io. Cargo.toml is the manifest file that contains metadata \
and dependency specifications. Cargo.lock ensures reproducible builds by \
recording exact dependency versions.";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Indexing RAG Pipeline Example ===\n");

    // -------------------------------------------------------------------------
    // Step 1: Set up the vector store and embedding model
    // -------------------------------------------------------------------------
    println!("--- Step 1: Setting up vector store ---\n");

    let embedding: Arc<dyn Embeddings> = Arc::new(DeterministicFakeEmbedding::new(64));
    let store = Arc::new(InMemoryVectorStore::new(embedding));

    println!("  Created InMemoryVectorStore with DeterministicFakeEmbedding (dim=64)\n");

    // -------------------------------------------------------------------------
    // Step 2: Create the indexing pipeline
    // -------------------------------------------------------------------------
    println!("--- Step 2: Creating indexing pipeline ---\n");

    let splitter = RecursiveCharacterTextSplitter::new()
        .with_chunk_size(150)
        .with_chunk_overlap(20);

    let record_manager = InMemoryRecordManager::new();

    let pipeline = IndexingPipeline::new(store.clone())
        .with_text_splitter(Box::new(splitter) as Box<dyn TextSplitter>)
        .with_record_manager(Box::new(record_manager))
        .with_cleanup_mode(CleanupMode::Incremental);

    println!("  Pipeline: split (chunk=150, overlap=20) -> deduplicate -> index");
    println!("  Cleanup mode: Incremental (removes stale documents)\n");

    // -------------------------------------------------------------------------
    // Step 3: First indexing pass -- add initial documents
    // -------------------------------------------------------------------------
    println!("--- Step 3: First indexing pass ---\n");

    let initial_docs = vec![
        Document::new(DOC_RUST),
        Document::new(DOC_TOKIO),
        Document::new(DOC_SERDE),
    ];

    println!("  Indexing {} documents...", initial_docs.len());
    let result1 = pipeline.index(initial_docs).await?;
    println!(
        "  Result: added={}, skipped={}, deleted={}",
        result1.num_added, result1.num_skipped, result1.num_deleted
    );

    // Show what's in the store.
    let all_docs = store.similarity_search("programming", 20).await?;
    println!("  Vector store now contains {} chunks\n", all_docs.len());

    // -------------------------------------------------------------------------
    // Step 4: Second indexing pass -- re-index same docs (should skip)
    // -------------------------------------------------------------------------
    println!("--- Step 4: Re-indexing same documents (deduplication test) ---\n");

    let same_docs = vec![
        Document::new(DOC_RUST),
        Document::new(DOC_TOKIO),
        Document::new(DOC_SERDE),
    ];

    let result2 = pipeline.index(same_docs).await?;
    println!(
        "  Result: added={}, skipped={}, deleted={}",
        result2.num_added, result2.num_skipped, result2.num_deleted
    );
    println!("  (All documents were skipped because content is unchanged)\n");

    // -------------------------------------------------------------------------
    // Step 5: Third indexing pass -- add new doc, remove one old doc
    // -------------------------------------------------------------------------
    println!("--- Step 5: Incremental update (add Cargo, drop Serde) ---\n");

    let updated_docs = vec![
        Document::new(DOC_RUST),
        Document::new(DOC_TOKIO),
        // Serde removed, Cargo added
        Document::new(DOC_CARGO),
    ];

    let result3 = pipeline.index(updated_docs).await?;
    println!(
        "  Result: added={}, skipped={}, deleted={}",
        result3.num_added, result3.num_skipped, result3.num_deleted
    );
    println!("  (Cargo chunks added, Rust/Tokio skipped, Serde chunks deleted)\n");

    // -------------------------------------------------------------------------
    // Step 6: Query the vector store directly
    // -------------------------------------------------------------------------
    println!("--- Step 6: Similarity search queries ---\n");

    let queries = [
        "How does Rust handle memory?",
        "What is Tokio used for?",
        "How do I manage dependencies?",
    ];

    for query in &queries {
        println!("  Query: \"{query}\"");
        let results = store.similarity_search_with_score(query, 2).await?;
        for (i, (doc, score)) in results.iter().enumerate() {
            let preview = doc.page_content.replace('\n', " ");
            let preview = if preview.len() > 70 {
                format!("{}...", &preview[..70])
            } else {
                preview
            };
            println!("    [{i}] (score={score:.4}) {preview}");
        }
        println!();
    }

    // -------------------------------------------------------------------------
    // Step 7: Build a RetrievalQAChain and answer questions
    // -------------------------------------------------------------------------
    println!("--- Step 7: RetrievalQA Chain ---\n");

    let retriever: Arc<dyn BaseRetriever> =
        Arc::new(store.as_retriever_with(SearchType::Similarity, 3));

    let llm: Arc<dyn BaseChatModel> = Arc::new(FakeListChatModel::new(vec![
        "Rust achieves memory safety through its ownership system. Each value has one owner, \
         and the borrow checker enforces reference rules at compile time. When the owner goes \
         out of scope, the value is automatically dropped."
            .into(),
        "Tokio is an asynchronous runtime for Rust. It provides a multithreaded runtime for \
         executing async code, async versions of standard library types, and a large ecosystem \
         of async libraries for networking and I/O."
            .into(),
        "You manage dependencies in Rust using Cargo, the build system and package manager. \
         Dependencies are specified in Cargo.toml, and Cargo.lock ensures reproducible builds \
         by recording exact versions."
            .into(),
    ]));

    let qa_chain = RetrievalQAChain::new(retriever, llm).with_k(3);
    println!("  Built RetrievalQAChain (k=3)\n");

    for query in &queries {
        println!("  Q: {query}");
        let answer = qa_chain.call_with_sources(query).await?;
        println!("  A: {}", answer.answer);
        println!(
            "  Sources: {} document(s) retrieved",
            answer.source_documents.len()
        );
        for (i, doc) in answer.source_documents.iter().enumerate() {
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
