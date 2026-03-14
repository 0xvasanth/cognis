//! RAG Pipeline Example
//!
//! Full RAG workflow: TextLoader -> split -> embed -> store -> retrieve -> LLM answer.
//! No API keys required -- uses deterministic fake embeddings.
//!
//! Run with: cargo run -p cognis-examples --example rag_pipeline

#[path = "../shared.rs"]
mod shared;

use cognis::document_loaders::text::TextLoader;
use cognis::text_splitter::RecursiveCharacterTextSplitter;
use cognis_core::document_loaders::BaseLoader;
use cognis_core::embeddings::Embeddings;
use cognis_core::embeddings_fake::DeterministicFakeEmbedding;
use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::{HumanMessage, Message, SystemMessage};
use cognis_core::retrievers::BaseRetriever;
use cognis_core::vectorstores::base::{SearchType, VectorStore};
use cognis_core::vectorstores::in_memory::InMemoryVectorStore;
use std::io::Write;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a temporary file with sample content.
    let mut tmp = tempfile::NamedTempFile::new()?;
    write!(
        tmp,
        "{}",
        concat!(
        "Rust is a systems programming language focused on safety, speed, and concurrency.\n",
        "It achieves memory safety without garbage collection through its ownership system.\n\n",
        "The ownership system has three rules: each value has an owner, there can only be\n",
        "one owner at a time, and when the owner goes out of scope the value is dropped.\n\n",
        "Cargo is Rust's build system and package manager. It handles downloading\n",
        "dependencies, compiling packages, and running tests.\n\n",
        "Async programming in Rust uses the async/await syntax with executors like tokio.\n",
        "Futures are lazy and only make progress when polled by a runtime."
    )
    )?;

    // Load and split.
    let docs = TextLoader::new(tmp.path()).load().await?;
    let splitter = RecursiveCharacterTextSplitter::new()
        .with_chunk_size(200)
        .with_chunk_overlap(30);
    let chunks = splitter.split_documents(&docs);
    println!(
        "Loaded {} doc(s), split into {} chunks",
        docs.len(),
        chunks.len()
    );

    // Embed and store.
    let embedding: Arc<dyn Embeddings> = Arc::new(DeterministicFakeEmbedding::new(64));
    let store = Arc::new(InMemoryVectorStore::new(embedding));
    store.add_documents(chunks, None).await?;

    // Retrieve relevant documents.
    let retriever = store.as_retriever_with(SearchType::Similarity, 3);
    let query = "How does Rust achieve memory safety?";
    let retrieved = retriever.get_relevant_documents(query).await?;
    println!("Retrieved {} documents for: \"{query}\"", retrieved.len());

    // Build context and ask the LLM.
    let context: String = retrieved
        .iter()
        .enumerate()
        .map(|(i, doc)| format!("[{}] {}", i + 1, doc.page_content.replace('\n', " ")))
        .collect::<Vec<_>>()
        .join("\n");

    let model: Arc<dyn BaseChatModel> = shared::get_chat_model(vec![
        "Rust achieves memory safety through its ownership system: each value has one owner, \
         and the borrow checker enforces reference rules at compile time."
            .into(),
    ]);

    let messages = vec![
        Message::System(SystemMessage::new(
            "Answer the user's question based only on the provided context. Be concise.",
        )),
        Message::Human(HumanMessage::new(&format!(
            "Context:\n{context}\n\nQuestion: {query}"
        ))),
    ];

    let result = model._generate(&messages, None).await?;
    println!("\nQ: {query}");
    println!("A: {}", result.generations[0].text);

    Ok(())
}
