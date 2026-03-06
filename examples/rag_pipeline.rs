//! RAG Pipeline Example
//!
//! Demonstrates a full Retrieval-Augmented Generation (RAG) workflow:
//! TextLoader -> RecursiveCharacterTextSplitter -> FakeEmbeddings ->
//! InMemoryVectorStore -> VectorStoreRetriever -> similarity search.
//!
//! No API keys required -- uses deterministic fake embeddings.

use std::io::Write;
use std::sync::Arc;

use rustchain::document_loaders::text::TextLoader;
use rustchain::text_splitter::{RecursiveCharacterTextSplitter, TextSplitter};
use rustchain_core::document_loaders::BaseLoader;
use rustchain_core::embeddings_fake::DeterministicFakeEmbedding;
use rustchain_core::retrievers::BaseRetriever;
use rustchain_core::vectorstores::base::{SearchType, VectorStore};
use rustchain_core::vectorstores::in_memory::InMemoryVectorStore;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RAG Pipeline Example ===\n");

    // Step 1: Create a temporary file with sample text content.
    //
    // In a real application, this could be any text file, PDF, or web page.
    let mut tmp = tempfile::NamedTempFile::new()?;
    let sample_text = r#"Rust is a systems programming language focused on safety, speed, and concurrency.
It achieves memory safety without garbage collection through its ownership system.

The ownership system has three rules: each value has an owner, there can only be
one owner at a time, and when the owner goes out of scope the value is dropped.

Rust's type system and borrow checker catch many common bugs at compile time.
This eliminates data races, null pointer dereferences, and buffer overflows.

Cargo is Rust's build system and package manager. It handles downloading
dependencies, compiling packages, and running tests.

Traits in Rust are similar to interfaces in other languages. They define
shared behavior that types can implement. Trait objects enable dynamic dispatch.

Async programming in Rust uses the async/await syntax with executors like tokio.
Futures are lazy and only make progress when polled by a runtime."#;

    write!(tmp, "{}", sample_text)?;
    let file_path = tmp.path().to_path_buf();

    println!(
        "Step 1: Created temp file with {} bytes of content\n",
        sample_text.len()
    );

    // Step 2: Load the text file using TextLoader.
    let loader = TextLoader::new(&file_path);
    let docs = loader.load().await?;

    println!("Step 2: Loaded {} document(s) from file", docs.len());
    println!("  First doc length: {} chars\n", docs[0].page_content.len());

    // Step 3: Split documents using RecursiveCharacterTextSplitter.
    //
    // This splits the text into overlapping chunks, trying paragraph breaks first,
    // then line breaks, then spaces, then character-level.
    let splitter = RecursiveCharacterTextSplitter::new()
        .with_chunk_size(200)
        .with_chunk_overlap(30);

    let chunks = splitter.split_documents(&docs);

    println!(
        "Step 3: Split into {} chunks (chunk_size=200, overlap=30)",
        chunks.len()
    );
    for (i, chunk) in chunks.iter().enumerate() {
        println!(
            "  Chunk {}: {} chars - {:?}...",
            i + 1,
            chunk.page_content.len(),
            &chunk.page_content[..chunk.page_content.len().min(60)]
        );
    }
    println!();

    // Step 4: Create fake embeddings and an in-memory vector store.
    //
    // DeterministicFakeEmbedding produces hash-based embeddings that are
    // deterministic per text input. In production, use OpenAI, Ollama, etc.
    let embedding: Arc<dyn rustchain_core::embeddings::Embeddings> =
        Arc::new(DeterministicFakeEmbedding::new(64));

    let store = Arc::new(InMemoryVectorStore::new(embedding));

    // Add the document chunks to the vector store.
    let ids = store.add_documents(chunks, None).await?;

    println!(
        "Step 4: Stored {} document chunks in InMemoryVectorStore\n",
        ids.len()
    );

    // Step 5: Perform similarity search queries.
    println!("Step 5: Similarity Search Results\n");

    let queries = vec![
        "What is the ownership system?",
        "How does async work in Rust?",
        "What is Cargo?",
    ];

    for query in &queries {
        println!("  Query: \"{query}\"");
        let results = store.similarity_search_with_score(query, 2).await?;
        for (i, (doc, score)) in results.iter().enumerate() {
            let preview = doc.page_content.replace('\n', " ");
            let preview = if preview.len() > 80 {
                format!("{}...", &preview[..80])
            } else {
                preview
            };
            println!("    Result {}: (score={:.4}) {}", i + 1, score, preview);
        }
        println!();
    }

    // Step 6: Use the VectorStoreRetriever for retrieval.
    //
    // The retriever wraps the vector store and implements BaseRetriever,
    // which can be used in LCEL chains.
    println!("Step 6: Using VectorStoreRetriever (k=3)\n");

    let retriever = store.as_retriever_with(SearchType::Similarity, 3);
    let retrieved = retriever.get_relevant_documents("memory safety").await?;

    println!("  Query: \"memory safety\"");
    println!("  Retrieved {} documents:", retrieved.len());
    for (i, doc) in retrieved.iter().enumerate() {
        let preview = doc.page_content.replace('\n', " ");
        let preview = if preview.len() > 80 {
            format!("{}...", &preview[..80])
        } else {
            preview
        };
        println!("    {}: {}", i + 1, preview);
    }

    println!("\nDone!");
    Ok(())
}
