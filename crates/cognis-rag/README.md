# cognis-rag

Retrieval-Augmented Generation (RAG) primitives for Cognis v2, including embeddings, vector stores, and document processing.

## Purpose
`cognis-rag` provides the building blocks for creating RAG pipelines. It includes tools for loading documents from various formats, splitting them into manageable chunks, generating vector embeddings, and storing/retrieving them from vector databases.

## Key Features
- **Document Loaders**: Support for Text, Markdown, JSON, CSV, PDF, HTML, and directory-based loading.
- **Text Splitters**: Smart splitting strategies including `RecursiveCharSplitter` and `MarkdownSplitter`.
- **Embeddings**: Interface for generating text embeddings with support for OpenAI, Ollama, and more.
- **Vector Stores**: Pluggable vector database support (Chroma, Qdrant, Pinecone, Weaviate) plus a fast `InMemoryVectorStore`.
- **Retrievers**: Advanced retrieval logic like `VectorRetriever`, `BM25Retriever`, and `EnsembleRetriever`.
- **Indexing Pipeline**: A streamlined way to load, split, embed, and store documents in one operation.

## Usage
Add this to your `Cargo.toml`:
```toml
[dependencies]
cognis-rag = "0.1.0"
```

### Basic Example: Vector Search
```rust
use cognis_rag::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let store = InMemoryVectorStore::default();
    let embeddings = FakeEmbeddings::new(1536); // Replace with OpenAIEmbeddings etc.
    
    // Add documents
    let docs = vec![
        Document::new("Rust is a systems programming language."),
        Document::new("Python is great for data science."),
    ];
    store.add_documents(docs, &embeddings).await?;
    
    // Search
    let retriever = VectorRetriever::new(store, embeddings);
    let results = retriever.invoke("What is Rust?").await?;
    
    for doc in results {
        println!("Found: {}", doc.page_content);
    }
    Ok(())
}
```
