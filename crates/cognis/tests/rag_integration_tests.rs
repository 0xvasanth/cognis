//! End-to-end RAG pipeline integration tests.
//!
//! These tests exercise the full Retrieval-Augmented Generation pipeline:
//! document loading, text splitting, embedding, vector storage, retrieval,
//! and chain-based question answering -- all using fake/in-memory components
//! so no external services are required.

use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;

use cognis::chains::conversation_retrieval::ConversationalRetrievalChain;
use cognis::chains::retrieval::RetrievalQAChain;
use cognis::document_loaders::text::TextLoader;
use cognis::text_splitter::{CharacterTextSplitter, RecursiveCharacterTextSplitter, TextSplitter};
use cognis::vectorstores::in_memory::InMemoryVectorStore;
use cognis_core::document_loaders::BaseLoader;
use cognis_core::documents::Document;
use cognis_core::embeddings::Embeddings;
use cognis_core::embeddings_fake::DeterministicFakeEmbedding;
use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::language_models::fake::{FakeListChatModel, ParrotFakeChatModel};
use cognis_core::vectorstores::base::{SearchType, VectorStore, VectorStoreRetriever};
use serde_json::Value;
use tempfile::NamedTempFile;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn fake_embeddings() -> Arc<dyn Embeddings> {
    Arc::new(DeterministicFakeEmbedding::new(128))
}

fn fake_llm(responses: Vec<&str>) -> Arc<dyn BaseChatModel> {
    Arc::new(FakeListChatModel::new(
        responses.into_iter().map(String::from).collect(),
    ))
}

fn parrot_llm() -> Arc<dyn BaseChatModel> {
    Arc::new(ParrotFakeChatModel::new())
}

fn make_retriever(store: Arc<dyn VectorStore>) -> Arc<VectorStoreRetriever> {
    Arc::new(VectorStoreRetriever::from_vectorstore(store))
}

fn make_retriever_with_k(store: Arc<dyn VectorStore>, k: usize) -> Arc<VectorStoreRetriever> {
    Arc::new(VectorStoreRetriever::new(store, SearchType::Similarity, k))
}

// ---------------------------------------------------------------------------
// 1. Full RAG pipeline with TextLoader
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_full_rag_pipeline_with_text_loader() {
    // Create a temporary text file
    let mut tmp = NamedTempFile::new().unwrap();
    write!(
        tmp,
        "Rust is a systems programming language focused on safety and performance.\n\
         It was first released in 2015.\n\
         Rust uses a borrow checker to ensure memory safety without a garbage collector."
    )
    .unwrap();

    // Step 1: Load the document
    let loader = TextLoader::new(tmp.path());
    let docs = loader.load().await.unwrap();
    assert_eq!(docs.len(), 1);
    assert!(docs[0].page_content.contains("Rust"));

    // Step 2: Split into chunks
    let splitter = RecursiveCharacterTextSplitter::new()
        .with_chunk_size(80)
        .with_chunk_overlap(10);
    let chunks = splitter.split_documents(&docs);
    assert!(
        chunks.len() > 1,
        "Expected multiple chunks, got {}",
        chunks.len()
    );

    // Step 3: Embed and store in InMemoryVectorStore
    let embeddings = fake_embeddings();
    let store = Arc::new(
        InMemoryVectorStore::from_documents(chunks.clone(), embeddings)
            .await
            .unwrap(),
    );

    // Step 4: Verify similarity search works
    let results = store.similarity_search("borrow checker", 2).await.unwrap();
    assert!(!results.is_empty());

    // Step 5: Create RetrievalQAChain and query
    let retriever = make_retriever(store.clone() as Arc<dyn VectorStore>);
    let llm = fake_llm(vec!["Rust uses a borrow checker for memory safety."]);
    let chain = RetrievalQAChain::new(retriever, llm).with_k(2);
    let answer = chain
        .call("How does Rust ensure memory safety?")
        .await
        .unwrap();
    assert_eq!(answer, "Rust uses a borrow checker for memory safety.");
}

// ---------------------------------------------------------------------------
// 2. RAG with multiple documents
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_rag_with_multiple_documents() {
    let docs = vec![
        Document::new("Python is an interpreted, high-level programming language."),
        Document::new("Rust is a systems programming language focused on safety."),
        Document::new("JavaScript is the language of the web browser."),
        Document::new("Go is a statically typed language designed at Google."),
    ];

    let embeddings = fake_embeddings();
    let store = Arc::new(
        InMemoryVectorStore::from_documents(docs, embeddings)
            .await
            .unwrap(),
    );

    // The deterministic fake embeddings are hash-based, so searching for a text
    // that exactly matches a stored document should return that document first.
    let results = store
        .similarity_search(
            "Rust is a systems programming language focused on safety.",
            1,
        )
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].page_content.contains("Rust"));

    // Full chain query
    let retriever = make_retriever_with_k(store.clone() as Arc<dyn VectorStore>, 2);
    let llm = fake_llm(vec!["Rust focuses on safety and performance."]);
    let chain = RetrievalQAChain::new(retriever, llm);
    let result = chain.call_with_sources("Tell me about Rust").await.unwrap();
    assert_eq!(result.answer, "Rust focuses on safety and performance.");
    assert_eq!(result.source_documents.len(), 2);
}

// ---------------------------------------------------------------------------
// 3. RAG with text splitter
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_rag_with_text_splitter() {
    let long_text = "Artificial intelligence is transforming every industry. \
        Machine learning models can now generate text, images, and code. \
        Large language models like GPT and Claude have billions of parameters. \
        Transformer architectures use self-attention mechanisms. \
        Embeddings map words and sentences into dense vector spaces. \
        Retrieval-augmented generation combines search with language models. \
        Vector databases store and index high-dimensional embeddings. \
        Fine-tuning allows adapting pre-trained models to specific tasks.";

    // Split the long text into smaller chunks
    let splitter = CharacterTextSplitter::new()
        .with_separator(". ")
        .with_chunk_size(120)
        .with_chunk_overlap(0);
    let chunks = splitter.split_text(long_text);
    assert!(chunks.len() > 1, "Expected multiple chunks from splitter");

    // Convert chunks into Documents
    let docs: Vec<Document> = chunks.iter().map(|c| Document::new(c.clone())).collect();

    let embeddings = fake_embeddings();
    let store = Arc::new(
        InMemoryVectorStore::from_documents(docs, embeddings)
            .await
            .unwrap(),
    );

    let retriever = make_retriever_with_k(store.clone() as Arc<dyn VectorStore>, 2);
    let llm = fake_llm(vec!["RAG combines search with LLMs."]);
    let chain = RetrievalQAChain::new(retriever, llm).with_k(2);
    let result = chain.call_with_sources("What is RAG?").await.unwrap();

    assert_eq!(result.answer, "RAG combines search with LLMs.");
    assert!(result.source_documents.len() <= 2);
}

// ---------------------------------------------------------------------------
// 4. Conversational RAG multi-turn
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_conversational_rag_multi_turn() {
    let docs = vec![
        Document::new("Rust was created by Graydon Hoare at Mozilla."),
        Document::new("Rust 1.0 was released on May 15, 2015."),
        Document::new("Rust has won Stack Overflow's most loved language award multiple times."),
    ];

    let embeddings = fake_embeddings();
    let store = Arc::new(
        InMemoryVectorStore::from_documents(docs, embeddings)
            .await
            .unwrap(),
    );
    let retriever = make_retriever_with_k(store.clone() as Arc<dyn VectorStore>, 2);

    // Responses: first QA answer, then condensation output, then second QA answer
    let llm = fake_llm(vec![
        "Rust was created by Graydon Hoare.",     // 1st call: QA answer
        "When was Rust 1.0 released?",            // 2nd call: condensation
        "Rust 1.0 was released on May 15, 2015.", // 3rd call: QA answer
    ]);

    let chain = ConversationalRetrievalChain::new(retriever, llm).with_k(2);

    // First turn -- no history, so condensation is skipped
    let r1 = chain.call_with_sources("Who created Rust?").await.unwrap();
    assert_eq!(r1.condensed_question, "Who created Rust?");
    assert_eq!(r1.answer, "Rust was created by Graydon Hoare.");

    // Second turn -- history exists, so question is condensed
    let r2 = chain
        .call_with_sources("When was version 1.0 released?")
        .await
        .unwrap();
    assert_eq!(r2.condensed_question, "When was Rust 1.0 released?");
    assert_eq!(r2.answer, "Rust 1.0 was released on May 15, 2015.");
    assert!(!r2.source_documents.is_empty());
}

// ---------------------------------------------------------------------------
// 5. RAG with metadata filtering
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_rag_with_metadata_filtering() {
    let mut meta_rust = HashMap::new();
    meta_rust.insert("language".to_string(), Value::String("rust".into()));
    meta_rust.insert("category".to_string(), Value::String("systems".into()));

    let mut meta_python = HashMap::new();
    meta_python.insert("language".to_string(), Value::String("python".into()));
    meta_python.insert("category".to_string(), Value::String("scripting".into()));

    let docs = vec![
        Document::new("Rust has zero-cost abstractions.").with_metadata(meta_rust.clone()),
        Document::new("Python is great for data science.").with_metadata(meta_python.clone()),
        Document::new("Rust guarantees memory safety.").with_metadata(meta_rust.clone()),
    ];

    let embeddings = fake_embeddings();
    let store = Arc::new(
        InMemoryVectorStore::from_documents(docs, embeddings)
            .await
            .unwrap(),
    );

    // Search for Rust-related content
    let results = store
        .similarity_search("Rust has zero-cost abstractions.", 2)
        .await
        .unwrap();
    assert_eq!(results.len(), 2);

    // Verify metadata is preserved through the pipeline
    for doc in &results {
        assert!(
            doc.metadata.contains_key("language"),
            "Expected metadata key 'language' on document: {}",
            doc.page_content
        );
        assert!(
            doc.metadata.contains_key("category"),
            "Expected metadata key 'category' on document: {}",
            doc.page_content
        );
    }

    // Use in a chain and verify source metadata is returned
    let retriever = make_retriever_with_k(store.clone() as Arc<dyn VectorStore>, 3);
    let llm = fake_llm(vec!["Rust has zero-cost abstractions and memory safety."]);
    let chain = RetrievalQAChain::new(retriever, llm).with_k(3);
    let result = chain
        .call_with_sources("What are Rust's features?")
        .await
        .unwrap();

    // All source documents should have metadata
    for doc in &result.source_documents {
        assert!(
            !doc.metadata.is_empty(),
            "Source document metadata should not be empty"
        );
    }
}

// ---------------------------------------------------------------------------
// 6. VectorStore add and delete
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_vectorstore_add_and_delete() {
    let embeddings = fake_embeddings();
    let store = InMemoryVectorStore::new(embeddings);

    // Add documents with explicit IDs
    let docs = vec![
        Document::new("Alpha document").with_id("alpha"),
        Document::new("Beta document").with_id("beta"),
        Document::new("Gamma document").with_id("gamma"),
    ];
    let ids = store.add_documents(docs, None).await.unwrap();
    assert_eq!(ids, vec!["alpha", "beta", "gamma"]);

    // Verify all three are searchable
    let all = store.similarity_search("document", 10).await.unwrap();
    assert_eq!(all.len(), 3);

    // Delete "beta"
    let deleted = store.delete(Some(&["beta".to_string()])).await.unwrap();
    assert!(deleted);

    // Verify only two remain
    let remaining = store.similarity_search("document", 10).await.unwrap();
    assert_eq!(remaining.len(), 2);
    assert!(
        remaining.iter().all(|d| d.page_content != "Beta document"),
        "Deleted document should not appear in search results"
    );

    // Verify get_by_ids no longer returns the deleted doc
    let fetched = store.get_by_ids(&["beta".to_string()]).await.unwrap();
    assert!(
        fetched.is_empty(),
        "Deleted document should not be retrievable by ID"
    );

    // The remaining docs should still be retrievable
    let fetched_alpha = store.get_by_ids(&["alpha".to_string()]).await.unwrap();
    assert_eq!(fetched_alpha.len(), 1);
    assert_eq!(fetched_alpha[0].page_content, "Alpha document");
}

// ---------------------------------------------------------------------------
// 7. RAG with empty vectorstore
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_rag_empty_vectorstore() {
    let embeddings = fake_embeddings();
    let store = Arc::new(InMemoryVectorStore::new(embeddings));

    // Verify empty search returns no results
    let results = store.similarity_search("anything", 5).await.unwrap();
    assert!(results.is_empty());

    // Chain should still work -- the LLM gets an empty context
    let retriever = make_retriever(store.clone() as Arc<dyn VectorStore>);
    let llm = fake_llm(vec!["I don't have enough context to answer."]);
    let chain = RetrievalQAChain::new(retriever, llm);

    let result = chain
        .call_with_sources("What is the meaning of life?")
        .await
        .unwrap();
    assert_eq!(result.answer, "I don't have enough context to answer.");
    assert!(result.source_documents.is_empty());

    // Conversational chain should also handle empty store gracefully
    let embeddings2 = fake_embeddings();
    let store2 = Arc::new(InMemoryVectorStore::new(embeddings2));
    let retriever2 = make_retriever(store2.clone() as Arc<dyn VectorStore>);
    let llm2 = fake_llm(vec!["No information available."]);
    let conv_chain = ConversationalRetrievalChain::new(retriever2, llm2);

    let conv_result = conv_chain.call("question?").await.unwrap();
    assert_eq!(conv_result, "No information available.");
}

// ---------------------------------------------------------------------------
// 8. RAG with custom prompt
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_rag_with_custom_prompt() {
    let docs = vec![
        Document::new("The capital of France is Paris."),
        Document::new("France is a country in Western Europe."),
    ];

    let embeddings = fake_embeddings();
    let store = Arc::new(
        InMemoryVectorStore::from_documents(docs, embeddings)
            .await
            .unwrap(),
    );

    let retriever = make_retriever_with_k(store.clone() as Arc<dyn VectorStore>, 2);

    // Use a parrot LLM so we can verify the prompt was formatted correctly
    let llm = parrot_llm();

    let custom_template =
        "You are a geography expert.\n\nRelevant facts:\n{context}\n\nUser question: {query}\n\nProvide a concise answer:";
    let chain = RetrievalQAChain::new(retriever, llm)
        .with_prompt_template(custom_template)
        .with_k(2);

    let answer = chain.call("What is the capital of France?").await.unwrap();

    // The parrot echoes the formatted prompt back, so we can verify template substitution
    assert!(
        answer.contains("You are a geography expert."),
        "Custom prompt prefix should appear in the formatted output"
    );
    assert!(
        answer.contains("User question: What is the capital of France?"),
        "Query should be substituted in the template"
    );
    assert!(
        answer.contains("The capital of France is Paris."),
        "Context from retrieved docs should appear in the prompt"
    );
    assert!(
        answer.contains("Relevant facts:"),
        "Custom template structure should be preserved"
    );
}
