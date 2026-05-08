use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use cognis_core::documents::Document;
use cognis_core::embeddings::Embeddings;
use cognis_core::error::Result;
use cognis_core::retrievers::BaseRetriever;
use cognis_core::vectorstores::*;

/// A mock embedding model that produces deterministic embeddings based on text length.
/// Each text is mapped to a 3-dimensional vector derived from its character count.
struct MockEmbeddings;

#[async_trait]
impl Embeddings for MockEmbeddings {
    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let len = t.len() as f32;
                vec![len, len * 0.5, len * 0.1]
            })
            .collect())
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let len = text.len() as f32;
        Ok(vec![len, len * 0.5, len * 0.1])
    }
}

/// A mock embedding model that produces distinct directional vectors.
/// Used to test MMR diversity properly since texts get orthogonal-ish embeddings.
struct DirectionalEmbeddings;

#[async_trait]
impl Embeddings for DirectionalEmbeddings {
    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| text_to_directional_vec(t)).collect())
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        Ok(text_to_directional_vec(text))
    }
}

/// Map text to a directional vector based on its first character.
/// This ensures different texts get meaningfully different embeddings.
fn text_to_directional_vec(text: &str) -> Vec<f32> {
    match text.chars().next() {
        Some('a') => vec![1.0, 0.0, 0.0],
        Some('b') => vec![0.0, 1.0, 0.0],
        Some('c') => vec![0.0, 0.0, 1.0],
        Some('d') => vec![0.7, 0.7, 0.0],
        Some('e') => vec![0.7, 0.0, 0.7],
        _ => {
            let len = text.len() as f32;
            vec![len, len * 0.5, len * 0.1]
        }
    }
}

// =============================================================================
// add_documents tests
// =============================================================================

#[tokio::test]
async fn test_add_documents_returns_ids() {
    let store = InMemoryVectorStore::new(Arc::new(MockEmbeddings));
    let docs = vec![
        Document::new("hello world"),
        Document::new("goodbye world"),
        Document::new("hello rust"),
    ];
    let ids = store.add_documents(docs, None).await.unwrap();
    assert_eq!(ids.len(), 3);
    // IDs should be unique UUIDs
    assert_ne!(ids[0], ids[1]);
    assert_ne!(ids[1], ids[2]);
}

#[tokio::test]
async fn test_add_documents_with_explicit_ids() {
    let store = InMemoryVectorStore::new(Arc::new(MockEmbeddings));
    let docs = vec![Document::new("doc one"), Document::new("doc two")];
    let explicit_ids = vec!["id-1".to_string(), "id-2".to_string()];
    let ids = store
        .add_documents(docs, Some(explicit_ids.clone()))
        .await
        .unwrap();
    assert_eq!(ids, explicit_ids);
}

#[tokio::test]
async fn test_add_documents_uses_doc_id_if_present() {
    let store = InMemoryVectorStore::new(Arc::new(MockEmbeddings));
    let doc = Document::new("content").with_id("my-id");
    let ids = store.add_documents(vec![doc], None).await.unwrap();
    assert_eq!(ids, vec!["my-id".to_string()]);
}

#[tokio::test]
async fn test_add_documents_explicit_ids_override_doc_ids() {
    let store = InMemoryVectorStore::new(Arc::new(MockEmbeddings));
    let doc = Document::new("content").with_id("doc-id");
    let ids = store
        .add_documents(vec![doc], Some(vec!["explicit-id".to_string()]))
        .await
        .unwrap();
    assert_eq!(ids, vec!["explicit-id".to_string()]);
}

// =============================================================================
// add_texts tests
// =============================================================================

#[tokio::test]
async fn test_add_texts_basic() {
    let store = InMemoryVectorStore::new(Arc::new(MockEmbeddings));
    let texts = vec!["hello".to_string(), "world".to_string()];
    let ids = store.add_texts(&texts, None, None).await.unwrap();
    assert_eq!(ids.len(), 2);

    let results = store.similarity_search("hello", 10).await.unwrap();
    assert_eq!(results.len(), 2);
}

#[tokio::test]
async fn test_add_texts_with_metadata() {
    let store = InMemoryVectorStore::new(Arc::new(MockEmbeddings));
    let texts = vec!["hello".to_string(), "world".to_string()];
    let mut meta1 = HashMap::new();
    meta1.insert("source".to_string(), Value::String("test".to_string()));
    let mut meta2 = HashMap::new();
    meta2.insert("source".to_string(), Value::String("test2".to_string()));
    let metadatas = vec![meta1, meta2];
    let ids = store
        .add_texts(&texts, Some(&metadatas), None)
        .await
        .unwrap();
    assert_eq!(ids.len(), 2);

    let found = store.get_by_ids(&ids).await.unwrap();
    assert_eq!(found.len(), 2);
    assert!(found
        .iter()
        .any(|d| d.metadata.get("source") == Some(&Value::String("test".to_string()))));
}

#[tokio::test]
async fn test_add_texts_with_ids() {
    let store = InMemoryVectorStore::new(Arc::new(MockEmbeddings));
    let texts = vec!["alpha".to_string(), "beta".to_string()];
    let ids_input = vec!["a1".to_string(), "b2".to_string()];
    let ids = store
        .add_texts(&texts, None, Some(&ids_input))
        .await
        .unwrap();
    assert_eq!(ids, ids_input);
}

// =============================================================================
// delete tests
// =============================================================================

#[tokio::test]
async fn test_delete_removes_documents() {
    let store = InMemoryVectorStore::new(Arc::new(MockEmbeddings));
    let docs = vec![Document::new("test doc")];
    let ids = store.add_documents(docs, None).await.unwrap();
    store.delete(Some(&ids)).await.unwrap();
    let results = store.similarity_search("test", 10).await.unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_delete_with_none_is_noop() {
    let store = InMemoryVectorStore::new(Arc::new(MockEmbeddings));
    let docs = vec![Document::new("test doc")];
    store.add_documents(docs, None).await.unwrap();
    let result = store.delete(None).await.unwrap();
    assert!(result);
    let results = store.similarity_search("test", 10).await.unwrap();
    assert_eq!(results.len(), 1);
}

#[tokio::test]
async fn test_delete_nonexistent_id_succeeds() {
    let store = InMemoryVectorStore::new(Arc::new(MockEmbeddings));
    let result = store
        .delete(Some(&["nonexistent".to_string()]))
        .await
        .unwrap();
    assert!(result);
}

#[tokio::test]
async fn test_delete_subset() {
    let store = InMemoryVectorStore::new(Arc::new(MockEmbeddings));
    let docs = vec![
        Document::new("keep me"),
        Document::new("delete me"),
        Document::new("keep me too"),
    ];
    let ids = store.add_documents(docs, None).await.unwrap();
    store.delete(Some(&[ids[1].clone()])).await.unwrap();
    let remaining = store.similarity_search("keep", 10).await.unwrap();
    assert_eq!(remaining.len(), 2);
}

// =============================================================================
// get_by_ids tests
// =============================================================================

#[tokio::test]
async fn test_get_by_ids_found() {
    let store = InMemoryVectorStore::new(Arc::new(MockEmbeddings));
    let docs = vec![Document::new("findme")];
    let ids = store.add_documents(docs, None).await.unwrap();
    let found = store.get_by_ids(&ids).await.unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].page_content, "findme");
    assert_eq!(found[0].id, Some(ids[0].clone()));
}

#[tokio::test]
async fn test_get_by_ids_missing() {
    let store = InMemoryVectorStore::new(Arc::new(MockEmbeddings));
    let found = store.get_by_ids(&["no-such-id".to_string()]).await.unwrap();
    assert!(found.is_empty());
}

#[tokio::test]
async fn test_get_by_ids_mixed() {
    let store = InMemoryVectorStore::new(Arc::new(MockEmbeddings));
    let docs = vec![Document::new("exists")];
    let ids = store.add_documents(docs, None).await.unwrap();
    let lookup = vec![ids[0].clone(), "nonexistent".to_string()];
    let found = store.get_by_ids(&lookup).await.unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].page_content, "exists");
}

// =============================================================================
// similarity_search tests
// =============================================================================

#[tokio::test]
async fn test_similarity_search_basic() {
    let store = InMemoryVectorStore::new(Arc::new(MockEmbeddings));
    let docs = vec![
        Document::new("hello world"),
        Document::new("goodbye world"),
        Document::new("hello rust"),
    ];
    store.add_documents(docs, None).await.unwrap();
    let results = store.similarity_search("hello", 2).await.unwrap();
    assert_eq!(results.len(), 2);
}

#[tokio::test]
async fn test_similarity_search_empty_store() {
    let store = InMemoryVectorStore::new(Arc::new(MockEmbeddings));
    let results = store.similarity_search("anything", 5).await.unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_similarity_search_k_larger_than_store() {
    let store = InMemoryVectorStore::new(Arc::new(MockEmbeddings));
    let docs = vec![Document::new("only one")];
    store.add_documents(docs, None).await.unwrap();
    let results = store.similarity_search("query", 10).await.unwrap();
    assert_eq!(results.len(), 1);
}

#[tokio::test]
async fn test_similarity_search_returns_exact_match_first() {
    let store = InMemoryVectorStore::new(Arc::new(MockEmbeddings));
    // "hello" (5 chars) should be most similar to a query of same length
    let docs = vec![
        Document::new("a very long document with many characters"),
        Document::new("short"), // 5 chars - same as "query" len
        Document::new("medium length text"),
    ];
    store.add_documents(docs, None).await.unwrap();
    // "query" is 5 chars, same as "short" - should match most closely
    let results = store.similarity_search("query", 1).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].page_content, "short");
}

// =============================================================================
// similarity_search_with_score tests
// =============================================================================

#[tokio::test]
async fn test_similarity_search_with_score_ordered() {
    let store = InMemoryVectorStore::new(Arc::new(MockEmbeddings));
    let docs = vec![
        Document::new("short"),
        Document::new("a longer document here"),
    ];
    store.add_documents(docs, None).await.unwrap();
    let results = store
        .similarity_search_with_score("short", 2)
        .await
        .unwrap();
    assert_eq!(results.len(), 2);
    // First result should have higher score (more similar)
    assert!(results[0].1 >= results[1].1);
}

#[tokio::test]
async fn test_similarity_search_with_score_identical_query() {
    let store = InMemoryVectorStore::new(Arc::new(MockEmbeddings));
    let docs = vec![Document::new("exact")];
    store.add_documents(docs, None).await.unwrap();
    let results = store
        .similarity_search_with_score("exact", 1)
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    // Identical text => identical embedding => cosine similarity = 1.0
    assert!((results[0].1 - 1.0).abs() < 1e-6);
}

#[tokio::test]
async fn test_similarity_search_with_score_empty_store() {
    let store = InMemoryVectorStore::new(Arc::new(MockEmbeddings));
    let results = store
        .similarity_search_with_score("query", 5)
        .await
        .unwrap();
    assert!(results.is_empty());
}

// =============================================================================
// similarity_search_by_vector tests
// =============================================================================

#[tokio::test]
async fn test_similarity_search_by_vector_basic() {
    let store = InMemoryVectorStore::new(Arc::new(DirectionalEmbeddings));
    // "alpha" => [1, 0, 0], "beta" => [0, 1, 0]
    let docs = vec![Document::new("alpha doc"), Document::new("beta doc")];
    store.add_documents(docs, None).await.unwrap();
    // Query vector aligned with alpha direction
    let query_vec = vec![1.0_f32, 0.0, 0.0];
    let results = store
        .similarity_search_by_vector(&query_vec, 1)
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].page_content, "alpha doc");
}

#[tokio::test]
async fn test_similarity_search_by_vector_empty_store() {
    let store = InMemoryVectorStore::new(Arc::new(MockEmbeddings));
    let results = store
        .similarity_search_by_vector(&[1.0, 0.0, 0.0], 5)
        .await
        .unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_similarity_search_by_vector_returns_k_results() {
    let store = InMemoryVectorStore::new(Arc::new(MockEmbeddings));
    let docs = vec![
        Document::new("a"),
        Document::new("bb"),
        Document::new("ccc"),
        Document::new("dddd"),
    ];
    store.add_documents(docs, None).await.unwrap();
    let results = store
        .similarity_search_by_vector(&[2.0, 1.0, 0.2], 2)
        .await
        .unwrap();
    assert_eq!(results.len(), 2);
}

// =============================================================================
// max_marginal_relevance_search tests
// =============================================================================

#[tokio::test]
async fn test_mmr_search_basic() {
    let store = InMemoryVectorStore::new(Arc::new(DirectionalEmbeddings));
    let docs = vec![
        Document::new("alpha one"),   // [1.0, 0.0, 0.0]
        Document::new("alpha two"),   // [1.0, 0.0, 0.0]
        Document::new("beta one"),    // [0.0, 1.0, 0.0]
        Document::new("charlie one"), // [0.0, 0.0, 1.0]
    ];
    store.add_documents(docs, None).await.unwrap();
    let results = store
        .max_marginal_relevance_search("alpha query", 3, 4, 0.5)
        .await
        .unwrap();
    assert_eq!(results.len(), 3);
}

#[tokio::test]
async fn test_mmr_search_diversity() {
    let store = InMemoryVectorStore::new(Arc::new(DirectionalEmbeddings));
    // Two docs with identical embeddings to query, one orthogonal
    let docs = vec![
        Document::new("alpha one"), // [1.0, 0.0, 0.0]
        Document::new("alpha two"), // [1.0, 0.0, 0.0] - same direction
        Document::new("beta one"),  // [0.0, 1.0, 0.0] - orthogonal
    ];
    store.add_documents(docs, None).await.unwrap();
    // With low lambda (diversity-focused), second result should differ from first
    let results = store
        .max_marginal_relevance_search("alpha query", 2, 3, 0.0)
        .await
        .unwrap();
    assert_eq!(results.len(), 2);
    // First should be one of the "alpha" docs (most similar to query)
    assert!(results[0].page_content.starts_with("alpha"));
    // Second should be "beta" (most diverse from the first selection)
    assert_eq!(results[1].page_content, "beta one");
}

#[tokio::test]
async fn test_mmr_search_pure_relevance() {
    let store = InMemoryVectorStore::new(Arc::new(DirectionalEmbeddings));
    let docs = vec![
        Document::new("alpha one"), // [1.0, 0.0, 0.0]
        Document::new("alpha two"), // [1.0, 0.0, 0.0]
        Document::new("beta one"),  // [0.0, 1.0, 0.0]
    ];
    store.add_documents(docs, None).await.unwrap();
    // With lambda=1.0 (pure relevance), both "alpha" docs should come first
    let results = store
        .max_marginal_relevance_search("alpha query", 2, 3, 1.0)
        .await
        .unwrap();
    assert_eq!(results.len(), 2);
    assert!(results[0].page_content.starts_with("alpha"));
    assert!(results[1].page_content.starts_with("alpha"));
}

#[tokio::test]
async fn test_mmr_search_empty_store() {
    let store = InMemoryVectorStore::new(Arc::new(MockEmbeddings));
    let results = store
        .max_marginal_relevance_search("query", 3, 10, 0.5)
        .await
        .unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_mmr_search_k_greater_than_store() {
    let store = InMemoryVectorStore::new(Arc::new(MockEmbeddings));
    let docs = vec![Document::new("only one")];
    store.add_documents(docs, None).await.unwrap();
    let results = store
        .max_marginal_relevance_search("query", 5, 10, 0.5)
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
}

#[tokio::test]
async fn test_mmr_search_fetch_k_limits_candidates() {
    let store = InMemoryVectorStore::new(Arc::new(MockEmbeddings));
    let docs = vec![
        Document::new("a"),
        Document::new("bb"),
        Document::new("ccc"),
        Document::new("dddd"),
        Document::new("eeeee"),
    ];
    store.add_documents(docs, None).await.unwrap();
    // fetch_k=2 means only 2 candidates, so at most 2 results even if k=5
    let results = store
        .max_marginal_relevance_search("query", 5, 2, 0.5)
        .await
        .unwrap();
    assert_eq!(results.len(), 2);
}

// =============================================================================
// from_texts tests
// =============================================================================

#[tokio::test]
async fn test_from_texts_basic() {
    let store = InMemoryVectorStore::from_texts(
        vec!["hello".into(), "world".into()],
        Arc::new(MockEmbeddings),
        None,
    )
    .await
    .unwrap();
    let results = store.similarity_search("hello", 1).await.unwrap();
    assert_eq!(results.len(), 1);
}

#[tokio::test]
async fn test_from_texts_with_metadata() {
    let mut meta = HashMap::new();
    meta.insert("key".to_string(), Value::String("value".to_string()));
    let store = InMemoryVectorStore::from_texts(
        vec!["text1".into()],
        Arc::new(MockEmbeddings),
        Some(vec![meta.clone()]),
    )
    .await
    .unwrap();
    let results = store.similarity_search("text1", 1).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].metadata.get("key"),
        Some(&Value::String("value".to_string()))
    );
}

// =============================================================================
// VectorStoreRetriever tests
// =============================================================================

#[tokio::test]
async fn test_retriever_similarity_search() {
    let store = Arc::new(InMemoryVectorStore::new(Arc::new(MockEmbeddings)));
    let docs = vec![Document::new("hello world"), Document::new("goodbye world")];
    store.add_documents(docs, None).await.unwrap();
    let retriever = store.as_retriever();
    let results = retriever.get_relevant_documents("hello").await.unwrap();
    // Default k=4, but only 2 docs exist
    assert_eq!(results.len(), 2);
}

#[tokio::test]
async fn test_retriever_with_custom_k() {
    let store = Arc::new(InMemoryVectorStore::new(Arc::new(MockEmbeddings)));
    let docs = vec![
        Document::new("one"),
        Document::new("two"),
        Document::new("three"),
    ];
    store.add_documents(docs, None).await.unwrap();
    let retriever = store.as_retriever_with(SearchType::Similarity, 1);
    let results = retriever.get_relevant_documents("one").await.unwrap();
    assert_eq!(results.len(), 1);
}

#[tokio::test]
async fn test_retriever_mmr_mode() {
    let store = Arc::new(InMemoryVectorStore::new(Arc::new(DirectionalEmbeddings)));
    let docs = vec![
        Document::new("alpha one"),
        Document::new("alpha two"),
        Document::new("beta one"),
    ];
    store.add_documents(docs, None).await.unwrap();
    let retriever = store.as_retriever_with(
        SearchType::Mmr {
            fetch_k: 10,
            lambda_mult: 0.5,
        },
        2,
    );
    let results = retriever.get_relevant_documents("alpha").await.unwrap();
    assert_eq!(results.len(), 2);
}

#[tokio::test]
async fn test_retriever_score_threshold_mode() {
    let store = Arc::new(InMemoryVectorStore::new(Arc::new(DirectionalEmbeddings)));
    // "alpha" => [1,0,0], "beta" => [0,1,0] - orthogonal, cosine sim = 0
    let docs = vec![Document::new("alpha match"), Document::new("beta nomatch")];
    store.add_documents(docs, None).await.unwrap();
    let retriever = store.as_retriever_with(
        SearchType::SimilarityScoreThreshold {
            score_threshold: 0.5,
        },
        10,
    );
    // "alpha query" => [1,0,0], only "alpha match" has sim=1.0, "beta" has sim=0.0
    let results = retriever
        .get_relevant_documents("alpha query")
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].page_content, "alpha match");
}

// =============================================================================
// SearchType tests
// =============================================================================

#[test]
fn test_search_type_default() {
    let st = SearchType::default();
    match st {
        SearchType::Similarity => {}
        _ => panic!("Expected Similarity"),
    }
}

// =============================================================================
// Utility function tests
// =============================================================================

#[test]
fn test_cosine_similarity_identical() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let score = cosine_similarity(&a, &b);
    assert!((score - 1.0).abs() < 1e-6);
}

#[test]
fn test_cosine_similarity_orthogonal() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    let score = cosine_similarity(&a, &b);
    assert!(score.abs() < 1e-6);
}

#[test]
fn test_cosine_similarity_zero_vector() {
    let a = vec![1.0, 0.0];
    let zero = vec![0.0, 0.0];
    assert_eq!(cosine_similarity(&a, &zero), 0.0);
}

#[test]
fn test_cosine_relevance_score_values() {
    assert!((cosine_relevance_score(0.0) - 1.0).abs() < 1e-6);
    assert!((cosine_relevance_score(1.0) - 0.0).abs() < 1e-6);
}

#[test]
fn test_euclidean_relevance_score_values() {
    assert!((euclidean_relevance_score(0.0) - 1.0).abs() < 1e-6);
}

// =============================================================================
// Document metadata preservation tests
// =============================================================================

#[tokio::test]
async fn test_metadata_preserved_through_search() {
    let store = InMemoryVectorStore::new(Arc::new(MockEmbeddings));
    let mut meta = HashMap::new();
    meta.insert("author".to_string(), Value::String("alice".to_string()));
    meta.insert(
        "page".to_string(),
        Value::Number(serde_json::Number::from(42)),
    );
    let doc = Document::new("test content").with_metadata(meta.clone());
    store.add_documents(vec![doc], None).await.unwrap();

    // similarity_search
    let results = store.similarity_search("test content", 1).await.unwrap();
    assert_eq!(
        results[0].metadata.get("author"),
        Some(&Value::String("alice".to_string()))
    );
    assert_eq!(
        results[0].metadata.get("page"),
        Some(&Value::Number(serde_json::Number::from(42)))
    );

    // similarity_search_with_score
    let results = store
        .similarity_search_with_score("test content", 1)
        .await
        .unwrap();
    assert_eq!(
        results[0].0.metadata.get("author"),
        Some(&Value::String("alice".to_string()))
    );
}

#[tokio::test]
async fn test_id_preserved_through_search() {
    let store = InMemoryVectorStore::new(Arc::new(MockEmbeddings));
    let doc = Document::new("content").with_id("my-custom-id");
    store.add_documents(vec![doc], None).await.unwrap();

    let results = store.similarity_search("content", 1).await.unwrap();
    assert_eq!(results[0].id, Some("my-custom-id".to_string()));
}

// =============================================================================
// Upsert (overwrite) behavior tests
// =============================================================================

#[tokio::test]
async fn test_add_documents_overwrites_existing_id() {
    let store = InMemoryVectorStore::new(Arc::new(MockEmbeddings));
    let doc1 = Document::new("original");
    let ids = store
        .add_documents(vec![doc1], Some(vec!["same-id".to_string()]))
        .await
        .unwrap();
    assert_eq!(ids, vec!["same-id"]);

    let doc2 = Document::new("updated");
    store
        .add_documents(vec![doc2], Some(vec!["same-id".to_string()]))
        .await
        .unwrap();

    let found = store.get_by_ids(&["same-id".to_string()]).await.unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].page_content, "updated");
}
