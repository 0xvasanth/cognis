//! End-to-end RAG round-trip: embed documents, store, search by query,
//! verify sensible ranking. Uses FakeEmbeddings — no network deps.

use std::sync::Arc;

use cognis2_rag::{Distance, Embeddings, FakeEmbeddings, InMemoryVectorStore, VectorStore};

#[tokio::test]
async fn rag_round_trip_with_fake_embeddings() {
    let embedder: Arc<dyn Embeddings> = Arc::new(FakeEmbeddings::new(32));
    let mut store = InMemoryVectorStore::new(embedder.clone());

    let docs = vec![
        "the quick brown fox jumps over the lazy dog".to_string(),
        "rust is a systems programming language".to_string(),
        "embeddings encode text as vectors".to_string(),
        "vector stores enable similarity search".to_string(),
    ];
    let ids = store.add_texts(docs.clone(), None).await.unwrap();
    assert_eq!(ids.len(), 4);
    assert_eq!(store.len(), 4);

    // Query: an exact match should rank itself top.
    let r = store.similarity_search(&docs[2], 4).await.unwrap();
    assert_eq!(r[0].text, docs[2]);
    assert!(r[0].score > 0.99, "exact match score should be ~1.0, got {}", r[0].score);

    // k bounds.
    let r2 = store.similarity_search(&docs[0], 2).await.unwrap();
    assert_eq!(r2.len(), 2);
}

#[tokio::test]
async fn distance_variants_all_work() {
    let embedder: Arc<dyn Embeddings> = Arc::new(FakeEmbeddings::new(16));
    for distance in [Distance::Cosine, Distance::Euclidean, Distance::Dot] {
        let mut store = InMemoryVectorStore::with_distance(embedder.clone(), distance);
        store
            .add_texts(vec!["x".into(), "y".into(), "z".into()], None)
            .await
            .unwrap();
        let r = store.similarity_search("x", 3).await.unwrap();
        assert_eq!(r.len(), 3);
        // Top result is "x" (exact match) regardless of distance variant.
        assert_eq!(r[0].text, "x", "top match for exact query failed under {:?}", distance);
    }
}

#[tokio::test]
async fn add_vectors_skips_embedder() {
    let embedder: Arc<dyn Embeddings> = Arc::new(FakeEmbeddings::new(8));
    let mut store = InMemoryVectorStore::new(embedder);

    // Provide vectors directly — store doesn't call embed_documents.
    let vectors = vec![vec![1.0; 8], vec![0.5; 8]];
    let texts = vec!["one".to_string(), "two".to_string()];
    let ids = store.add_vectors(vectors, texts, None).await.unwrap();
    assert_eq!(ids.len(), 2);

    // Search by precomputed query vector.
    let r = store.similarity_search_by_vector(vec![1.0; 8], 1).await.unwrap();
    assert_eq!(r.len(), 1);
    // The stored vector [1.0; 8] is closer to the query than [0.5; 8].
    assert_eq!(r[0].text, "one");
}

#[tokio::test]
async fn metadata_filtering_by_score() {
    let embedder: Arc<dyn Embeddings> = Arc::new(FakeEmbeddings::new(8));
    let mut store = InMemoryVectorStore::new(embedder);

    let mut md_a = std::collections::HashMap::new();
    md_a.insert("category".into(), serde_json::json!("animal"));
    let mut md_b = std::collections::HashMap::new();
    md_b.insert("category".into(), serde_json::json!("plant"));

    store
        .add_texts(
            vec!["dog".into(), "tree".into()],
            Some(vec![md_a.clone(), md_b.clone()]),
        )
        .await
        .unwrap();

    let r = store.similarity_search("dog", 2).await.unwrap();
    assert_eq!(r.len(), 2);
    let dog = r.iter().find(|x| x.text == "dog").unwrap();
    assert_eq!(dog.metadata.get("category").unwrap(), "animal");
}
