//! Tests for indexing module: InMemoryDocumentIndex and InMemoryRecordManager.

use rustchain_core::documents::Document;
use rustchain_core::indexing::{
    DocumentIndex, InMemoryDocumentIndex, InMemoryRecordManager, RecordManager,
};

// ============================================================
// InMemoryDocumentIndex tests
// ============================================================

#[tokio::test]
async fn in_memory_doc_index_upsert_with_id() {
    let index = InMemoryDocumentIndex::new();
    let docs = vec![
        Document::new("Hello world").with_id("doc1"),
        Document::new("Goodbye world").with_id("doc2"),
    ];

    let result = index.upsert(docs).await.unwrap();
    assert_eq!(result.succeeded, vec!["doc1", "doc2"]);
    assert!(result.failed.is_empty());
    assert_eq!(index.len(), 2);
}

#[tokio::test]
async fn in_memory_doc_index_upsert_without_id() {
    let index = InMemoryDocumentIndex::new();
    let docs = vec![Document::new("No ID doc")];

    let result = index.upsert(docs).await.unwrap();
    assert_eq!(result.succeeded.len(), 1);
    assert!(!result.succeeded[0].is_empty());
    assert_eq!(index.len(), 1);
}

#[tokio::test]
async fn in_memory_doc_index_upsert_overwrites() {
    let index = InMemoryDocumentIndex::new();
    let docs1 = vec![Document::new("version 1").with_id("doc1")];
    index.upsert(docs1).await.unwrap();

    let docs2 = vec![Document::new("version 2").with_id("doc1")];
    index.upsert(docs2).await.unwrap();

    assert_eq!(index.len(), 1);
    let retrieved = index.get(&["doc1".into()]).await.unwrap();
    assert_eq!(retrieved[0].page_content, "version 2");
}

#[tokio::test]
async fn in_memory_doc_index_get() {
    let index = InMemoryDocumentIndex::new();
    let docs = vec![
        Document::new("Doc A").with_id("a"),
        Document::new("Doc B").with_id("b"),
        Document::new("Doc C").with_id("c"),
    ];
    index.upsert(docs).await.unwrap();

    let result = index.get(&["a".into(), "c".into()]).await.unwrap();
    assert_eq!(result.len(), 2);

    let contents: Vec<&str> = result.iter().map(|d| d.page_content.as_str()).collect();
    assert!(contents.contains(&"Doc A"));
    assert!(contents.contains(&"Doc C"));
}

#[tokio::test]
async fn in_memory_doc_index_get_missing_key() {
    let index = InMemoryDocumentIndex::new();
    let result = index.get(&["nonexistent".into()]).await.unwrap();
    assert!(result.is_empty());
}

#[tokio::test]
async fn in_memory_doc_index_delete() {
    let index = InMemoryDocumentIndex::new();
    let docs = vec![
        Document::new("Doc A").with_id("a"),
        Document::new("Doc B").with_id("b"),
    ];
    index.upsert(docs).await.unwrap();

    let del_result = index.delete(&["a".into()]).await.unwrap();
    assert_eq!(del_result.num_deleted, Some(1));
    assert_eq!(del_result.succeeded, Some(vec!["a".into()]));
    assert_eq!(index.len(), 1);

    let remaining = index.get(&["b".into()]).await.unwrap();
    assert_eq!(remaining.len(), 1);
}

#[tokio::test]
async fn in_memory_doc_index_delete_missing_key() {
    let index = InMemoryDocumentIndex::new();
    let del_result = index.delete(&["nonexistent".into()]).await.unwrap();
    assert_eq!(del_result.num_deleted, Some(0));
    assert_eq!(del_result.succeeded, Some(vec![]));
}

#[tokio::test]
async fn in_memory_doc_index_is_empty() {
    let index = InMemoryDocumentIndex::new();
    assert!(index.is_empty());

    index
        .upsert(vec![Document::new("doc").with_id("1")])
        .await
        .unwrap();
    assert!(!index.is_empty());
}

#[tokio::test]
async fn in_memory_doc_index_search() {
    let index = InMemoryDocumentIndex::new().with_top_k(2);
    let docs = vec![
        Document::new("rust rust rust").with_id("a"),
        Document::new("rust").with_id("b"),
        Document::new("python").with_id("c"),
        Document::new("rust rust").with_id("d"),
    ];
    index.upsert(docs).await.unwrap();

    let results = index.search("rust");
    assert_eq!(results.len(), 2);
    // Most relevant first
    assert_eq!(results[0].page_content, "rust rust rust");
    assert_eq!(results[1].page_content, "rust rust");
}

#[tokio::test]
async fn in_memory_doc_index_search_no_match() {
    let index = InMemoryDocumentIndex::new();
    let docs = vec![Document::new("hello world").with_id("a")];
    index.upsert(docs).await.unwrap();

    let results = index.search("nonexistent");
    // Still returns docs, just with 0 count
    assert_eq!(results.len(), 1);
}

#[tokio::test]
async fn in_memory_doc_index_default() {
    let index = InMemoryDocumentIndex::default();
    assert!(index.is_empty());
}

// ============================================================
// InMemoryRecordManager additional tests
// ============================================================

#[tokio::test]
async fn record_manager_update_and_list_with_time_filters() {
    let rm = InMemoryRecordManager::new("test");

    // Insert with a far-future time_at_least so it gets that timestamp
    let far_future = 99999999990.0;
    rm.update(
        &["future_key".into()],
        &[Some("group1".into())],
        Some(far_future),
    )
    .await
    .unwrap();

    // Insert with current time (no time_at_least)
    rm.update(
        &["now_key".into()],
        &[Some("group1".into())],
        None,
    )
    .await
    .unwrap();

    // List keys before far_future: now_key should appear (its timestamp is current time)
    let keys = rm
        .list_keys(Some(far_future), None, None, None)
        .await
        .unwrap();
    assert!(keys.contains(&"now_key".to_string()));
    assert!(!keys.contains(&"future_key".to_string()));

    // List keys after current time - 1: both should appear
    let current = rm.get_time().await.unwrap();
    let keys = rm
        .list_keys(None, Some(current - 1.0), None, None)
        .await
        .unwrap();
    assert!(keys.contains(&"now_key".to_string()) || keys.contains(&"future_key".to_string()));
}

#[tokio::test]
async fn record_manager_delete_nonexistent_key() {
    let rm = InMemoryRecordManager::new("test");
    // Should not error
    rm.delete_keys(&["nonexistent".into()]).await.unwrap();
}

#[tokio::test]
async fn record_manager_update_overwrites() {
    let rm = InMemoryRecordManager::new("test");
    rm.update(
        &["key1".into()],
        &[Some("group_a".into())],
        None,
    )
    .await
    .unwrap();

    // Overwrite with different group
    rm.update(
        &["key1".into()],
        &[Some("group_b".into())],
        None,
    )
    .await
    .unwrap();

    // Should only appear in group_b
    let keys_a = rm
        .list_keys(None, None, Some(&["group_a".into()]), None)
        .await
        .unwrap();
    assert!(keys_a.is_empty());

    let keys_b = rm
        .list_keys(None, None, Some(&["group_b".into()]), None)
        .await
        .unwrap();
    assert_eq!(keys_b, vec!["key1".to_string()]);
}

#[tokio::test]
async fn record_manager_list_keys_no_group_id_excluded() {
    let rm = InMemoryRecordManager::new("test");
    rm.update(&["key1".into()], &[None], None).await.unwrap();

    // Filtering by group should exclude keys with no group
    let keys = rm
        .list_keys(None, None, Some(&["some_group".into()]), None)
        .await
        .unwrap();
    assert!(keys.is_empty());
}

#[tokio::test]
async fn record_manager_list_keys_sorted() {
    let rm = InMemoryRecordManager::new("test");
    rm.update(
        &["c".into(), "a".into(), "b".into()],
        &[None, None, None],
        None,
    )
    .await
    .unwrap();

    let keys = rm.list_keys(None, None, None, None).await.unwrap();
    assert_eq!(keys, vec!["a", "b", "c"]);
}
