use async_trait::async_trait;
use rustchain_core::documents::{
    BaseDocumentCompressor, BaseDocumentTransformer, Blob, BlobData, Document,
};
use rustchain_core::error::Result;
use serde_json::json;
use std::collections::HashMap;

// ============================================================
// Document struct field tests
// ============================================================

#[test]
fn document_has_doc_type_field() {
    let doc = Document::new("content").with_doc_type("ChatDocument");
    assert_eq!(doc.doc_type, Some("ChatDocument".into()));
}

#[test]
fn document_doc_type_default_is_none() {
    let doc = Document::new("content");
    assert!(doc.doc_type.is_none());
}

#[test]
fn document_doc_type_serde_roundtrip() {
    let doc = Document::new("test")
        .with_id("id1")
        .with_doc_type("Document");
    let json_str = serde_json::to_string(&doc).unwrap();
    let back: Document = serde_json::from_str(&json_str).unwrap();
    assert_eq!(back.doc_type, Some("Document".into()));
    assert_eq!(doc, back);
}

#[test]
fn document_doc_type_skipped_when_none() {
    let doc = Document::new("test");
    let json_str = serde_json::to_string(&doc).unwrap();
    assert!(!json_str.contains("doc_type"));
}

#[test]
fn document_new() {
    let doc = Document::new("Hello world");
    assert_eq!(doc.page_content, "Hello world");
    assert!(doc.id.is_none());
    assert!(doc.metadata.is_empty());
}

#[test]
fn document_with_id_and_metadata() {
    let mut meta = HashMap::new();
    meta.insert("source".into(), json!("https://example.com"));
    let doc = Document::new("content")
        .with_id("doc_1")
        .with_metadata(meta);
    assert_eq!(doc.id, Some("doc_1".into()));
    assert_eq!(doc.metadata["source"], json!("https://example.com"));
}

#[test]
fn document_display_no_metadata() {
    let doc = Document::new("Hello");
    assert_eq!(format!("{}", doc), "page_content='Hello'");
}

#[test]
fn document_display_with_metadata() {
    let mut meta = HashMap::new();
    meta.insert("key".into(), json!("val"));
    let doc = Document::new("Hi").with_metadata(meta);
    let s = format!("{}", doc);
    assert!(s.starts_with("page_content='Hi' metadata="));
}

#[test]
fn document_serde_roundtrip() {
    let doc = Document::new("test").with_id("id1");
    let json = serde_json::to_string(&doc).unwrap();
    let back: Document = serde_json::from_str(&json).unwrap();
    assert_eq!(doc, back);
}

#[test]
fn blob_from_string() {
    let blob = Blob::from_string("hello");
    assert_eq!(blob.as_string().unwrap(), "hello");
    assert_eq!(blob.as_bytes().unwrap(), b"hello");
}

#[test]
fn blob_from_bytes() {
    let blob = Blob::from_bytes(vec![1, 2, 3]);
    assert_eq!(blob.as_bytes().unwrap(), vec![1, 2, 3]);
}

#[test]
fn blob_from_path() {
    let blob = Blob::from_path("/tmp/test.txt");
    assert_eq!(blob.source(), Some("/tmp/test.txt".into()));
    assert!(blob.data.is_none());
}

#[test]
fn blob_with_mimetype() {
    let blob = Blob::from_string("data").with_mimetype("text/plain");
    assert_eq!(blob.mimetype, Some("text/plain".into()));
}

#[test]
fn blob_source_from_metadata() {
    let mut blob = Blob::from_string("data");
    blob.metadata
        .insert("source".into(), json!("https://example.com"));
    assert_eq!(blob.source(), Some("https://example.com".into()));
}

#[test]
fn blob_no_data_no_path_errors() {
    let blob = Blob {
        data: None,
        path: None,
        mimetype: None,
        encoding: "utf-8".into(),
        id: None,
        metadata: HashMap::new(),
    };
    assert!(blob.as_string().is_err());
    assert!(blob.as_bytes().is_err());
}

#[test]
fn blob_data_enum() {
    let text = BlobData::Text("hi".into());
    let bytes = BlobData::Bytes(vec![1, 2]);
    assert_ne!(text, bytes);
}

// --- Document trait tests ---

struct MockCompressor;

#[async_trait]
impl BaseDocumentCompressor for MockCompressor {
    async fn compress_documents(
        &self,
        documents: &[Document],
        query: &str,
    ) -> Result<Vec<Document>> {
        Ok(documents
            .iter()
            .filter(|d| d.page_content.contains(query))
            .cloned()
            .collect())
    }
}

#[tokio::test]
async fn document_compressor_filters_by_query() {
    let docs = vec![
        Document::new("Rust is great"),
        Document::new("Python is nice"),
        Document::new("Rust and safety"),
    ];
    let compressor = MockCompressor;
    let result = compressor.compress_documents(&docs, "Rust").await.unwrap();
    assert_eq!(result.len(), 2);
}

struct UpperCaseTransformer;

#[async_trait]
impl BaseDocumentTransformer for UpperCaseTransformer {
    async fn transform_documents(&self, documents: Vec<Document>) -> Result<Vec<Document>> {
        Ok(documents
            .into_iter()
            .map(|mut d| {
                d.page_content = d.page_content.to_uppercase();
                d
            })
            .collect())
    }
}

#[tokio::test]
async fn document_transformer_uppercases() {
    let docs = vec![Document::new("hello world")];
    let transformer = UpperCaseTransformer;
    let result = transformer.transform_documents(docs).await.unwrap();
    assert_eq!(result[0].page_content, "HELLO WORLD");
}

#[tokio::test]
async fn document_compressor_empty_input() {
    let compressor = MockCompressor;
    let result = compressor
        .compress_documents(&[], "anything")
        .await
        .unwrap();
    assert!(result.is_empty());
}

#[tokio::test]
async fn document_compressor_no_match() {
    let docs = vec![Document::new("hello world")];
    let compressor = MockCompressor;
    let result = compressor
        .compress_documents(&docs, "nonexistent")
        .await
        .unwrap();
    assert!(result.is_empty());
}

#[tokio::test]
async fn document_transformer_preserves_metadata() {
    let mut meta = HashMap::new();
    meta.insert("key".into(), json!("value"));
    let docs = vec![Document::new("hello").with_metadata(meta)];
    let transformer = UpperCaseTransformer;
    let result = transformer.transform_documents(docs).await.unwrap();
    assert_eq!(result[0].metadata["key"], json!("value"));
    assert_eq!(result[0].page_content, "HELLO");
}

#[tokio::test]
async fn document_transformer_empty_input() {
    let transformer = UpperCaseTransformer;
    let result = transformer.transform_documents(vec![]).await.unwrap();
    assert!(result.is_empty());
}

/// A compressor that scores documents and only keeps those above a threshold.
struct ScoringCompressor {
    threshold: usize,
}

#[async_trait]
impl BaseDocumentCompressor for ScoringCompressor {
    async fn compress_documents(
        &self,
        documents: &[Document],
        query: &str,
    ) -> Result<Vec<Document>> {
        Ok(documents
            .iter()
            .filter(|d| d.page_content.matches(query).count() >= self.threshold)
            .cloned()
            .collect())
    }
}

#[tokio::test]
async fn scoring_compressor_with_threshold() {
    let docs = vec![
        Document::new("rust rust rust"),
        Document::new("rust"),
        Document::new("python"),
    ];
    let compressor = ScoringCompressor { threshold: 2 };
    let result = compressor.compress_documents(&docs, "rust").await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].page_content, "rust rust rust");
}

/// A transformer that filters out documents with empty content.
struct FilterEmptyTransformer;

#[async_trait]
impl BaseDocumentTransformer for FilterEmptyTransformer {
    async fn transform_documents(&self, documents: Vec<Document>) -> Result<Vec<Document>> {
        Ok(documents
            .into_iter()
            .filter(|d| !d.page_content.is_empty())
            .collect())
    }
}

#[tokio::test]
async fn filter_empty_transformer() {
    let docs = vec![
        Document::new("content"),
        Document::new(""),
        Document::new("more content"),
    ];
    let transformer = FilterEmptyTransformer;
    let result = transformer.transform_documents(docs).await.unwrap();
    assert_eq!(result.len(), 2);
}
