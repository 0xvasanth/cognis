mod character;
pub mod code;
mod html;
mod json;
mod markdown;
mod recursive;
pub mod sentence;
mod token;
pub mod token_aware;

pub use character::CharacterTextSplitter;
pub use code::Language;
pub use html::HTMLHeaderTextSplitter;
pub use json::RecursiveJsonSplitter;
pub use markdown::{MarkdownHeaderTextSplitter, MarkdownTextSplitter};
pub use recursive::{
    KeepSeparator, LengthFunction, RecursiveCharacterTextSplitter,
    RecursiveCharacterTextSplitterBuilder,
};
pub use sentence::{SentencePattern, SentenceTextSplitter, SentenceTextSplitterBuilder};
pub use token::TokenTextSplitter;
pub use token_aware::TokenAwareTextSplitter;

use rustchain_core::documents::Document;
use serde_json::Value;
use std::collections::HashMap;

/// Base trait for all text splitters.
pub trait TextSplitter: Send + Sync {
    /// Split a single text string into chunks.
    fn split_text(&self, text: &str) -> Vec<String>;

    /// The chunk size limit.
    fn chunk_size(&self) -> usize;

    /// The overlap between chunks.
    fn chunk_overlap(&self) -> usize;

    /// Create documents from texts with optional metadata.
    fn create_documents(
        &self,
        texts: &[&str],
        metadatas: Option<&[HashMap<String, Value>]>,
    ) -> Vec<Document> {
        let mut docs = Vec::new();
        for (i, text) in texts.iter().enumerate() {
            let metadata = metadatas
                .and_then(|m| m.get(i))
                .cloned()
                .unwrap_or_default();
            for chunk in self.split_text(text) {
                docs.push(Document::new(chunk).with_metadata(metadata.clone()));
            }
        }
        docs
    }

    /// Split documents into smaller documents.
    fn split_documents(&self, documents: &[Document]) -> Vec<Document> {
        let texts: Vec<&str> = documents.iter().map(|d| d.page_content.as_str()).collect();
        let metadatas: Vec<HashMap<String, Value>> =
            documents.iter().map(|d| d.metadata.clone()).collect();
        self.create_documents(&texts, Some(&metadatas))
    }
}

/// Merge small splits into chunks up to chunk_size, with overlap.
pub fn merge_splits(
    splits: &[&str],
    separator: &str,
    chunk_size: usize,
    chunk_overlap: usize,
) -> Vec<String> {
    let sep_len = separator.len();
    let mut docs: Vec<String> = Vec::new();
    let mut current_doc: Vec<&str> = Vec::new();
    let mut total: usize = 0;

    for piece in splits {
        let len = piece.len();
        let added = if current_doc.is_empty() {
            len
        } else {
            len + sep_len
        };

        if total + added > chunk_size && !current_doc.is_empty() {
            let doc = current_doc.join(separator);
            if !doc.is_empty() {
                docs.push(doc);
            }
            // Keep overlap
            if chunk_overlap == 0 {
                current_doc.clear();
                total = 0;
            } else {
                while total > chunk_overlap && current_doc.len() > 1 {
                    let removed = current_doc[0].len() + sep_len;
                    total -= removed;
                    current_doc.remove(0);
                }
            }
        }

        current_doc.push(piece);
        total = if current_doc.len() == 1 {
            len
        } else {
            total + len + sep_len
        };
    }

    if !current_doc.is_empty() {
        let doc = current_doc.join(separator);
        if !doc.is_empty() {
            docs.push(doc);
        }
    }
    docs
}
