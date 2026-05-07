//! Text splitters — chunk a [`Document`] into smaller `Document`s suitable for
//! embedding.

pub mod character;
pub mod code;
pub mod html;
pub mod json_splitter;
pub mod markdown;
pub mod recursive;
pub mod sentence;
pub mod token_aware;

pub use character::{CharacterSplitter, LengthFn};
pub use code::{CodeLanguage, CodeSplitter};
pub use html::HtmlSplitter;
pub use json_splitter::JsonSplitter;
pub use markdown::MarkdownSplitter;
pub use recursive::RecursiveCharSplitter;
pub use sentence::SentenceSplitter;
pub use token_aware::{CharTokenizer, FnTokenizer, TokenAwareSplitter, Tokenizer};

use crate::document::Document;

/// A text splitter takes a document and emits chunk-sized documents.
///
/// Implementations preserve the source's metadata on each chunk and add a
/// `chunk_index` field so callers can re-order if needed.
pub trait TextSplitter: Send + Sync {
    /// Split `doc` into chunks.
    fn split(&self, doc: &Document) -> Vec<Document>;

    /// Convenience: split many documents and concatenate the chunks.
    fn split_all(&self, docs: &[Document]) -> Vec<Document> {
        docs.iter().flat_map(|d| self.split(d)).collect()
    }
}

/// Helper used by every splitter to copy a parent document's metadata onto a
/// chunk and tag it with `chunk_index`.
pub(crate) fn child_doc(parent: &Document, content: String, chunk_index: usize) -> Document {
    let mut metadata = parent.metadata.clone();
    metadata.insert(
        "chunk_index".into(),
        serde_json::Value::Number(chunk_index.into()),
    );
    Document {
        id: None,
        content,
        metadata,
    }
}
