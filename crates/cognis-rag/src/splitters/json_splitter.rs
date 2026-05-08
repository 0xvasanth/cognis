//! JSON-aware splitter — splits on top-level array elements / object pairs.
//!
//! Operates on the *string form* of the JSON. Parses, then walks the
//! structure top-level-first and emits chunks that fit `chunk_size`.

use crate::document::Document;

use super::{child_doc, TextSplitter};

/// JSON splitter. Behaviour:
/// - Top-level array → one chunk per element (or grouped if small).
/// - Top-level object → one chunk per key/value pair (or grouped if small).
/// - Anything else → one chunk for the whole value.
pub struct JsonSplitter {
    chunk_size: usize,
}

impl Default for JsonSplitter {
    fn default() -> Self {
        Self { chunk_size: 1000 }
    }
}

impl JsonSplitter {
    /// Construct.
    pub fn new() -> Self {
        Self::default()
    }
    /// Cap chunk size.
    pub fn with_chunk_size(mut self, n: usize) -> Self {
        self.chunk_size = n;
        self
    }
}

impl TextSplitter for JsonSplitter {
    fn split(&self, doc: &Document) -> Vec<Document> {
        let value: serde_json::Value = match serde_json::from_str(&doc.content) {
            Ok(v) => v,
            Err(_) => {
                // Not JSON — emit one chunk and let downstream cope.
                return vec![child_doc(doc, doc.content.clone(), 0)];
            }
        };

        let pieces = match value {
            serde_json::Value::Array(items) => items
                .into_iter()
                .map(|v| serde_json::to_string(&v).unwrap_or_default())
                .collect::<Vec<_>>(),
            serde_json::Value::Object(map) => map
                .into_iter()
                .map(|(k, v)| {
                    serde_json::to_string(&serde_json::json!({ k: v })).unwrap_or_default()
                })
                .collect::<Vec<_>>(),
            other => vec![serde_json::to_string(&other).unwrap_or_default()],
        };

        // Pack pieces into chunks up to chunk_size. A single element
        // larger than chunk_size still gets its own chunk (we'd rather
        // emit one oversized chunk than mangle the JSON to hit the cap).
        let mut chunks: Vec<String> = Vec::new();
        let mut buf = String::new();
        for p in pieces {
            let plen = p.chars().count();
            // Oversized element on its own: flush buf, emit p alone.
            if plen > self.chunk_size {
                if !buf.is_empty() {
                    chunks.push(std::mem::take(&mut buf));
                }
                chunks.push(p);
                continue;
            }
            if !buf.is_empty() && buf.chars().count() + plen + 1 > self.chunk_size {
                chunks.push(std::mem::take(&mut buf));
            }
            if !buf.is_empty() {
                buf.push('\n');
            }
            buf.push_str(&p);
        }
        if !buf.is_empty() {
            chunks.push(buf);
        }

        chunks
            .into_iter()
            .enumerate()
            .map(|(i, c)| child_doc(doc, c, i))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_root_one_chunk_per_element() {
        let doc = Document::new(r#"[{"a":1}, {"b":2}, {"c":3}]"#);
        let chunks = JsonSplitter::new().with_chunk_size(15).split(&doc);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn object_root_one_chunk_per_pair() {
        let doc = Document::new(r#"{"a":1, "b":2, "c":3}"#);
        let chunks = JsonSplitter::new().with_chunk_size(10).split(&doc);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn invalid_json_falls_through() {
        let doc = Document::new("not json");
        let chunks = JsonSplitter::new().split(&doc);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "not json");
    }
}
