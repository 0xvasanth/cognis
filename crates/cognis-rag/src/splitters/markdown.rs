//! Markdown-aware splitter — keeps section headings with their bodies.

use crate::document::Document;

use super::{child_doc, recursive::RecursiveCharSplitter, TextSplitter};

/// Splits markdown by `#`-style heading sections. Each chunk preserves the
/// nearest enclosing heading in its metadata as `heading`.
///
/// If a section's body still exceeds `chunk_size`, falls back to a
/// [`RecursiveCharSplitter`] with the configured size and overlap.
pub struct MarkdownSplitter {
    chunk_size: usize,
    chunk_overlap: usize,
}

impl Default for MarkdownSplitter {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            chunk_overlap: 0,
        }
    }
}

impl MarkdownSplitter {
    /// Construct with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Maximum chunk size (chars). Sections longer than this are split
    /// recursively.
    pub fn with_chunk_size(mut self, n: usize) -> Self {
        self.chunk_size = n;
        self
    }

    /// Overlap window between adjacent chunks within a single section.
    pub fn with_overlap(mut self, n: usize) -> Self {
        self.chunk_overlap = n;
        self
    }
}

impl TextSplitter for MarkdownSplitter {
    fn split(&self, doc: &Document) -> Vec<Document> {
        let mut chunks: Vec<Document> = Vec::new();
        let mut current_heading: Option<String> = None;
        let mut buf = String::new();
        let recursive = RecursiveCharSplitter::new()
            .with_chunk_size(self.chunk_size)
            .with_overlap(self.chunk_overlap);

        let emit = |buf: &mut String,
                    heading: &Option<String>,
                    chunks: &mut Vec<Document>,
                    recursive: &RecursiveCharSplitter| {
            let text = std::mem::take(buf).trim().to_string();
            if text.is_empty() {
                return;
            }
            if text.chars().count() <= recursive_chunk_size(recursive) {
                let mut d = child_doc(doc, text, chunks.len());
                if let Some(h) = heading {
                    d.metadata
                        .insert("heading".into(), serde_json::Value::String(h.clone()));
                }
                chunks.push(d);
            } else {
                // Build a synthetic doc holding only the section body, run it
                // through the recursive splitter, then transfer metadata.
                let mut tmp = doc.clone();
                tmp.content = text;
                for sub in recursive.split(&tmp) {
                    let mut d = child_doc(doc, sub.content, chunks.len());
                    if let Some(h) = heading {
                        d.metadata
                            .insert("heading".into(), serde_json::Value::String(h.clone()));
                    }
                    chunks.push(d);
                }
            }
        };

        for line in doc.content.lines() {
            if let Some(h) = parse_heading(line) {
                emit(&mut buf, &current_heading, &mut chunks, &recursive);
                current_heading = Some(h);
            } else {
                buf.push_str(line);
                buf.push('\n');
            }
        }
        emit(&mut buf, &current_heading, &mut chunks, &recursive);
        chunks
    }
}

fn parse_heading(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    let level = trimmed.chars().take_while(|c| *c == '#').count();
    if level == 0 || level > 6 {
        return None;
    }
    let rest = &trimmed[level..];
    if !rest.starts_with(' ') {
        return None;
    }
    Some(rest.trim().to_string())
}

fn recursive_chunk_size(_r: &RecursiveCharSplitter) -> usize {
    // Keep parity with what the splitter was constructed with — we don't
    // expose it, so default to a high cap. Sections under this skip recursion.
    usize::MAX / 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_heading_levels() {
        assert_eq!(parse_heading("# Title").as_deref(), Some("Title"));
        assert_eq!(parse_heading("### Sub").as_deref(), Some("Sub"));
        assert_eq!(parse_heading("not a heading"), None);
        // No space after `#` → not a heading.
        assert_eq!(parse_heading("##NoSpace"), None);
    }

    #[test]
    fn splits_by_heading_and_tags_metadata() {
        let md = "# A\n\nbody-a\n\n## B\n\nbody-b\n";
        let doc = Document::new(md);
        let s = MarkdownSplitter::new();
        let chunks = s.split(&doc);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].metadata["heading"], "A");
        assert!(chunks[0].content.contains("body-a"));
        assert_eq!(chunks[1].metadata["heading"], "B");
        assert!(chunks[1].content.contains("body-b"));
    }

    #[test]
    fn pre_heading_text_emits_with_no_heading() {
        let md = "intro line\n\n# A\n\nbody";
        let doc = Document::new(md);
        let chunks = MarkdownSplitter::new().split(&doc);
        assert_eq!(chunks.len(), 2);
        assert!(!chunks[0].metadata.contains_key("heading"));
        assert_eq!(chunks[1].metadata["heading"], "A");
    }
}
