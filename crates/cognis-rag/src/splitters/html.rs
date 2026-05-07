//! HTML-aware splitter — chunks an HTML document at heading boundaries
//! while preserving the heading hierarchy as document metadata.
//!
//! Lightweight implementation: scans the input character-by-character,
//! tracking `<h1>` … `<h6>` and `<section>` openings without pulling in a
//! full HTML parser. This keeps `cognis-rag` free of native deps while
//! covering the common cases. For full DOM-aware splitting use a custom
//! splitter built on `scraper`/`html5ever`.
//!
//! Customization knobs:
//! - `with_levels(set)` — which heading levels create chunk boundaries
//!   (default: 1..=6).
//! - `with_strip_tags(bool)` — strip remaining HTML tags from chunk text
//!   (default: `true`).
//! - `with_min_chunk_size(n)` — coalesce smaller-than-n leftover chunks
//!   into the previous one.

use std::collections::BTreeSet;

use crate::document::Document;

use super::{child_doc, TextSplitter};

/// HTML-aware splitter. Splits at heading boundaries; metadata records
/// the active heading at each level.
///
/// **Extension path**: implement [`crate::splitters::TextSplitter`]
/// directly when the heading-based scheme doesn't fit. The
/// [`HtmlSplitter::with_levels`] / [`HtmlSplitter::with_strip_tags`] /
/// [`HtmlSplitter::with_min_chunk_size`] knobs cover the common
/// configuration axes for the built-in heuristic.
#[derive(Debug, Clone)]
pub struct HtmlSplitter {
    /// Heading levels (1..=6) at which to create boundaries.
    levels: BTreeSet<u8>,
    /// Strip remaining HTML tags from chunk text.
    strip_tags: bool,
    /// Coalesce chunks smaller than this into the previous chunk.
    min_chunk_size: usize,
}

impl Default for HtmlSplitter {
    fn default() -> Self {
        Self {
            levels: (1u8..=6).collect(),
            strip_tags: true,
            min_chunk_size: 0,
        }
    }
}

impl HtmlSplitter {
    /// New splitter with default settings (split on every heading level).
    pub fn new() -> Self {
        Self::default()
    }

    /// Restrict to specific heading levels (e.g. `[1, 2]` for top-two).
    pub fn with_levels<I: IntoIterator<Item = u8>>(mut self, levels: I) -> Self {
        self.levels = levels.into_iter().filter(|n| (1..=6).contains(n)).collect();
        self
    }

    /// Whether to strip remaining HTML tags from chunk text. Default `true`.
    pub fn with_strip_tags(mut self, strip: bool) -> Self {
        self.strip_tags = strip;
        self
    }

    /// Coalesce trailing chunks smaller than `n` chars into the previous one.
    pub fn with_min_chunk_size(mut self, n: usize) -> Self {
        self.min_chunk_size = n;
        self
    }

    /// Locate every heading boundary (start byte, level, heading text)
    /// in `input`. Boundaries are returned in document order.
    fn find_boundaries(&self, input: &str) -> Vec<(usize, u8, String)> {
        let bytes = input.as_bytes();
        let mut out: Vec<(usize, u8, String)> = Vec::new();
        let mut i = 0;
        while i + 4 <= bytes.len() {
            // Look for `<hN` where N in 1..=6.
            if bytes[i] == b'<'
                && (bytes[i + 1] == b'h' || bytes[i + 1] == b'H')
                && bytes[i + 2].is_ascii_digit()
                && (1..=6).contains(&(bytes[i + 2] - b'0'))
            {
                let level = bytes[i + 2] - b'0';
                if !self.levels.contains(&level) {
                    i += 1;
                    continue;
                }
                // Find `>` ending the open-tag.
                let close = match input[i..].find('>') {
                    Some(p) => i + p,
                    None => break,
                };
                // Find `</hN>`.
                let needle = format!("</h{level}>");
                let needle_lower = needle.to_lowercase();
                let after = close + 1;
                let end_rel = input[after..].to_lowercase().find(&needle_lower);
                let end = match end_rel {
                    Some(p) => after + p,
                    None => break,
                };
                let heading_text = strip_tags(&input[after..end]);
                out.push((i, level, heading_text.trim().to_string()));
                i = end + needle.len();
                continue;
            }
            i += 1;
        }
        out
    }
}

fn strip_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut depth = 0i32;
    for ch in s.chars() {
        match ch {
            '<' => depth += 1,
            '>' if depth > 0 => depth -= 1,
            _ if depth == 0 => out.push(ch),
            _ => {}
        }
    }
    out
}

impl TextSplitter for HtmlSplitter {
    fn split(&self, doc: &Document) -> Vec<Document> {
        if doc.content.is_empty() {
            return Vec::new();
        }
        let boundaries = self.find_boundaries(&doc.content);
        if boundaries.is_empty() {
            // No headings — return whole doc (optionally tag-stripped).
            let content = if self.strip_tags {
                strip_tags(&doc.content).trim().to_string()
            } else {
                doc.content.clone()
            };
            if content.is_empty() {
                return Vec::new();
            }
            return vec![child_doc(doc, content, 0)];
        }
        // Walk boundaries → produce sections from boundary[i].start to
        // boundary[i+1].start (or end of doc for the last).
        let mut chunks: Vec<(String, Vec<(u8, String)>)> = Vec::new();
        // Track the current heading at each level.
        let mut current_levels: [Option<String>; 7] = Default::default();
        // Optional preamble before the first heading.
        let preamble_end = boundaries[0].0;
        if preamble_end > 0 {
            let body = &doc.content[..preamble_end];
            let stripped = if self.strip_tags {
                strip_tags(body).trim().to_string()
            } else {
                body.to_string()
            };
            if !stripped.is_empty() {
                chunks.push((stripped, Vec::new()));
            }
        }
        for (i, b) in boundaries.iter().enumerate() {
            let next_start = boundaries
                .get(i + 1)
                .map(|n| n.0)
                .unwrap_or(doc.content.len());
            let section = &doc.content[b.0..next_start];
            let level = b.1 as usize;
            current_levels[level] = Some(b.2.clone());
            // Reset deeper levels (they belong to the previous parent).
            for slot in current_levels.iter_mut().skip(level + 1) {
                *slot = None;
            }
            let stripped = if self.strip_tags {
                strip_tags(section).trim().to_string()
            } else {
                section.to_string()
            };
            if stripped.is_empty() {
                continue;
            }
            let crumbs: Vec<(u8, String)> = (1..=6)
                .filter_map(|lvl| {
                    current_levels[lvl as usize]
                        .as_ref()
                        .map(|s| (lvl, s.clone()))
                })
                .collect();
            chunks.push((stripped, crumbs));
        }
        // Coalesce small chunks.
        if self.min_chunk_size > 0 {
            let mut i = 0;
            while i + 1 < chunks.len() {
                if chunks[i + 1].0.chars().count() < self.min_chunk_size {
                    let trailing = chunks.remove(i + 1);
                    chunks[i].0.push_str("\n\n");
                    chunks[i].0.push_str(&trailing.0);
                    continue;
                }
                i += 1;
            }
        }
        chunks
            .into_iter()
            .enumerate()
            .map(|(i, (content, crumbs))| {
                let mut child = child_doc(doc, content, i);
                for (lvl, name) in crumbs {
                    child
                        .metadata
                        .insert(format!("h{lvl}"), serde_json::Value::String(name));
                }
                child
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(s: &str) -> Document {
        Document::new(s)
    }

    #[test]
    fn splits_on_h1_boundaries() {
        let s = HtmlSplitter::new();
        let d = doc("<h1>One</h1>first<h1>Two</h1>second");
        let out = s.split(&d);
        assert_eq!(out.len(), 2);
        assert!(out[0].content.contains("first"));
        assert!(out[1].content.contains("second"));
    }

    #[test]
    fn metadata_records_heading_breadcrumbs() {
        let s = HtmlSplitter::new();
        let d = doc("<h1>Top</h1>intro<h2>Sub</h2>body");
        let out = s.split(&d);
        // Last chunk has both h1 and h2 in scope.
        let last = out.last().unwrap();
        assert_eq!(
            last.metadata.get("h1"),
            Some(&serde_json::Value::String("Top".into()))
        );
        assert_eq!(
            last.metadata.get("h2"),
            Some(&serde_json::Value::String("Sub".into()))
        );
    }

    #[test]
    fn deeper_headings_clear_when_parent_changes() {
        let s = HtmlSplitter::new();
        let d = doc("<h1>A</h1>x<h2>A2</h2>y<h1>B</h1>z");
        let out = s.split(&d);
        // The chunk for "B" should not carry "A2".
        let last = out.last().unwrap();
        assert!(!last.metadata.contains_key("h2"));
        assert_eq!(
            last.metadata.get("h1"),
            Some(&serde_json::Value::String("B".into()))
        );
    }

    #[test]
    fn level_filter_only_splits_at_selected_levels() {
        let s = HtmlSplitter::new().with_levels([1u8]);
        let d = doc("<h1>One</h1>a<h2>Sub</h2>b<h1>Two</h1>c");
        let out = s.split(&d);
        // Only two chunks (h2 ignored as a boundary).
        assert_eq!(out.len(), 2);
        assert!(out[0].content.contains("a"));
        assert!(out[0].content.contains("b"));
    }

    #[test]
    fn strip_tags_strips_inner_markup() {
        let s = HtmlSplitter::new();
        let d = doc("<h1>One</h1><p>hello <b>world</b></p>");
        let out = s.split(&d);
        assert_eq!(out.len(), 1);
        assert!(out[0].content.contains("hello"));
        assert!(out[0].content.contains("world"));
        assert!(!out[0].content.contains("<b>"));
    }

    #[test]
    fn strip_tags_can_be_disabled() {
        let s = HtmlSplitter::new().with_strip_tags(false);
        let d = doc("<h1>One</h1><p>hello</p>");
        let out = s.split(&d);
        assert!(out[0].content.contains("<p>"));
    }

    #[test]
    fn doc_without_headings_returns_one_chunk() {
        let s = HtmlSplitter::new();
        let d = doc("<p>just a paragraph</p>");
        let out = s.split(&d);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].content, "just a paragraph");
    }

    #[test]
    fn preamble_before_first_heading_is_kept() {
        let s = HtmlSplitter::new();
        let d = doc("<p>preamble</p><h1>One</h1>body");
        let out = s.split(&d);
        assert!(out.len() >= 2);
        assert_eq!(out[0].content, "preamble");
    }

    #[test]
    fn min_chunk_size_coalesces_small_tail() {
        let s = HtmlSplitter::new().with_min_chunk_size(50);
        let d = doc("<h1>Big</h1>this is a longer body text<h1>Tiny</h1>x");
        let out = s.split(&d);
        // The "Tiny" section has only 1 char of body, below 50 → merged
        // back into the previous chunk.
        assert_eq!(out.len(), 1);
        assert!(out[0].content.contains("this is"));
        assert!(out[0].content.contains("x"));
    }

    #[test]
    fn empty_input_returns_empty() {
        let s = HtmlSplitter::new();
        let d = doc("");
        assert!(s.split(&d).is_empty());
    }
}
