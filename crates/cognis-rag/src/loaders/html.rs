//! HTML loader — strips tags via a small pure-Rust scanner.
//!
//! Doesn't use a full HTML parser (would pull a heavy dependency tree); instead
//! uses a streaming `<...>`-tag stripper that handles `<script>`/`<style>`
//! exclusion and decodes the most common HTML entities.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use futures::stream;

use cognis_core::{CognisError, Result};

use crate::document::Document;

use super::{DocumentLoader, DocumentStream};

/// Loads an HTML file as one [`Document`] containing its visible text.
pub struct HtmlLoader {
    path: PathBuf,
}

impl HtmlLoader {
    /// Construct a loader for the file at `path`.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }
}

#[async_trait]
impl DocumentLoader for HtmlLoader {
    async fn load(&self) -> Result<DocumentStream> {
        let html = tokio::fs::read_to_string(&self.path).await.map_err(|e| {
            CognisError::Configuration(format!("HtmlLoader: read `{}`: {e}", self.path.display()))
        })?;
        let text = strip_html(&html);
        let doc = Document::new(text)
            .with_metadata("source", self.path.display().to_string())
            .with_metadata("format", "html");
        Ok(Box::pin(stream::iter(vec![Ok(doc)])))
    }
}

/// Strip `<...>` tags from `html`, skipping the contents of `<script>` and
/// `<style>` blocks. Decodes the common entities `&amp; &lt; &gt; &quot;
/// &apos; &nbsp; &#39;`. Whitespace is collapsed.
pub(crate) fn strip_html(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let bytes = html.as_bytes();
    let lower = html.to_ascii_lowercase();
    let lower_bytes = lower.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        let b = bytes[i];
        if b == b'<' {
            // Skip <script>...</script> and <style>...</style> blocks.
            if let Some(end) = skip_block(lower_bytes, i, b"script") {
                out.push(' ');
                i = end;
                continue;
            }
            if let Some(end) = skip_block(lower_bytes, i, b"style") {
                out.push(' ');
                i = end;
                continue;
            }
            // Generic tag: skip to the next '>'.
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] != b'>' {
                j += 1;
            }
            out.push(' ');
            i = j.saturating_add(1);
            continue;
        }
        if b == b'&' {
            if let Some((entity, len)) = match_entity(&html[i..]) {
                out.push_str(entity);
                i += len;
                continue;
            }
        }
        // Multi-byte safe: advance one full UTF-8 char.
        let ch = html[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }

    // Collapse runs of whitespace.
    let mut collapsed = String::with_capacity(out.len());
    let mut prev_ws = false;
    for c in out.chars() {
        if c.is_whitespace() {
            if !prev_ws {
                collapsed.push(' ');
            }
            prev_ws = true;
        } else {
            collapsed.push(c);
            prev_ws = false;
        }
    }
    collapsed.trim().to_string()
}

/// If `<tag` starts at `start` in `lower`, return the byte position just
/// after the matching `</tag>`.
fn skip_block(lower: &[u8], start: usize, tag: &[u8]) -> Option<usize> {
    if start + tag.len() + 1 > lower.len() {
        return None;
    }
    if &lower[start..start + 1] != b"<" {
        return None;
    }
    if &lower[start + 1..start + 1 + tag.len()] != tag {
        return None;
    }
    // The next byte must be a tag terminator: `>`, whitespace, or EOF — i.e.
    // we shouldn't match `<scripted>` as `<script>`.
    let next = lower.get(start + 1 + tag.len())?;
    if !matches!(*next, b' ' | b'\t' | b'\n' | b'\r' | b'>' | b'/') {
        return None;
    }
    let close = [b"</", tag, b">"].concat();
    let rest = &lower[start..];
    let off = find_subslice(rest, &close)?;
    Some(start + off + close.len())
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn match_entity(s: &str) -> Option<(&'static str, usize)> {
    let entities = [
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", "\""),
        ("&apos;", "'"),
        ("&#39;", "'"),
        ("&nbsp;", " "),
    ];
    for (src, replacement) in entities {
        if s.starts_with(src) {
            return Some((replacement, src.len()));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn strip_basic() {
        let s = strip_html("<p>hello <b>world</b></p>");
        assert_eq!(s, "hello world");
    }

    #[test]
    fn strip_script_and_style() {
        let s = strip_html(
            "<style>body{color:red}</style>\
             before<script>alert(1)</script>after",
        );
        assert_eq!(s, "before after");
    }

    #[test]
    fn decode_entities() {
        let s = strip_html("a &amp; b &lt;c&gt;");
        assert_eq!(s, "a & b <c>");
    }

    #[tokio::test]
    async fn loads_html_file() {
        let mut f = NamedTempFile::new().unwrap();
        write!(f, "<html><body><p>hi</p></body></html>").unwrap();
        let docs = HtmlLoader::new(f.path()).load_all().await.unwrap();
        assert_eq!(docs[0].content, "hi");
        assert_eq!(docs[0].metadata["format"], "html");
    }
}
