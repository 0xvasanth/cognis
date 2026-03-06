//! Markdown file document loader.
//!
//! Parses YAML frontmatter (delimited by `---`) into metadata and loads
//! the remaining content as the document body.

use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use futures::stream;
use rustchain_core::document_loaders::BaseLoader;
use rustchain_core::document_loaders::DocumentStream;
use rustchain_core::documents::Document;
use rustchain_core::error::Result;
use serde_json::Value;

/// Loads a Markdown file as a single [`Document`].
///
/// If the file begins with YAML frontmatter (between `---` delimiters),
/// the key-value pairs are extracted into the document's metadata.
/// The content after the frontmatter becomes `page_content`.
///
/// # Example
/// ```no_run
/// use rustchain::document_loaders::markdown::MarkdownLoader;
/// use rustchain_core::document_loaders::BaseLoader;
///
/// # async fn example() -> rustchain_core::error::Result<()> {
/// let loader = MarkdownLoader::new("docs/readme.md");
/// let docs = loader.load().await?;
/// assert_eq!(docs.len(), 1);
/// # Ok(())
/// # }
/// ```
pub struct MarkdownLoader {
    path: PathBuf,
}

impl MarkdownLoader {
    /// Create a new `MarkdownLoader` for the given file path.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

/// Parse YAML frontmatter from markdown content.
///
/// Returns a tuple of (metadata, content_after_frontmatter).
/// If no frontmatter is found, metadata is empty and the full content is returned.
fn parse_frontmatter(raw: &str) -> (HashMap<String, Value>, String) {
    let trimmed = raw.trim_start();

    // Must start with "---" followed by a newline
    if !trimmed.starts_with("---") {
        return (HashMap::new(), raw.to_string());
    }

    let after_opening = &trimmed[3..];
    let after_opening = match after_opening.strip_prefix('\n') {
        Some(s) => s,
        None if after_opening.is_empty() => after_opening,
        None => return (HashMap::new(), raw.to_string()),
    };

    // Look for closing "---" at the start of a line
    // Handle the case where frontmatter is empty (closing --- is the very first line)
    if let Some(stripped) = after_opening.strip_prefix("---") {
        let remaining = stripped.strip_prefix('\n').unwrap_or(stripped);
        return (HashMap::new(), remaining.to_string());
    }

    if let Some(end_pos) = after_opening.find("\n---") {
        let frontmatter_str = &after_opening[..end_pos];
        let content_start = end_pos + 4; // skip "\n---"
        let remaining = &after_opening[content_start..];
        let remaining = remaining.strip_prefix('\n').unwrap_or(remaining);

        let metadata = parse_yaml_simple(frontmatter_str);
        (metadata, remaining.to_string())
    } else {
        // No closing delimiter found — treat as no frontmatter
        (HashMap::new(), raw.to_string())
    }
}

/// Simple YAML key-value parser for frontmatter.
///
/// Handles flat `key: value` pairs. Nested or complex values are stored
/// as their string representation.
fn parse_yaml_simple(yaml: &str) -> HashMap<String, Value> {
    let mut map = HashMap::new();

    for line in yaml.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, value)) = line.split_once(':') {
            let key = key.trim().to_string();
            let value = value.trim();

            if key.is_empty() {
                continue;
            }

            let json_value = parse_yaml_value(value);
            map.insert(key, json_value);
        }
    }

    map
}

/// Convert a YAML scalar value to a `serde_json::Value`.
fn parse_yaml_value(s: &str) -> Value {
    if s.is_empty() {
        return Value::Null;
    }

    // Boolean
    match s {
        "true" | "True" | "TRUE" | "yes" | "Yes" | "YES" => return Value::Bool(true),
        "false" | "False" | "FALSE" | "no" | "No" | "NO" => return Value::Bool(false),
        _ => {}
    }

    // Integer
    if let Ok(n) = s.parse::<i64>() {
        return Value::Number(n.into());
    }

    // Float
    if let Ok(f) = s.parse::<f64>() {
        if let Some(n) = serde_json::Number::from_f64(f) {
            return Value::Number(n);
        }
    }

    // Strip surrounding quotes if present
    let unquoted =
        if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
            &s[1..s.len() - 1]
        } else {
            s
        };

    // YAML list shorthand [a, b, c]
    if unquoted.starts_with('[') && unquoted.ends_with(']') {
        let inner = &unquoted[1..unquoted.len() - 1];
        let items: Vec<Value> = inner
            .split(',')
            .map(|v| parse_yaml_value(v.trim()))
            .collect();
        return Value::Array(items);
    }

    Value::String(unquoted.to_string())
}

#[async_trait]
impl BaseLoader for MarkdownLoader {
    async fn lazy_load(&self) -> Result<DocumentStream> {
        let raw = tokio::fs::read_to_string(&self.path).await?;
        let (mut metadata, content) = parse_frontmatter(&raw);

        metadata.insert(
            "source".to_string(),
            Value::String(self.path.display().to_string()),
        );
        metadata.insert(
            "content_type".to_string(),
            Value::String("text/markdown".to_string()),
        );

        let doc = Document::new(content).with_metadata(metadata);
        Ok(Box::pin(stream::iter(vec![Ok(doc)])))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_load_with_frontmatter() {
        let mut tmp = NamedTempFile::with_suffix(".md").unwrap();
        write!(
            tmp,
            "---\ntitle: Hello World\nauthor: Test\n---\n# Heading\n\nBody text."
        )
        .unwrap();

        let loader = MarkdownLoader::new(tmp.path());
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 1);
        assert_eq!(
            docs[0].metadata.get("title").unwrap(),
            &Value::String("Hello World".to_string())
        );
        assert_eq!(
            docs[0].metadata.get("author").unwrap(),
            &Value::String("Test".to_string())
        );
        assert_eq!(docs[0].page_content, "# Heading\n\nBody text.");
    }

    #[tokio::test]
    async fn test_load_without_frontmatter() {
        let mut tmp = NamedTempFile::with_suffix(".md").unwrap();
        write!(tmp, "# Just a heading\n\nSome content.").unwrap();

        let loader = MarkdownLoader::new(tmp.path());
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "# Just a heading\n\nSome content.");
        // Only source and content_type metadata, no frontmatter keys
        assert!(docs[0].metadata.get("title").is_none());
    }

    #[tokio::test]
    async fn test_load_with_empty_frontmatter() {
        let mut tmp = NamedTempFile::with_suffix(".md").unwrap();
        write!(tmp, "---\n---\nContent after empty frontmatter.").unwrap();

        let loader = MarkdownLoader::new(tmp.path());
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "Content after empty frontmatter.");
        // Should have source and content_type but no frontmatter keys
        assert!(docs[0].metadata.contains_key("source"));
        assert!(docs[0].metadata.contains_key("content_type"));
    }

    #[tokio::test]
    async fn test_load_with_complex_metadata() {
        let mut tmp = NamedTempFile::with_suffix(".md").unwrap();
        write!(
            tmp,
            "---\ntitle: Complex Post\ndate: 2025-01-15\ndraft: true\ncount: 42\ntags: [rust, llm, ai]\n---\n# Complex\n\nBody."
        )
        .unwrap();

        let loader = MarkdownLoader::new(tmp.path());
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 1);
        assert_eq!(
            docs[0].metadata.get("title").unwrap(),
            &Value::String("Complex Post".to_string())
        );
        assert_eq!(docs[0].metadata.get("draft").unwrap(), &Value::Bool(true));
        assert_eq!(
            docs[0].metadata.get("count").unwrap(),
            &Value::Number(42.into())
        );
        // tags parsed as array
        let tags = docs[0].metadata.get("tags").unwrap();
        assert!(tags.is_array());
        assert_eq!(tags.as_array().unwrap().len(), 3);
    }

    #[tokio::test]
    async fn test_page_content_excludes_frontmatter() {
        let mut tmp = NamedTempFile::with_suffix(".md").unwrap();
        write!(tmp, "---\ntitle: Test\n---\nActual content here.").unwrap();

        let loader = MarkdownLoader::new(tmp.path());
        let docs = loader.load().await.unwrap();

        assert_eq!(docs[0].page_content, "Actual content here.");
        assert!(!docs[0].page_content.contains("---"));
        assert!(!docs[0].page_content.contains("title"));
    }

    #[tokio::test]
    async fn test_source_metadata_present() {
        let mut tmp = NamedTempFile::with_suffix(".md").unwrap();
        write!(tmp, "# Hello").unwrap();

        let loader = MarkdownLoader::new(tmp.path());
        let docs = loader.load().await.unwrap();

        assert_eq!(
            docs[0].metadata.get("source").unwrap(),
            &Value::String(tmp.path().display().to_string())
        );
    }

    #[test]
    fn test_parse_frontmatter_no_closing_delimiter() {
        let raw = "---\ntitle: Broken\nNo closing delimiter\n";
        let (metadata, content) = parse_frontmatter(raw);
        assert!(metadata.is_empty());
        assert_eq!(content, raw);
    }

    #[test]
    fn test_parse_yaml_value_types() {
        assert_eq!(parse_yaml_value("true"), Value::Bool(true));
        assert_eq!(parse_yaml_value("false"), Value::Bool(false));
        assert_eq!(parse_yaml_value("42"), Value::Number(42.into()));
        assert_eq!(
            parse_yaml_value("hello"),
            Value::String("hello".to_string())
        );
        assert_eq!(parse_yaml_value(""), Value::Null);
    }
}
