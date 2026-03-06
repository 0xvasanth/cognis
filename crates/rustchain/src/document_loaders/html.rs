//! HTML file document loader.

use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use futures::stream;
use regex::Regex;
use rustchain_core::document_loaders::BaseLoader;
use rustchain_core::document_loaders::DocumentStream;
use rustchain_core::documents::Document;
use rustchain_core::error::Result;
use serde_json::Value;

/// Extracts plain text from an HTML string.
///
/// This function:
/// 1. Strips `<script>` and `<style>` blocks entirely
/// 2. Strips all remaining HTML tags
/// 3. Decodes basic HTML entities
/// 4. Collapses multiple whitespace/newlines
/// 5. Trims leading/trailing whitespace
pub fn extract_text_from_html(html: &str) -> String {
    // Remove script blocks
    let re_script = Regex::new(r"(?is)<script[^>]*>.*?</script>").unwrap();
    let text = re_script.replace_all(html, "");

    // Remove style blocks
    let re_style = Regex::new(r"(?is)<style[^>]*>.*?</style>").unwrap();
    let text = re_style.replace_all(&text, "");

    // Strip all HTML tags
    let re_tags = Regex::new(r"<[^>]*>").unwrap();
    let text = re_tags.replace_all(&text, " ");

    // Decode HTML entities
    let text = text
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ");

    // Collapse whitespace
    let re_ws = Regex::new(r"\s+").unwrap();
    let text = re_ws.replace_all(&text, " ");

    text.trim().to_string()
}

/// Loads a local HTML file and extracts its text content.
///
/// The HTML is parsed using simple regex-based extraction: script and style
/// blocks are removed, tags are stripped, and basic HTML entities are decoded.
///
/// # Example
/// ```no_run
/// use rustchain::document_loaders::html::HTMLLoader;
/// use rustchain_core::document_loaders::BaseLoader;
///
/// # async fn example() -> rustchain_core::error::Result<()> {
/// let loader = HTMLLoader::new("page.html");
/// let docs = loader.load().await?;
/// assert_eq!(docs.len(), 1);
/// # Ok(())
/// # }
/// ```
pub struct HTMLLoader {
    path: PathBuf,
}

impl HTMLLoader {
    /// Create a new `HTMLLoader` for the given file path.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

#[async_trait]
impl BaseLoader for HTMLLoader {
    async fn lazy_load(&self) -> Result<DocumentStream> {
        let raw = tokio::fs::read_to_string(&self.path).await?;
        let content = extract_text_from_html(&raw);

        let mut metadata = HashMap::new();
        metadata.insert(
            "source".to_string(),
            Value::String(self.path.display().to_string()),
        );
        metadata.insert(
            "content_type".to_string(),
            Value::String("text/html".to_string()),
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

    #[test]
    fn test_strip_html_tags() {
        let html = "<p>Hello <b>world</b></p>";
        let text = extract_text_from_html(html);
        assert_eq!(text, "Hello world");
    }

    #[test]
    fn test_decode_html_entities() {
        let html = "Tom &amp; Jerry &lt;3&gt; &quot;friends&quot; &#39;forever&#39; &nbsp;ok";
        let text = extract_text_from_html(html);
        assert_eq!(text, "Tom & Jerry <3> \"friends\" 'forever' ok");
    }

    #[test]
    fn test_remove_script_and_style() {
        let html = r#"<html><head><style>body{color:red}</style></head>
            <body><script type="text/javascript">alert('hi');</script>
            <p>Visible text</p></body></html>"#;
        let text = extract_text_from_html(html);
        assert_eq!(text, "Visible text");
    }

    #[test]
    fn test_collapse_whitespace() {
        let html = "<p>  lots   of   \n\n  space  </p>";
        let text = extract_text_from_html(html);
        assert_eq!(text, "lots of space");
    }

    #[tokio::test]
    async fn test_html_loader() {
        let mut tmp = NamedTempFile::with_suffix(".html").unwrap();
        write!(
            tmp,
            "<html><body><h1>Title</h1><p>Some &amp; content</p></body></html>"
        )
        .unwrap();

        let loader = HTMLLoader::new(tmp.path());
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "Title Some & content");
        assert_eq!(
            docs[0].metadata.get("source").unwrap(),
            &Value::String(tmp.path().display().to_string())
        );
        assert_eq!(
            docs[0].metadata.get("content_type").unwrap(),
            &Value::String("text/html".to_string())
        );
    }

    #[test]
    fn test_multiline_script_removal() {
        let html = r#"<div>Before</div>
<script>
  var x = 1;
  var y = 2;
</script>
<div>After</div>"#;
        let text = extract_text_from_html(html);
        assert_eq!(text, "Before After");
    }
}
