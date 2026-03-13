//! HTML file document loader.

use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use cognis_core::document_loaders::BaseLoader;
use cognis_core::document_loaders::DocumentStream;
use cognis_core::documents::Document;
use cognis_core::error::Result;
use futures::stream;
use regex::Regex;
use serde_json::Value;

/// Extract metadata from `<meta>` tags in the HTML.
///
/// Looks for `<meta name="..." content="...">` and
/// `<meta property="..." content="...">` patterns.
pub fn extract_meta_tags(html: &str) -> HashMap<String, String> {
    let mut meta = HashMap::new();
    let re = Regex::new(
        r#"(?i)<meta\s+(?:name|property)\s*=\s*["']([^"']+)["']\s+content\s*=\s*["']([^"']+)["'][^>]*/?\s*>"#,
    )
    .unwrap();
    for cap in re.captures_iter(html) {
        if let (Some(name), Some(content)) = (cap.get(1), cap.get(2)) {
            meta.insert(name.as_str().to_string(), content.as_str().to_string());
        }
    }
    // Also match reversed attribute order: content before name
    let re_rev = Regex::new(
        r#"(?i)<meta\s+content\s*=\s*["']([^"']+)["']\s+(?:name|property)\s*=\s*["']([^"']+)["'][^>]*/?\s*>"#,
    )
    .unwrap();
    for cap in re_rev.captures_iter(html) {
        if let (Some(content), Some(name)) = (cap.get(1), cap.get(2)) {
            meta.insert(name.as_str().to_string(), content.as_str().to_string());
        }
    }
    meta
}

/// Extract the `<title>` content from HTML.
pub fn extract_title(html: &str) -> Option<String> {
    let re = Regex::new(r"(?i)<title[^>]*>(.*?)</title>").unwrap();
    re.captures(html)
        .and_then(|cap| cap.get(1))
        .map(|m| m.as_str().trim().to_string())
        .filter(|s| !s.is_empty())
}

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
/// use cognis::document_loaders::html::HTMLLoader;
/// use cognis_core::document_loaders::BaseLoader;
///
/// # async fn example() -> cognis_core::error::Result<()> {
/// let loader = HTMLLoader::new("page.html");
/// let docs = loader.load().await?;
/// assert_eq!(docs.len(), 1);
/// # Ok(())
/// # }
/// ```
pub struct HTMLLoader {
    path: PathBuf,
    /// When true, extract `<meta>` tags and `<title>` into metadata.
    extract_metadata: bool,
    /// Optional tag name to restrict content extraction to (e.g., `"article"`, `"main"`).
    content_tag: Option<String>,
}

impl HTMLLoader {
    /// Create a new `HTMLLoader` for the given file path.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            extract_metadata: false,
            content_tag: None,
        }
    }

    /// Enable extraction of `<meta>` tags and `<title>` into document metadata.
    pub fn with_metadata_extraction(mut self) -> Self {
        self.extract_metadata = true;
        self
    }

    /// Restrict content extraction to within a specific HTML tag (e.g., `"article"`, `"main"`).
    ///
    /// Only the text inside the first occurrence of the specified tag will be extracted.
    pub fn with_content_tag(mut self, tag: impl Into<String>) -> Self {
        self.content_tag = Some(tag.into());
        self
    }
}

/// Extract the content of a specific HTML tag (first occurrence).
fn extract_tag_content(html: &str, tag: &str) -> Option<String> {
    let pattern = format!(r"(?is)<{tag}[^>]*>(.*?)</{tag}>");
    let re = Regex::new(&pattern).ok()?;
    re.captures(html)
        .and_then(|cap| cap.get(1))
        .map(|m| m.as_str().to_string())
}

#[async_trait]
impl BaseLoader for HTMLLoader {
    async fn lazy_load(&self) -> Result<DocumentStream> {
        let raw = tokio::fs::read_to_string(&self.path).await?;

        // Determine the HTML fragment to extract text from.
        let html_fragment = match &self.content_tag {
            Some(tag) => extract_tag_content(&raw, tag).unwrap_or_else(|| raw.clone()),
            None => raw.clone(),
        };

        let content = extract_text_from_html(&html_fragment);

        let mut metadata = HashMap::new();
        metadata.insert(
            "source".to_string(),
            Value::String(self.path.display().to_string()),
        );
        metadata.insert(
            "content_type".to_string(),
            Value::String("text/html".to_string()),
        );

        if self.extract_metadata {
            if let Some(title) = extract_title(&raw) {
                metadata.insert("title".to_string(), Value::String(title));
            }
            for (key, value) in extract_meta_tags(&raw) {
                metadata.insert(format!("meta:{}", key), Value::String(value));
            }
        }

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

    #[tokio::test]
    async fn test_html_loader_with_metadata_extraction() {
        let mut tmp = NamedTempFile::with_suffix(".html").unwrap();
        write!(
            tmp,
            r#"<html><head>
            <title>My Page</title>
            <meta name="description" content="A test page">
            <meta property="og:title" content="OG Title">
            </head><body><p>Content here</p></body></html>"#
        )
        .unwrap();

        let loader = HTMLLoader::new(tmp.path()).with_metadata_extraction();
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 1);
        assert_eq!(
            docs[0].metadata.get("title").unwrap(),
            &Value::String("My Page".to_string())
        );
        assert_eq!(
            docs[0].metadata.get("meta:description").unwrap(),
            &Value::String("A test page".to_string())
        );
        assert_eq!(
            docs[0].metadata.get("meta:og:title").unwrap(),
            &Value::String("OG Title".to_string())
        );
    }

    #[tokio::test]
    async fn test_html_loader_with_content_tag() {
        let mut tmp = NamedTempFile::with_suffix(".html").unwrap();
        write!(
            tmp,
            r#"<html><body>
            <header>Navigation stuff</header>
            <article><p>Important article content</p></article>
            <footer>Footer stuff</footer>
            </body></html>"#
        )
        .unwrap();

        let loader = HTMLLoader::new(tmp.path()).with_content_tag("article");
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "Important article content");
        // Should NOT contain header or footer text
        assert!(!docs[0].page_content.contains("Navigation"));
        assert!(!docs[0].page_content.contains("Footer"));
    }
}
