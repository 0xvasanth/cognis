//! Web scraping document loader and crawler.
//!
//! Provides [`WebLoader`] for fetching multiple URLs and extracting text content,
//! [`WebCrawler`] for recursively crawling pages within a domain, and the simpler
//! [`WebBaseLoader`] for fetching a single URL.

use std::collections::{HashMap, HashSet, VecDeque};

use async_trait::async_trait;
use futures::stream;
use regex::Regex;
use reqwest::Client;
use rustchain_core::document_loaders::BaseLoader;
use rustchain_core::document_loaders::DocumentStream;
use rustchain_core::documents::Document;
use rustchain_core::error::{Result, RustChainError};
use serde_json::Value;

use super::html::{extract_text_from_html, extract_title};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Extract text content matching a CSS-selector-like pattern using regex.
///
/// Supports simple selectors:
/// - Tag name: `article`, `main`, `div`
/// - Class: `.content`, `div.content`
/// - ID: `#main`, `div#main`
///
/// Returns the inner HTML of the first matching element, or `None`.
pub fn extract_by_selector(html: &str, selector: &str) -> Option<String> {
    // Note: Rust's `regex` crate does not support backreferences, so all
    // patterns use an explicit tag name or a generic `\w+` tag match with
    // the closing tag also matched generically.
    let pattern = if let Some(id_part) = selector.strip_prefix('#') {
        // ID selector: #my-id
        let id = regex::escape(id_part);
        format!(r#"(?is)<\w+[^>]*\bid\s*=\s*["']{id}["'][^>]*>(.*?)</\w+>"#)
    } else if let Some(class_part) = selector.strip_prefix('.') {
        // Class selector: .my-class — match class attribute containing the class name
        let class = regex::escape(class_part);
        format!(r#"(?is)<\w+[^>]*\bclass\s*=\s*["'][^"']*{class}[^"']*["'][^>]*>(.*?)</\w+>"#)
    } else if selector.contains('#') {
        // tag#id
        let parts: Vec<&str> = selector.splitn(2, '#').collect();
        let tag = parts[0];
        let id = regex::escape(parts[1]);
        format!(r#"(?is)<{tag}[^>]*\bid\s*=\s*["']{id}["'][^>]*>(.*?)</{tag}>"#)
    } else if selector.contains('.') {
        // tag.class
        let parts: Vec<&str> = selector.splitn(2, '.').collect();
        let tag = parts[0];
        let class = regex::escape(parts[1]);
        format!(r#"(?is)<{tag}[^>]*\bclass\s*=\s*["'][^"']*{class}[^"']*["'][^>]*>(.*?)</{tag}>"#)
    } else {
        // Simple tag name
        format!(r"(?is)<{selector}[^>]*>(.*?)</{selector}>")
    };

    let re = Regex::new(&pattern).ok()?;
    re.captures(html)
        .and_then(|cap| cap.get(1).map(|m| m.as_str().to_string()))
}

/// Extract all `<a href="...">` links from HTML.
///
/// Filters out anchors (`#`), `javascript:`, and `mailto:` links.
pub fn extract_links(html: &str) -> Vec<String> {
    let re = Regex::new(r#"(?i)<a\s[^>]*href\s*=\s*["']([^"']+)["'][^>]*>"#).unwrap();
    re.captures_iter(html)
        .filter_map(|cap| {
            let href = cap.get(1)?.as_str().to_string();
            if href.starts_with('#')
                || href.starts_with("javascript:")
                || href.starts_with("mailto:")
            {
                None
            } else {
                Some(href)
            }
        })
        .collect()
}

/// Resolve a potentially relative URL against a base URL.
///
/// Handles common cases: absolute URLs, protocol-relative, root-relative,
/// and relative paths.
fn resolve_url(base: &str, href: &str) -> Option<String> {
    let href = href.trim();
    if href.is_empty() || href.starts_with("javascript:") || href.starts_with("mailto:") {
        return None;
    }
    if href.starts_with("http://") || href.starts_with("https://") {
        return Some(href.to_string());
    }
    let scheme_end = base.find("://").map(|i| i + 3)?;
    let scheme = &base[..scheme_end];
    let rest = &base[scheme_end..];
    let host_end = rest.find('/').unwrap_or(rest.len());
    let host = &rest[..host_end];
    let origin = format!("{}{}", scheme, host);

    if href.starts_with("//") {
        let proto = &base[..base.find("://").unwrap()];
        return Some(format!("{}:{}", proto, href));
    }
    if href.starts_with('/') {
        return Some(format!("{}{}", origin, href));
    }
    // Relative path: append to base directory
    let base_dir = if let Some(last_slash) = base.rfind('/') {
        if last_slash >= scheme_end {
            &base[..=last_slash]
        } else {
            &format!("{}/", base)
        }
    } else {
        base
    };
    Some(format!("{}{}", base_dir, href))
}

/// Extract the domain (host) from a URL.
fn extract_domain(url: &str) -> Option<String> {
    let after_scheme = url.find("://").map(|i| &url[i + 3..])?;
    let host_end = after_scheme.find('/').unwrap_or(after_scheme.len());
    Some(after_scheme[..host_end].to_string())
}

/// Build a `Document` from raw HTML and a source URL.
fn build_document_from_html(html: &str, source_url: &str, css_selector: Option<&str>) -> Document {
    let fragment = match css_selector {
        Some(sel) => extract_by_selector(html, sel).unwrap_or_else(|| html.to_string()),
        None => html.to_string(),
    };
    let content = extract_text_from_html(&fragment);
    let title = extract_title(html);

    let mut metadata = HashMap::new();
    metadata.insert("source".to_string(), Value::String(source_url.to_string()));
    metadata.insert(
        "content_type".to_string(),
        Value::String("text/html".to_string()),
    );
    if let Some(t) = title {
        metadata.insert("title".to_string(), Value::String(t));
    }
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    metadata.insert(
        "fetch_timestamp".to_string(),
        Value::Number(serde_json::Number::from(timestamp)),
    );

    Document::new(content).with_metadata(metadata)
}

// ---------------------------------------------------------------------------
// WebBaseLoader (simple single-URL loader, preserved for backward compat)
// ---------------------------------------------------------------------------

/// Fetches a URL and extracts its text content from HTML.
///
/// Uses `reqwest` to perform a GET request and then applies the same
/// HTML text extraction as [`super::html::HTMLLoader`].
///
/// # Example
/// ```no_run
/// use rustchain::document_loaders::web::WebBaseLoader;
/// use rustchain_core::document_loaders::BaseLoader;
///
/// # async fn example() -> rustchain_core::error::Result<()> {
/// let loader = WebBaseLoader::new("https://example.com");
/// let docs = loader.load().await?;
/// assert_eq!(docs.len(), 1);
/// # Ok(())
/// # }
/// ```
pub struct WebBaseLoader {
    url: String,
    client: Client,
}

impl WebBaseLoader {
    /// Create a new `WebBaseLoader` for the given URL.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            client: Client::new(),
        }
    }

    /// Use a custom `reqwest::Client`.
    pub fn with_client(mut self, client: Client) -> Self {
        self.client = client;
        self
    }
}

#[async_trait]
impl BaseLoader for WebBaseLoader {
    async fn lazy_load(&self) -> Result<DocumentStream> {
        let response = self
            .client
            .get(&self.url)
            .send()
            .await
            .map_err(|e| RustChainError::Other(format!("HTTP request failed: {}", e)))?;

        let status = response.status();
        if !status.is_success() {
            return Err(RustChainError::Other(format!(
                "HTTP request returned status {}",
                status
            )));
        }

        let raw_html = response
            .text()
            .await
            .map_err(|e| RustChainError::Other(format!("Failed to read response body: {}", e)))?;

        let content = extract_text_from_html(&raw_html);

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), Value::String(self.url.clone()));
        metadata.insert(
            "content_type".to_string(),
            Value::String("text/html".to_string()),
        );

        let doc = Document::new(content).with_metadata(metadata);
        Ok(Box::pin(stream::iter(vec![Ok(doc)])))
    }
}

// ---------------------------------------------------------------------------
// WebLoader (multi-URL with builder pattern)
// ---------------------------------------------------------------------------

/// A document loader that fetches multiple URLs and extracts text content.
///
/// Supports custom headers, CSS-selector-like content extraction, configurable
/// timeouts, and user-agent strings. Uses a builder pattern for configuration.
///
/// # Example
/// ```no_run
/// use rustchain::document_loaders::web::WebLoader;
///
/// # async fn example() -> rustchain_core::error::Result<()> {
/// let docs = WebLoader::new(vec!["https://example.com".to_string()])
///     .with_timeout(15)
///     .with_css_selector("article")
///     .load_async()
///     .await?;
/// # Ok(())
/// # }
/// ```
pub struct WebLoader {
    /// URLs to fetch.
    urls: Vec<String>,
    /// Optional CSS selector for content extraction.
    css_selector: Option<String>,
    /// Custom HTTP headers.
    headers: HashMap<String, String>,
    /// Request timeout in seconds.
    timeout_secs: u64,
    /// User-Agent string.
    user_agent: String,
}

impl WebLoader {
    /// Create a new `WebLoader` for the given URLs.
    pub fn new(urls: Vec<String>) -> Self {
        Self {
            urls,
            css_selector: None,
            headers: HashMap::new(),
            timeout_secs: 30,
            user_agent: "rustchain-web-loader/0.1".to_string(),
        }
    }

    /// Set a CSS selector to extract specific content from the page.
    ///
    /// Supported selector forms: `tag`, `.class`, `#id`, `tag.class`, `tag#id`.
    pub fn with_css_selector(mut self, selector: impl Into<String>) -> Self {
        self.css_selector = Some(selector.into());
        self
    }

    /// Set custom HTTP headers to send with each request.
    pub fn with_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers = headers;
        self
    }

    /// Add a single custom header.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// Set the request timeout in seconds (default: 30).
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Set a custom User-Agent string.
    pub fn with_user_agent(mut self, agent: impl Into<String>) -> Self {
        self.user_agent = agent.into();
        self
    }

    /// Build a `reqwest::Client` from the current configuration.
    fn build_client(&self) -> std::result::Result<Client, RustChainError> {
        let mut builder = Client::builder()
            .timeout(std::time::Duration::from_secs(self.timeout_secs))
            .user_agent(&self.user_agent);

        let mut header_map = reqwest::header::HeaderMap::new();
        for (k, v) in &self.headers {
            let name = reqwest::header::HeaderName::from_bytes(k.as_bytes()).map_err(|e| {
                RustChainError::Other(format!("Invalid header name '{}': {}", k, e))
            })?;
            let val = reqwest::header::HeaderValue::from_str(v).map_err(|e| {
                RustChainError::Other(format!("Invalid header value '{}': {}", v, e))
            })?;
            header_map.insert(name, val);
        }
        builder = builder.default_headers(header_map);

        builder
            .build()
            .map_err(|e| RustChainError::Other(format!("Failed to build HTTP client: {}", e)))
    }

    /// Asynchronously load all URLs and return documents.
    ///
    /// Failed URLs are skipped with an `eprintln!` warning rather than
    /// aborting the entire operation.
    pub async fn load_async(&self) -> Result<Vec<Document>> {
        let client = self.build_client()?;
        let mut documents = Vec::new();

        for url in &self.urls {
            match self.fetch_url(&client, url).await {
                Ok(doc) => documents.push(doc),
                Err(e) => {
                    eprintln!("Warning: failed to fetch '{}': {}", url, e);
                }
            }
        }

        Ok(documents)
    }

    /// Synchronous-style load (blocks on the async runtime).
    ///
    /// This creates a new tokio runtime. Prefer `load_async` when already
    /// inside an async context.
    pub fn load_sync(&self) -> Result<Vec<Document>> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| RustChainError::Other(format!("Failed to create runtime: {}", e)))?;
        rt.block_on(self.load_async())
    }

    /// Fetch a single URL and build a document.
    async fn fetch_url(&self, client: &Client, url: &str) -> Result<Document> {
        let response = client.get(url).send().await.map_err(|e| {
            RustChainError::Other(format!("HTTP request to '{}' failed: {}", url, e))
        })?;

        let status = response.status();
        if !status.is_success() {
            return Err(RustChainError::Other(format!(
                "HTTP {} for '{}'",
                status, url
            )));
        }

        let raw_html = response.text().await.map_err(|e| {
            RustChainError::Other(format!("Failed to read body from '{}': {}", url, e))
        })?;

        Ok(build_document_from_html(
            &raw_html,
            url,
            self.css_selector.as_deref(),
        ))
    }
}

#[async_trait]
impl BaseLoader for WebLoader {
    async fn lazy_load(&self) -> Result<DocumentStream> {
        let docs = self.load_async().await?;
        Ok(Box::pin(stream::iter(docs.into_iter().map(Ok))))
    }
}

// ---------------------------------------------------------------------------
// WebCrawler
// ---------------------------------------------------------------------------

/// A simple web crawler that follows links within the same domain.
///
/// Starting from a root URL, it fetches pages, extracts links, and crawls
/// up to `max_depth` levels deep. Only same-domain links are followed.
///
/// # Example
/// ```no_run
/// use rustchain::document_loaders::web::WebCrawler;
///
/// # async fn example() -> rustchain_core::error::Result<()> {
/// let docs = WebCrawler::new("https://example.com")
///     .with_max_depth(2)
///     .crawl()
///     .await?;
/// # Ok(())
/// # }
/// ```
pub struct WebCrawler {
    /// Starting URL for the crawl.
    start_url: String,
    /// Maximum depth to crawl (0 = only the start page).
    max_depth: usize,
    /// Request timeout in seconds.
    timeout_secs: u64,
    /// User-Agent string.
    user_agent: String,
    /// Custom HTTP headers.
    headers: HashMap<String, String>,
    /// Optional CSS selector for content extraction.
    css_selector: Option<String>,
}

impl WebCrawler {
    /// Create a new `WebCrawler` starting at the given URL.
    pub fn new(start_url: impl Into<String>) -> Self {
        Self {
            start_url: start_url.into(),
            max_depth: 1,
            timeout_secs: 30,
            user_agent: "rustchain-web-crawler/0.1".to_string(),
            headers: HashMap::new(),
            css_selector: None,
        }
    }

    /// Set the maximum crawl depth (default: 1).
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Set the request timeout in seconds.
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Set a custom User-Agent string.
    pub fn with_user_agent(mut self, agent: impl Into<String>) -> Self {
        self.user_agent = agent.into();
        self
    }

    /// Set custom HTTP headers.
    pub fn with_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers = headers;
        self
    }

    /// Set a CSS selector to extract specific content.
    pub fn with_css_selector(mut self, selector: impl Into<String>) -> Self {
        self.css_selector = Some(selector.into());
        self
    }

    /// Extract same-domain links from HTML, resolved against the given base URL.
    pub fn extract_same_domain_links(html: &str, base_url: &str) -> Vec<String> {
        let base_domain = match extract_domain(base_url) {
            Some(d) => d,
            None => return vec![],
        };

        extract_links(html)
            .into_iter()
            .filter_map(|href| resolve_url(base_url, &href))
            .filter(|url| {
                extract_domain(url)
                    .map(|d| d == base_domain)
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Crawl starting from `start_url` and return documents for all visited pages.
    pub async fn crawl(&self) -> Result<Vec<Document>> {
        let client = {
            let mut builder = Client::builder()
                .timeout(std::time::Duration::from_secs(self.timeout_secs))
                .user_agent(&self.user_agent);

            let mut header_map = reqwest::header::HeaderMap::new();
            for (k, v) in &self.headers {
                if let (Ok(name), Ok(val)) = (
                    reqwest::header::HeaderName::from_bytes(k.as_bytes()),
                    reqwest::header::HeaderValue::from_str(v),
                ) {
                    header_map.insert(name, val);
                }
            }
            builder = builder.default_headers(header_map);
            builder
                .build()
                .map_err(|e| RustChainError::Other(format!("Failed to build client: {}", e)))?
        };

        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();
        let mut documents: Vec<Document> = Vec::new();

        queue.push_back((self.start_url.clone(), 0));
        visited.insert(self.start_url.clone());

        while let Some((url, depth)) = queue.pop_front() {
            let response = match client.get(&url).send().await {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Warning: failed to fetch '{}': {}", url, e);
                    continue;
                }
            };

            if !response.status().is_success() {
                eprintln!("Warning: HTTP {} for '{}'", response.status(), url);
                continue;
            }

            let raw_html = match response.text().await {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Warning: failed to read body from '{}': {}", url, e);
                    continue;
                }
            };

            let doc = build_document_from_html(&raw_html, &url, self.css_selector.as_deref());
            documents.push(doc);

            // Extract and enqueue links if we haven't hit max depth
            if depth < self.max_depth {
                let links = Self::extract_same_domain_links(&raw_html, &url);
                for link in links {
                    if visited.insert(link.clone()) {
                        queue.push_back((link, depth + 1));
                    }
                }
            }
        }

        Ok(documents)
    }
}

#[async_trait]
impl BaseLoader for WebCrawler {
    async fn lazy_load(&self) -> Result<DocumentStream> {
        let docs = self.crawl().await?;
        Ok(Box::pin(stream::iter(docs.into_iter().map(Ok))))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- HTML text extraction --

    #[test]
    fn test_extract_text_from_html_basic() {
        let html = r#"<html><head><title>Test Page</title></head>
            <body><h1>Hello</h1><p>World &amp; friends</p></body></html>"#;
        let text = extract_text_from_html(html);
        assert!(text.contains("Hello"));
        assert!(text.contains("World & friends"));
    }

    // -- CSS selector extraction --

    #[test]
    fn test_extract_by_selector_tag() {
        let html = r#"<html><body>
            <header>Nav</header>
            <article><p>Article content</p></article>
            <footer>Foot</footer>
        </body></html>"#;
        let inner = extract_by_selector(html, "article").unwrap();
        assert!(inner.contains("Article content"));
    }

    #[test]
    fn test_extract_by_selector_class() {
        let html = r#"<div class="sidebar">Side</div>
            <div class="main-content">Main stuff</div>"#;
        let inner = extract_by_selector(html, ".main-content").unwrap();
        assert!(inner.contains("Main stuff"));
    }

    #[test]
    fn test_extract_by_selector_id() {
        let html = r#"<div id="header">Header</div>
            <div id="content">Body text</div>"#;
        let inner = extract_by_selector(html, "#content").unwrap();
        assert!(inner.contains("Body text"));
    }

    #[test]
    fn test_extract_by_selector_tag_class() {
        let html = r#"<span class="info">Info</span>
            <div class="info">Div info</div>"#;
        let inner = extract_by_selector(html, "div.info").unwrap();
        assert!(inner.contains("Div info"));
    }

    #[test]
    fn test_extract_by_selector_not_found() {
        let html = "<p>Just a paragraph</p>";
        assert!(extract_by_selector(html, "article").is_none());
    }

    // -- Metadata population --

    #[test]
    fn test_build_document_metadata() {
        let html = r#"<html><head><title>My Title</title></head>
            <body><p>Content</p></body></html>"#;
        let doc = build_document_from_html(html, "https://example.com/page", None);

        assert_eq!(
            doc.metadata.get("source").unwrap(),
            &Value::String("https://example.com/page".to_string())
        );
        assert_eq!(
            doc.metadata.get("title").unwrap(),
            &Value::String("My Title".to_string())
        );
        assert_eq!(
            doc.metadata.get("content_type").unwrap(),
            &Value::String("text/html".to_string())
        );
        assert!(doc.metadata.contains_key("fetch_timestamp"));
        assert!(doc.page_content.contains("Content"));
    }

    #[test]
    fn test_build_document_with_selector() {
        let html = r#"<html><head><title>Page</title></head>
            <body><nav>Menu</nav><article>Important text</article></body></html>"#;
        let doc = build_document_from_html(html, "https://example.com", Some("article"));
        assert_eq!(doc.page_content, "Important text");
        assert!(!doc.page_content.contains("Menu"));
    }

    // -- Multiple URLs / builder pattern --

    #[test]
    fn test_web_loader_builder() {
        let loader = WebLoader::new(vec![
            "https://example.com/a".to_string(),
            "https://example.com/b".to_string(),
        ])
        .with_timeout(10)
        .with_user_agent("test/1.0")
        .with_css_selector("main")
        .with_header("Authorization", "Bearer token");

        assert_eq!(loader.urls.len(), 2);
        assert_eq!(loader.timeout_secs, 10);
        assert_eq!(loader.user_agent, "test/1.0");
        assert_eq!(loader.css_selector.as_deref(), Some("main"));
        assert_eq!(loader.headers.get("Authorization").unwrap(), "Bearer token");
    }

    // -- Custom headers --

    #[test]
    fn test_web_loader_custom_headers() {
        let mut headers = HashMap::new();
        headers.insert("X-Custom".to_string(), "value1".to_string());
        headers.insert("Accept-Language".to_string(), "en-US".to_string());

        let loader = WebLoader::new(vec!["https://example.com".to_string()]).with_headers(headers);

        assert_eq!(loader.headers.len(), 2);
        assert_eq!(loader.headers.get("X-Custom").unwrap(), "value1");
    }

    // -- Link extraction --

    #[test]
    fn test_extract_links() {
        let html = r##"
            <a href="https://example.com/page1">Page 1</a>
            <a href="/page2">Page 2</a>
            <a href="page3">Page 3</a>
            <a href="javascript:void(0)">JS link</a>
            <a href="mailto:test@test.com">Email</a>
            <a href="#section">Anchor</a>
        "##;
        let links = extract_links(html);
        assert_eq!(links.len(), 3);
        assert!(links.contains(&"https://example.com/page1".to_string()));
        assert!(links.contains(&"/page2".to_string()));
        assert!(links.contains(&"page3".to_string()));
    }

    // -- URL resolution --

    #[test]
    fn test_resolve_url_absolute() {
        let resolved = resolve_url("https://example.com/", "https://other.com/page").unwrap();
        assert_eq!(resolved, "https://other.com/page");
    }

    #[test]
    fn test_resolve_url_root_relative() {
        let resolved = resolve_url("https://example.com/dir/page", "/about").unwrap();
        assert_eq!(resolved, "https://example.com/about");
    }

    #[test]
    fn test_resolve_url_relative() {
        let resolved = resolve_url("https://example.com/dir/page", "other").unwrap();
        assert_eq!(resolved, "https://example.com/dir/other");
    }

    #[test]
    fn test_resolve_url_protocol_relative() {
        let resolved = resolve_url("https://example.com/", "//cdn.example.com/file").unwrap();
        assert_eq!(resolved, "https://cdn.example.com/file");
    }

    #[test]
    fn test_resolve_url_skips_javascript() {
        assert!(resolve_url("https://example.com/", "javascript:void(0)").is_none());
    }

    #[test]
    fn test_resolve_url_skips_mailto() {
        assert!(resolve_url("https://example.com/", "mailto:a@b.com").is_none());
    }

    // -- WebCrawler link extraction / depth limiting --

    #[test]
    fn test_crawler_extract_same_domain_links() {
        let html = r#"
            <a href="https://example.com/about">About</a>
            <a href="https://other.com/external">External</a>
            <a href="/contact">Contact</a>
            <a href="sub/page">Relative</a>
        "#;
        let links = WebCrawler::extract_same_domain_links(html, "https://example.com/dir/page");
        assert!(links.contains(&"https://example.com/about".to_string()));
        assert!(links.contains(&"https://example.com/contact".to_string()));
        assert!(links.contains(&"https://example.com/dir/sub/page".to_string()));
        assert!(!links.iter().any(|l| l.contains("other.com")));
    }

    #[test]
    fn test_crawler_builder_and_depth() {
        let crawler = WebCrawler::new("https://example.com")
            .with_max_depth(3)
            .with_timeout(60)
            .with_user_agent("test-crawler/1.0")
            .with_css_selector("#content");

        assert_eq!(crawler.start_url, "https://example.com");
        assert_eq!(crawler.max_depth, 3);
        assert_eq!(crawler.timeout_secs, 60);
        assert_eq!(crawler.user_agent, "test-crawler/1.0");
        assert_eq!(crawler.css_selector.as_deref(), Some("#content"));
    }

    #[test]
    fn test_crawler_depth_zero() {
        let crawler = WebCrawler::new("https://example.com").with_max_depth(0);
        assert_eq!(crawler.max_depth, 0);
    }

    // -- Error handling --

    #[test]
    fn test_extract_domain() {
        assert_eq!(
            extract_domain("https://example.com/path"),
            Some("example.com".to_string())
        );
        assert_eq!(
            extract_domain("http://sub.example.com:8080/path"),
            Some("sub.example.com:8080".to_string())
        );
        assert_eq!(extract_domain("not-a-url"), None);
    }

    #[tokio::test]
    async fn test_web_loader_invalid_url_skips() {
        let loader = WebLoader::new(vec!["http://localhost:1/nonexistent".to_string()]);
        let docs = loader.load_async().await.unwrap();
        assert!(docs.is_empty());
    }

    #[test]
    fn test_web_base_loader_construction() {
        let loader = WebBaseLoader::new("https://example.com");
        assert_eq!(loader.url, "https://example.com");
    }

    #[test]
    fn test_web_base_loader_with_custom_client() {
        let client = Client::builder().user_agent("test-agent").build().unwrap();
        let loader = WebBaseLoader::new("https://example.com").with_client(client);
        assert_eq!(loader.url, "https://example.com");
    }

    #[tokio::test]
    async fn test_web_base_loader_invalid_url() {
        let loader = WebBaseLoader::new("http://localhost:1/nonexistent");
        let result = loader.load().await;
        assert!(result.is_err());
    }
}
