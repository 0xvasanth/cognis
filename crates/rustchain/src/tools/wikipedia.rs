//! Wikipedia summary lookup tool.
//!
//! Uses the Wikipedia REST API to fetch article summaries by title. Supports
//! configurable language editions.

use async_trait::async_trait;
use rustchain_core::error::{Result, RustChainError};
use rustchain_core::tools::base::BaseTool;
use rustchain_core::tools::types::{ToolInput, ToolOutput};
use serde::Deserialize;
use serde_json::{json, Value};

/// A tool that fetches Wikipedia article summaries.
///
/// # Builder
///
/// ```rust,ignore
/// let tool = WikipediaTool::builder()
///     .lang("de")
///     .num_sentences(5)
///     .build();
/// ```
pub struct WikipediaTool {
    /// Wikipedia language edition (default: `"en"`).
    lang: String,
    /// Number of sentences to include from the extract (default: 3).
    ///
    /// Note: The Wikipedia summary API returns a fixed extract; this field
    /// is used to truncate the result to approximately this many sentences.
    num_sentences: usize,
    /// Shared HTTP client.
    client: reqwest::Client,
}

/// Builder for [`WikipediaTool`].
pub struct WikipediaToolBuilder {
    lang: String,
    num_sentences: usize,
    client: Option<reqwest::Client>,
}

impl WikipediaToolBuilder {
    /// Set the Wikipedia language edition (default: `"en"`).
    pub fn lang(mut self, lang: impl Into<String>) -> Self {
        self.lang = lang.into();
        self
    }

    /// Set the maximum number of sentences to return (default: 3).
    pub fn num_sentences(mut self, n: usize) -> Self {
        self.num_sentences = n;
        self
    }

    /// Provide a custom [`reqwest::Client`].
    pub fn client(mut self, client: reqwest::Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Build the [`WikipediaTool`].
    pub fn build(self) -> WikipediaTool {
        WikipediaTool {
            lang: self.lang,
            num_sentences: self.num_sentences,
            client: self.client.unwrap_or_default(),
        }
    }
}

impl WikipediaTool {
    /// Create a new `WikipediaTool` with default settings.
    pub fn new() -> Self {
        Self::builder().build()
    }

    /// Create a new builder.
    pub fn builder() -> WikipediaToolBuilder {
        WikipediaToolBuilder {
            lang: "en".to_string(),
            num_sentences: 3,
            client: None,
        }
    }

    /// Build the summary API URL for the given title.
    pub(crate) fn build_url(&self, title: &str) -> String {
        let encoded_title = urlencoded(title);
        format!(
            "https://{}.wikipedia.org/api/rest_v1/page/summary/{}",
            self.lang, encoded_title
        )
    }
}

impl Default for WikipediaTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Minimal percent-encoding for URL path segments.
fn urlencoded(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            ' ' => "%20".to_string(),
            '#' => "%23".to_string(),
            '?' => "%3F".to_string(),
            '&' => "%26".to_string(),
            '%' => "%25".to_string(),
            '+' => "%2B".to_string(),
            _ if c.is_ascii_alphanumeric() || "-._~/:@!$'()*,;=".contains(c) => {
                c.to_string()
            }
            _ => {
                let mut buf = [0u8; 4];
                let encoded = c.encode_utf8(&mut buf);
                encoded.bytes().map(|b| format!("%{:02X}", b)).collect()
            }
        })
        .collect()
}

/// Truncate a text to approximately `n` sentences.
pub(crate) fn truncate_sentences(text: &str, n: usize) -> String {
    if n == 0 {
        return String::new();
    }
    let mut count = 0;
    let mut end = 0;
    for (i, c) in text.char_indices() {
        if c == '.' || c == '!' || c == '?' {
            count += 1;
            end = i + c.len_utf8();
            if count >= n {
                break;
            }
        }
    }
    if end == 0 || count < n {
        // Fewer sentences than requested; return the whole text.
        text.to_string()
    } else {
        text[..end].to_string()
    }
}

/// Relevant fields from the Wikipedia summary API response.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct WikiSummary {
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub extract: String,
    #[serde(default)]
    pub description: String,
    #[serde(default, rename = "type")]
    pub page_type: String,
}

/// Format a Wikipedia summary response into readable text.
pub(crate) fn format_wiki_response(summary: &WikiSummary, num_sentences: usize) -> String {
    if summary.extract.is_empty() {
        return format!(
            "No Wikipedia article found for \"{}\". Try a different search term.",
            summary.title
        );
    }

    let extract = truncate_sentences(&summary.extract, num_sentences);
    let mut parts = Vec::new();

    if !summary.title.is_empty() {
        parts.push(format!("# {}", summary.title));
    }
    if !summary.description.is_empty() {
        parts.push(summary.description.clone());
    }
    parts.push(String::new()); // blank line
    parts.push(extract);

    parts.join("\n")
}

#[async_trait]
impl BaseTool for WikipediaTool {
    fn name(&self) -> &str {
        "wikipedia"
    }

    fn description(&self) -> &str {
        "Look up information on Wikipedia. Input should be a topic or article title."
    }

    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The topic or article title to look up"
                }
            },
            "required": ["query"]
        }))
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let query = extract_query(&input)?;
        let url = self.build_url(&query);

        let resp = self
            .client
            .get(&url)
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(|e| {
                RustChainError::ToolException(format!("Wikipedia request failed: {e}"))
            })?;

        if resp.status().as_u16() == 404 {
            return Ok(ToolOutput::Content(Value::String(format!(
                "No Wikipedia article found for \"{query}\". Try a different search term."
            ))));
        }

        if !resp.status().is_success() {
            return Err(RustChainError::ToolException(format!(
                "Wikipedia returned status {}",
                resp.status()
            )));
        }

        let summary: WikiSummary = resp.json().await.map_err(|e| {
            RustChainError::ToolException(format!("Failed to parse Wikipedia response: {e}"))
        })?;

        let formatted = format_wiki_response(&summary, self.num_sentences);
        Ok(ToolOutput::Content(Value::String(formatted)))
    }
}

/// Extract a query string from various input formats.
fn extract_query(input: &ToolInput) -> Result<String> {
    match input {
        ToolInput::Text(s) => Ok(s.clone()),
        ToolInput::Structured(map) => {
            if let Some(Value::String(q)) = map.get("query") {
                Ok(q.clone())
            } else {
                Err(RustChainError::ToolValidationError(
                    "Missing required field 'query'".into(),
                ))
            }
        }
        ToolInput::ToolCall(tc) => {
            if let Some(Value::String(q)) = tc.args.get("query") {
                Ok(q.clone())
            } else {
                Err(RustChainError::ToolValidationError(
                    "Missing required field 'query'".into(),
                ))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wikipedia_builder_defaults() {
        let tool = WikipediaTool::new();
        assert_eq!(tool.name(), "wikipedia");
        assert_eq!(tool.lang, "en");
        assert_eq!(tool.num_sentences, 3);
    }

    #[test]
    fn test_wikipedia_builder_custom() {
        let tool = WikipediaTool::builder()
            .lang("de")
            .num_sentences(5)
            .build();
        assert_eq!(tool.lang, "de");
        assert_eq!(tool.num_sentences, 5);
    }

    #[test]
    fn test_wikipedia_url_construction() {
        let tool = WikipediaTool::new();
        let url = tool.build_url("Rust (programming language)");
        assert_eq!(
            url,
            "https://en.wikipedia.org/api/rest_v1/page/summary/Rust%20(programming%20language)"
        );
    }

    #[test]
    fn test_wikipedia_url_encoding_special_chars() {
        let tool = WikipediaTool::builder().lang("fr").build();
        let url = tool.build_url("C++ language");
        assert_eq!(
            url,
            "https://fr.wikipedia.org/api/rest_v1/page/summary/C%2B%2B%20language"
        );
    }

    #[test]
    fn test_wikipedia_response_parsing() {
        let json_str = r#"{
            "title": "Rust (programming language)",
            "extract": "Rust is a general-purpose programming language. It was designed by Graydon Hoare. It emphasizes performance and safety.",
            "description": "Programming language",
            "type": "standard"
        }"#;

        let summary: WikiSummary = serde_json::from_str(json_str).unwrap();
        assert_eq!(summary.title, "Rust (programming language)");
        assert!(!summary.extract.is_empty());

        let formatted = format_wiki_response(&summary, 2);
        assert!(formatted.contains("# Rust (programming language)"));
        assert!(formatted.contains("Programming language"));
        // Should have only 2 sentences
        assert!(formatted.contains("It was designed by Graydon Hoare."));
        assert!(!formatted.contains("It emphasizes performance and safety."));
    }

    #[test]
    fn test_wikipedia_empty_extract() {
        let summary = WikiSummary {
            title: "Nonexistent Page".to_string(),
            extract: String::new(),
            description: String::new(),
            page_type: String::new(),
        };
        let formatted = format_wiki_response(&summary, 3);
        assert!(formatted.contains("No Wikipedia article found"));
    }

    #[test]
    fn test_wikipedia_args_schema() {
        let tool = WikipediaTool::new();
        let schema = tool.args_schema().unwrap();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["query"]["type"], "string");
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&Value::String("query".to_string())));
    }

    #[test]
    fn test_truncate_sentences() {
        let text = "First sentence. Second sentence. Third sentence. Fourth sentence.";
        assert_eq!(truncate_sentences(text, 2), "First sentence. Second sentence.");
        assert_eq!(truncate_sentences(text, 4), text);
        assert_eq!(truncate_sentences(text, 10), text);
        assert_eq!(truncate_sentences(text, 0), "");
    }

    #[test]
    fn test_extract_query_from_text() {
        let input = ToolInput::Text("Rust language".to_string());
        assert_eq!(extract_query(&input).unwrap(), "Rust language");
    }

    #[test]
    fn test_extract_query_from_structured() {
        let mut map = std::collections::HashMap::new();
        map.insert("query".to_string(), Value::String("test topic".to_string()));
        let input = ToolInput::Structured(map);
        assert_eq!(extract_query(&input).unwrap(), "test topic");
    }
}
