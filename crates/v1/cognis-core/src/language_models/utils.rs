//! Language model utility functions.
//!
//! Mirrors Python `langchain_core.language_models._utils`.

use serde_json::Value;

/// Check if a content block matches the OpenAI data block format.
/// OpenAI blocks have `{"type": "...", ...}` format.
pub fn is_openai_data_block(block: &Value) -> bool {
    block.is_object() && block.get("type").and_then(|t| t.as_str()).is_some()
}

/// Parse a data URI into its components: (mimetype, data).
/// Format: `data:<mimetype>;base64,<data>`
pub fn parse_data_uri(uri: &str) -> Option<(String, String)> {
    let stripped = uri.strip_prefix("data:")?;
    let (meta, data) = stripped.split_once(',')?;
    let mimetype = meta.strip_suffix(";base64").unwrap_or(meta);
    Some((mimetype.to_string(), data.to_string()))
}

/// Build a data URI from mimetype and base64 data.
pub fn build_data_uri(mimetype: &str, data: &str) -> String {
    format!("data:{};base64,{}", mimetype, data)
}

/// Extract the text content from a message content value.
/// Handles both string content and array-of-blocks content.
pub fn extract_text_from_content(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|block| {
                if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                    block
                        .get("text")
                        .and_then(|t| t.as_str())
                        .map(|s| s.to_string())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// Estimate token count for a string (rough approximation: ~4 chars per token).
pub fn estimate_token_count(text: &str) -> usize {
    text.len().div_ceil(4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_is_openai_data_block_valid() {
        let block = json!({"type": "text", "text": "hello"});
        assert!(is_openai_data_block(&block));
    }

    #[test]
    fn test_is_openai_data_block_missing_type() {
        let block = json!({"text": "hello"});
        assert!(!is_openai_data_block(&block));
    }

    #[test]
    fn test_is_openai_data_block_not_object() {
        assert!(!is_openai_data_block(&json!("string")));
        assert!(!is_openai_data_block(&json!(42)));
        assert!(!is_openai_data_block(&json!(null)));
    }

    #[test]
    fn test_is_openai_data_block_type_not_string() {
        let block = json!({"type": 123});
        assert!(!is_openai_data_block(&block));
    }

    #[test]
    fn test_parse_data_uri_base64() {
        let uri = "data:image/png;base64,iVBORw0KGgo=";
        let result = parse_data_uri(uri);
        assert_eq!(
            result,
            Some(("image/png".to_string(), "iVBORw0KGgo=".to_string()))
        );
    }

    #[test]
    fn test_parse_data_uri_no_base64() {
        let uri = "data:text/plain,Hello%20World";
        let result = parse_data_uri(uri);
        assert_eq!(
            result,
            Some(("text/plain".to_string(), "Hello%20World".to_string()))
        );
    }

    #[test]
    fn test_parse_data_uri_invalid() {
        assert_eq!(parse_data_uri("not-a-data-uri"), None);
        assert_eq!(parse_data_uri("data:no-comma"), None);
    }

    #[test]
    fn test_build_data_uri() {
        let result = build_data_uri("image/png", "iVBORw0KGgo=");
        assert_eq!(result, "data:image/png;base64,iVBORw0KGgo=");
    }

    #[test]
    fn test_extract_text_from_string_content() {
        let content = json!("Hello, world!");
        assert_eq!(extract_text_from_content(&content), "Hello, world!");
    }

    #[test]
    fn test_extract_text_from_array_content() {
        let content = json!([
            {"type": "text", "text": "Hello, "},
            {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
            {"type": "text", "text": "world!"}
        ]);
        assert_eq!(extract_text_from_content(&content), "Hello, world!");
    }

    #[test]
    fn test_extract_text_from_empty_array() {
        let content = json!([]);
        assert_eq!(extract_text_from_content(&content), "");
    }

    #[test]
    fn test_extract_text_from_non_text_content() {
        assert_eq!(extract_text_from_content(&json!(null)), "");
        assert_eq!(extract_text_from_content(&json!(42)), "");
    }

    #[test]
    fn test_estimate_token_count() {
        assert_eq!(estimate_token_count(""), 0);
        assert_eq!(estimate_token_count("a"), 1);
        assert_eq!(estimate_token_count("abcd"), 1);
        assert_eq!(estimate_token_count("abcde"), 2);
        assert_eq!(estimate_token_count("Hello, world!"), 4); // 13 chars -> (13+3)/4 = 4
    }
}
