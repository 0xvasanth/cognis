//! JSON query tool using dot-notation paths.
//!
//! Supports dot notation (`a.b.c`), array indexing (`items[0]`), and nested
//! combinations (`data.users[2].name`).

use async_trait::async_trait;
use rustchain_core::error::{Result, RustChainError};
use rustchain_core::tools::base::BaseTool;
use rustchain_core::tools::types::{ToolInput, ToolOutput};
use serde_json::{json, Value};

/// Extract values from JSON data using dot-notation paths.
pub struct JsonQueryTool;

#[async_trait]
impl BaseTool for JsonQueryTool {
    fn name(&self) -> &str {
        "json_query"
    }

    fn description(&self) -> &str {
        "Extract values from JSON data using dot-notation paths like 'data.users[0].name'."
    }

    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "json_data": {
                    "type": "string",
                    "description": "JSON string to query"
                },
                "path": {
                    "type": "string",
                    "description": "Dot-notation path to extract (e.g. 'data.users[0].name')"
                }
            },
            "required": ["json_data", "path"]
        }))
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let (json_data, path) = extract_args(&input)?;

        let value: Value = serde_json::from_str(&json_data).map_err(|e| {
            RustChainError::ToolException(format!("Invalid JSON: {e}"))
        })?;

        let result = query_path(&value, &path).map_err(|e| {
            RustChainError::ToolException(format!("Path query failed: {e}"))
        })?;

        Ok(ToolOutput::Content(result.clone()))
    }
}

/// Extract json_data and path from various input formats.
fn extract_args(input: &ToolInput) -> Result<(String, String)> {
    match input {
        ToolInput::Structured(map) => {
            let json_data = match map.get("json_data") {
                Some(Value::String(s)) => s.clone(),
                _ => {
                    return Err(RustChainError::ToolValidationError(
                        "Missing required field 'json_data'".into(),
                    ))
                }
            };
            let path = match map.get("path") {
                Some(Value::String(s)) => s.clone(),
                _ => {
                    return Err(RustChainError::ToolValidationError(
                        "Missing required field 'path'".into(),
                    ))
                }
            };
            Ok((json_data, path))
        }
        ToolInput::ToolCall(tc) => {
            let json_data = match tc.args.get("json_data") {
                Some(Value::String(s)) => s.clone(),
                _ => {
                    return Err(RustChainError::ToolValidationError(
                        "Missing required field 'json_data'".into(),
                    ))
                }
            };
            let path = match tc.args.get("path") {
                Some(Value::String(s)) => s.clone(),
                _ => {
                    return Err(RustChainError::ToolValidationError(
                        "Missing required field 'path'".into(),
                    ))
                }
            };
            Ok((json_data, path))
        }
        ToolInput::Text(_) => Err(RustChainError::ToolValidationError(
            "JsonQueryTool requires structured input with 'json_data' and 'path' fields".into(),
        )),
    }
}

/// A single path segment: either a key or an array index.
#[derive(Debug)]
enum PathSegment {
    Key(String),
    Index(usize),
}

/// Parse a dot-notation path into segments.
///
/// Examples:
///   "name"               -> [Key("name")]
///   "a.b.c"              -> [Key("a"), Key("b"), Key("c")]
///   "items[0]"           -> [Key("items"), Index(0)]
///   "data.users[1].name" -> [Key("data"), Key("users"), Index(1), Key("name")]
fn parse_path(path: &str) -> std::result::Result<Vec<PathSegment>, String> {
    let mut segments = Vec::new();

    for part in path.split('.') {
        if part.is_empty() {
            return Err("Empty path segment".into());
        }

        // Check for array indexing: key[0] or just [0]
        if let Some(bracket_pos) = part.find('[') {
            let key = &part[..bracket_pos];
            if !key.is_empty() {
                segments.push(PathSegment::Key(key.to_string()));
            }

            // Parse all [N] indices in this part
            let mut rest = &part[bracket_pos..];
            while rest.starts_with('[') {
                let end = rest.find(']').ok_or("Missing closing bracket")?;
                let idx_str = &rest[1..end];
                let idx: usize = idx_str
                    .parse()
                    .map_err(|_| format!("Invalid array index: {idx_str}"))?;
                segments.push(PathSegment::Index(idx));
                rest = &rest[end + 1..];
            }
            if !rest.is_empty() {
                return Err(format!("Unexpected characters after bracket: {rest}"));
            }
        } else {
            segments.push(PathSegment::Key(part.to_string()));
        }
    }

    Ok(segments)
}

/// Traverse a JSON value using parsed path segments.
fn query_path<'a>(value: &'a Value, path: &str) -> std::result::Result<&'a Value, String> {
    let segments = parse_path(path)?;
    let mut current = value;

    for segment in &segments {
        match segment {
            PathSegment::Key(key) => {
                current = current
                    .get(key.as_str())
                    .ok_or_else(|| format!("Key '{key}' not found"))?;
            }
            PathSegment::Index(idx) => {
                current = current
                    .get(*idx)
                    .ok_or_else(|| format!("Index {idx} out of bounds"))?;
            }
        }
    }

    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_json_query_simple() {
        let tool = JsonQueryTool;
        let input = ToolInput::Structured(
            [
                (
                    "json_data".to_string(),
                    Value::String(r#"{"name": "test"}"#.to_string()),
                ),
                ("path".to_string(), Value::String("name".to_string())),
            ]
            .into_iter()
            .collect(),
        );
        let result = tool._run(input).await.unwrap();
        match result {
            ToolOutput::Content(v) => assert_eq!(v, Value::String("test".to_string())),
            other => panic!("Expected Content, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_json_query_nested() {
        let tool = JsonQueryTool;
        let input = ToolInput::Structured(
            [
                (
                    "json_data".to_string(),
                    Value::String(r#"{"a": {"b": {"c": 42}}}"#.to_string()),
                ),
                ("path".to_string(), Value::String("a.b.c".to_string())),
            ]
            .into_iter()
            .collect(),
        );
        let result = tool._run(input).await.unwrap();
        match result {
            ToolOutput::Content(v) => assert_eq!(v, json!(42)),
            other => panic!("Expected Content, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_json_query_array() {
        let tool = JsonQueryTool;
        let input = ToolInput::Structured(
            [
                (
                    "json_data".to_string(),
                    Value::String(r#"{"items": ["a", "b", "c"]}"#.to_string()),
                ),
                ("path".to_string(), Value::String("items[0]".to_string())),
            ]
            .into_iter()
            .collect(),
        );
        let result = tool._run(input).await.unwrap();
        match result {
            ToolOutput::Content(v) => assert_eq!(v, Value::String("a".to_string())),
            other => panic!("Expected Content, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_json_query_complex() {
        let tool = JsonQueryTool;
        let json_data = r#"{
            "data": {
                "users": [
                    {"name": "Alice"},
                    {"name": "Bob"},
                    {"name": "Charlie"}
                ]
            }
        }"#;
        let input = ToolInput::Structured(
            [
                (
                    "json_data".to_string(),
                    Value::String(json_data.to_string()),
                ),
                (
                    "path".to_string(),
                    Value::String("data.users[1].name".to_string()),
                ),
            ]
            .into_iter()
            .collect(),
        );
        let result = tool._run(input).await.unwrap();
        match result {
            ToolOutput::Content(v) => assert_eq!(v, Value::String("Bob".to_string())),
            other => panic!("Expected Content, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_json_query_via_run_json() {
        let tool = JsonQueryTool;
        let input = serde_json::json!({
            "json_data": r#"{"x": [10, 20, 30]}"#,
            "path": "x[2]"
        });
        let result = tool.run_json(&input).await.unwrap();
        assert_eq!(result, json!(30));
    }
}
