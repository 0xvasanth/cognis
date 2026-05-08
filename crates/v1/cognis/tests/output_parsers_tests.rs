//! Tests for output fixing, retry, and structured output parsers.

use std::sync::Arc;

use serde_json::json;

use cognis::output_parsers::{OutputFixingParser, RetryOutputParser, StructuredOutputParser};
use cognis_core::language_models::FakeListChatModel;
use cognis_core::output_parsers::{JsonOutputParser, OutputParser};
use cognis_core::runnables::base::Runnable;

// ─── OutputFixingParser ───

#[tokio::test]
async fn test_output_fixing_parser_fixes_malformed_json() {
    // The inner parser will fail on the trailing comma.
    // The FakeListChatModel returns corrected JSON.
    let corrected = r#"{"name": "Alice", "age": 30}"#;
    let llm = Arc::new(FakeListChatModel::new(vec![corrected.to_string()]));
    let parser = OutputFixingParser::builder()
        .parser(JsonOutputParser::new())
        .llm(llm)
        .build();

    let malformed = r#"{"name": "Alice", "age": 30,}"#;
    let result = parser
        .invoke(serde_json::Value::String(malformed.to_string()), None)
        .await
        .unwrap();

    assert_eq!(result, json!({"name": "Alice", "age": 30}));
}

#[tokio::test]
async fn test_output_fixing_parser_passes_through_valid_output() {
    // LLM should not be called for valid JSON -- the parser succeeds on the
    // first attempt. We set up an LLM that would return garbage to confirm
    // it isn't invoked.
    let llm = Arc::new(FakeListChatModel::new(vec!["GARBAGE".to_string()]));
    let parser = OutputFixingParser::builder()
        .parser(JsonOutputParser::new())
        .llm(llm)
        .build();

    let valid = r#"{"key": "value"}"#;
    let result = parser
        .invoke(serde_json::Value::String(valid.to_string()), None)
        .await
        .unwrap();

    assert_eq!(result, json!({"key": "value"}));
}

#[tokio::test]
async fn test_output_fixing_parser_when_fix_also_fails() {
    // LLM returns something that also fails to parse.
    let llm = Arc::new(FakeListChatModel::new(vec!["still broken {{{".to_string()]));
    let parser = OutputFixingParser::builder()
        .parser(JsonOutputParser::new())
        .llm(llm)
        .build();

    let malformed = r#"not json at all"#;
    let result = parser
        .invoke(serde_json::Value::String(malformed.to_string()), None)
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_output_fixing_parser_new_constructor() {
    let corrected = r#"{"ok": true}"#;
    let llm = Arc::new(FakeListChatModel::new(vec![corrected.to_string()]));
    let parser = OutputFixingParser::new(JsonOutputParser::new(), llm);

    let result = parser
        .invoke(serde_json::Value::String("{bad".to_string()), None)
        .await
        .unwrap();

    assert_eq!(result, json!({"ok": true}));
}

// ─── RetryOutputParser ───

#[tokio::test]
async fn test_retry_parser_succeeds_on_retry() {
    // First parse fails, LLM returns corrected JSON on retry.
    let corrected = r#"{"status": "ok"}"#;
    let llm = Arc::new(FakeListChatModel::new(vec![corrected.to_string()]));
    let parser = RetryOutputParser::builder()
        .parser(JsonOutputParser::new())
        .llm(llm)
        .max_retries(3)
        .build();

    let malformed = r#"{"status": "ok",}"#;
    let result = parser
        .invoke(serde_json::Value::String(malformed.to_string()), None)
        .await
        .unwrap();

    assert_eq!(result, json!({"status": "ok"}));
}

#[tokio::test]
async fn test_retry_parser_exhausts_retries() {
    // LLM always returns bad output.
    let llm = Arc::new(FakeListChatModel::new(vec![
        "still bad".to_string(),
        "nope".to_string(),
        "also bad".to_string(),
    ]));
    let parser = RetryOutputParser::builder()
        .parser(JsonOutputParser::new())
        .llm(llm)
        .max_retries(3)
        .build();

    let result = parser
        .invoke(serde_json::Value::String("not json".to_string()), None)
        .await;

    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("Failed to parse after 3 retries"));
}

#[tokio::test]
async fn test_retry_parser_with_max_retries_zero() {
    // With max_retries=0, should fail immediately on parse error.
    let llm = Arc::new(FakeListChatModel::new(vec![
        r#"{"fixed": true}"#.to_string()
    ]));
    let parser = RetryOutputParser::builder()
        .parser(JsonOutputParser::new())
        .llm(llm)
        .max_retries(0)
        .build();

    let result = parser
        .invoke(serde_json::Value::String("not json".to_string()), None)
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_retry_parser_valid_input_no_retry() {
    // Valid JSON on first attempt -- LLM should not be called.
    let llm = Arc::new(FakeListChatModel::new(vec!["GARBAGE".to_string()]));
    let parser = RetryOutputParser::builder()
        .parser(JsonOutputParser::new())
        .llm(llm)
        .max_retries(3)
        .build();

    let valid = r#"[1, 2, 3]"#;
    let result = parser
        .invoke(serde_json::Value::String(valid.to_string()), None)
        .await
        .unwrap();

    assert_eq!(result, json!([1, 2, 3]));
}

// ─── StructuredOutputParser ───

#[tokio::test]
async fn test_structured_parser_valid_json() {
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    });
    let parser = StructuredOutputParser::new("Person", schema);

    let result = parser.parse(r#"{"name": "Alice", "age": 30}"#).unwrap();
    assert_eq!(result, json!({"name": "Alice", "age": 30}));
}

#[tokio::test]
async fn test_structured_parser_invalid_json() {
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        },
        "required": ["name"]
    });
    let parser = StructuredOutputParser::new("Person", schema);

    let result = parser.parse("not json at all");
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("Failed to parse JSON for Person")
            || err_msg.contains("No JSON found in Person"),
        "unexpected error: {err_msg}"
    );
}

#[tokio::test]
async fn test_structured_parser_format_instructions() {
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    });
    let parser = StructuredOutputParser::new("Person", schema);
    let instructions = parser.get_format_instructions().unwrap();

    assert!(instructions.contains("JSON schema"));
    assert!(instructions.contains("name"));
    assert!(instructions.contains("age"));
}

#[tokio::test]
async fn test_structured_parser_missing_required_fields() {
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string"}
        },
        "required": ["name", "age", "email"]
    });
    let parser = StructuredOutputParser::new("User", schema);

    // Missing "age" and "email"
    let result = parser.parse(r#"{"name": "Bob"}"#);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("Missing required field(s)"));
    assert!(err_msg.contains("age"));
    assert!(err_msg.contains("email"));
}

#[tokio::test]
async fn test_structured_parser_type_validation() {
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    });
    let parser = StructuredOutputParser::new("Person", schema);

    // age is a string instead of integer
    let result = parser.parse(r#"{"name": "Alice", "age": "thirty"}"#);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("Type validation errors"));
    assert!(err_msg.contains("age"));
    assert!(err_msg.contains("expected integer"));
}

#[tokio::test]
async fn test_structured_parser_builder_pattern() {
    let parser = StructuredOutputParser::builder()
        .type_name("Config")
        .schema(json!({
            "type": "object",
            "properties": {
                "debug": {"type": "boolean"}
            },
            "required": ["debug"]
        }))
        .build();

    assert_eq!(parser.parser_type(), "structured_output_parser");

    let result = parser.parse(r#"{"debug": true}"#).unwrap();
    assert_eq!(result, json!({"debug": true}));
}

#[tokio::test]
async fn test_structured_parser_strips_markdown_fences() {
    let schema = json!({
        "type": "object",
        "properties": {
            "value": {"type": "number"}
        },
        "required": ["value"]
    });
    let parser = StructuredOutputParser::new("Result", schema);

    let input = "```json\n{\"value\": 42}\n```";
    let result = parser.parse(input).unwrap();
    assert_eq!(result, json!({"value": 42}));
}

#[tokio::test]
async fn test_structured_parser_non_object_when_object_expected() {
    let schema = json!({
        "type": "object",
        "properties": {
            "x": {"type": "number"}
        }
    });
    let parser = StructuredOutputParser::new("Point", schema);

    let result = parser.parse("[1, 2, 3]");
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("Expected JSON object"));
}

#[tokio::test]
async fn test_structured_parser_runnable_invoke() {
    let parser = StructuredOutputParser::new(
        "Item",
        json!({
            "type": "object",
            "properties": {
                "id": {"type": "integer"}
            },
            "required": ["id"]
        }),
    );

    let result = parser
        .invoke(serde_json::Value::String(r#"{"id": 1}"#.to_string()), None)
        .await
        .unwrap();

    assert_eq!(result, json!({"id": 1}));
}
