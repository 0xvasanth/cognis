use std::collections::HashMap;

use serde_json::{json, Value};

use cognis_core::messages::{AIMessage, ToolCall};
use cognis_core::output_parsers::*;
use cognis_core::outputs::{ChatGeneration, Generation};
use cognis_core::runnables::Runnable;

// ─── StrOutputParser Tests ───

#[test]
fn test_str_parser_parse() {
    let parser = StrOutputParser;
    let result = parser.parse("Hello world").unwrap();
    assert_eq!(result, json!("Hello world"));
}

#[test]
fn test_str_parser_type() {
    let parser = StrOutputParser;
    assert_eq!(parser.parser_type(), "str_output_parser");
}

#[tokio::test]
async fn test_str_parser_runnable() {
    let parser = StrOutputParser;
    let result = parser.invoke(json!("test input"), None).await.unwrap();
    assert_eq!(result, json!("test input"));
}

#[tokio::test]
async fn test_str_parser_runnable_non_string() {
    let parser = StrOutputParser;
    let result = parser.invoke(json!(42), None).await.unwrap();
    assert_eq!(result, json!("42"));
}

// ─── JsonOutputParser Tests ───

#[test]
fn test_json_parser_basic() {
    let parser = JsonOutputParser::new();
    let result = parser.parse(r#"{"key": "value"}"#).unwrap();
    assert_eq!(result, json!({"key": "value"}));
}

#[test]
fn test_json_parser_with_fences() {
    let parser = JsonOutputParser::new();
    let input = "```json\n{\"key\": \"value\"}\n```";
    let result = parser.parse(input).unwrap();
    assert_eq!(result, json!({"key": "value"}));
}

#[test]
fn test_json_parser_with_json_fence() {
    let parser = JsonOutputParser::new();
    let input = "```json\n[1, 2, 3]\n```";
    let result = parser.parse(input).unwrap();
    assert_eq!(result, json!([1, 2, 3]));
}

#[test]
fn test_json_parser_plain_fence() {
    let parser = JsonOutputParser::new();
    let input = "```\n{\"a\": 1}\n```";
    let result = parser.parse(input).unwrap();
    assert_eq!(result, json!({"a": 1}));
}

#[test]
fn test_json_parser_invalid_json() {
    let parser = JsonOutputParser::new();
    let result = parser.parse("not json at all");
    assert!(result.is_err());
}

#[test]
fn test_json_parser_partial_mode() {
    let parser = JsonOutputParser::new();
    let gens = vec![Generation::new("invalid json")];
    let result = parser.parse_result(&gens, true).unwrap();
    assert_eq!(result, Value::Null); // Partial mode returns Null on failure
}

#[test]
fn test_json_parser_format_instructions_no_schema() {
    let parser = JsonOutputParser::new();
    let instructions = parser.get_format_instructions().unwrap();
    assert!(instructions.contains("JSON"));
}

#[test]
fn test_json_parser_format_instructions_with_schema() {
    let parser = JsonOutputParser::with_schema(json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }
    }));
    let instructions = parser.get_format_instructions().unwrap();
    assert!(instructions.contains("name"));
    assert!(instructions.contains("age"));
}

#[tokio::test]
async fn test_json_parser_runnable() {
    let parser = JsonOutputParser::new();
    let result = parser
        .invoke(json!(r#"{"result": 42}"#), None)
        .await
        .unwrap();
    assert_eq!(result, json!({"result": 42}));
}

// ─── CommaSeparatedListOutputParser Tests ───

#[test]
fn test_csv_parser_basic() {
    let parser = CommaSeparatedListOutputParser;
    let result = parser.parse("foo, bar, baz").unwrap();
    assert_eq!(result, json!(["foo", "bar", "baz"]));
}

#[test]
fn test_csv_parser_single_item() {
    let parser = CommaSeparatedListOutputParser;
    let result = parser.parse("single").unwrap();
    assert_eq!(result, json!(["single"]));
}

#[test]
fn test_csv_parser_whitespace() {
    let parser = CommaSeparatedListOutputParser;
    let result = parser.parse("  a ,  b  , c  ").unwrap();
    assert_eq!(result, json!(["a", "b", "c"]));
}

#[test]
fn test_csv_parser_format_instructions() {
    let parser = CommaSeparatedListOutputParser;
    let instructions = parser.get_format_instructions().unwrap();
    assert!(instructions.contains("comma separated"));
}

#[tokio::test]
async fn test_csv_parser_runnable() {
    let parser = CommaSeparatedListOutputParser;
    let result = parser.invoke(json!("a, b, c"), None).await.unwrap();
    assert_eq!(result, json!(["a", "b", "c"]));
}

// ─── NumberedListOutputParser Tests ───

#[test]
fn test_numbered_list_parser() {
    let parser = NumberedListOutputParser;
    let result = parser.parse("1. First\n2. Second\n3. Third").unwrap();
    assert_eq!(result, json!(["First", "Second", "Third"]));
}

#[test]
fn test_numbered_list_parser_with_extra_text() {
    let parser = NumberedListOutputParser;
    let result = parser
        .parse("Here are items:\n1. Alpha\n2. Beta\nDone.")
        .unwrap();
    assert_eq!(result, json!(["Alpha", "Beta"]));
}

#[test]
fn test_numbered_list_format_instructions() {
    let parser = NumberedListOutputParser;
    let instructions = parser.get_format_instructions().unwrap();
    assert!(instructions.contains("numbered"));
}

// ─── MarkdownListOutputParser Tests ───

#[test]
fn test_markdown_list_parser_dash() {
    let parser = MarkdownListOutputParser;
    let result = parser.parse("- foo\n- bar\n- baz").unwrap();
    assert_eq!(result, json!(["foo", "bar", "baz"]));
}

#[test]
fn test_markdown_list_parser_asterisk() {
    let parser = MarkdownListOutputParser;
    let result = parser.parse("* alpha\n* beta").unwrap();
    assert_eq!(result, json!(["alpha", "beta"]));
}

#[test]
fn test_markdown_list_parser_mixed_content() {
    let parser = MarkdownListOutputParser;
    let result = parser
        .parse("Some text\n- item1\nMore text\n- item2")
        .unwrap();
    assert_eq!(result, json!(["item1", "item2"]));
}

#[test]
fn test_markdown_list_format_instructions() {
    let parser = MarkdownListOutputParser;
    let instructions = parser.get_format_instructions().unwrap();
    assert!(instructions.contains("markdown"));
}

// ─── ToolCallOutputParser Tests ───

#[test]
fn test_tool_call_parser_from_chat_generation() {
    let mut ai_msg = AIMessage::new("I'll use the calculator");
    ai_msg.tool_calls = vec![ToolCall {
        name: "calculator".into(),
        args: {
            let mut m = HashMap::new();
            m.insert("expression".into(), json!("2+2"));
            m
        },
        id: Some("call_123".into()),
    }];
    let gen = ChatGeneration::new(ai_msg);
    let parser = ToolCallOutputParser::new();
    let result = parser.parse_chat_generation(&gen).unwrap();

    let arr = result.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["type"], "calculator");
    assert_eq!(arr[0]["args"]["expression"], "2+2");
    assert_eq!(arr[0]["id"], "call_123");
}

#[test]
fn test_tool_call_parser_first_only() {
    let mut ai_msg = AIMessage::new("");
    ai_msg.tool_calls = vec![
        ToolCall {
            name: "tool_a".into(),
            args: HashMap::new(),
            id: Some("1".into()),
        },
        ToolCall {
            name: "tool_b".into(),
            args: HashMap::new(),
            id: Some("2".into()),
        },
    ];
    let gen = ChatGeneration::new(ai_msg);
    let parser = ToolCallOutputParser::new().first_only();
    let result = parser.parse_chat_generation(&gen).unwrap();
    assert_eq!(result["type"], "tool_a");
}

#[test]
fn test_tool_call_parser_without_id() {
    let mut ai_msg = AIMessage::new("");
    ai_msg.tool_calls = vec![ToolCall {
        name: "search".into(),
        args: HashMap::new(),
        id: Some("x".into()),
    }];
    let gen = ChatGeneration::new(ai_msg);
    let parser = ToolCallOutputParser::new().without_id();
    let result = parser.parse_chat_generation(&gen).unwrap();
    assert!(result[0].get("id").is_none());
}

#[test]
fn test_tool_call_parser_empty_tool_calls() {
    let ai_msg = AIMessage::new("No tools used");
    let gen = ChatGeneration::new(ai_msg);
    let parser = ToolCallOutputParser::new();
    let result = parser.parse_chat_generation(&gen).unwrap();
    assert_eq!(result, json!([]));
}

#[test]
fn test_tool_call_parser_first_only_empty_errors() {
    let ai_msg = AIMessage::new("No tools");
    let gen = ChatGeneration::new(ai_msg);
    let parser = ToolCallOutputParser::new().first_only();
    assert!(parser.parse_chat_generation(&gen).is_err());
}

// ─── XmlOutputParser Tests ───

#[test]
fn test_xml_parser_simple() {
    let parser = XmlOutputParser::new();
    let result = parser.parse("<answer>42</answer>").unwrap();
    assert_eq!(result, json!({"answer": "42"}));
}

#[test]
fn test_xml_parser_nested() {
    let parser = XmlOutputParser::new();
    let result = parser
        .parse("<root><name>Alice</name><age>30</age></root>")
        .unwrap();
    assert_eq!(result, json!({"root": {"name": "Alice", "age": "30"}}));
}

#[test]
fn test_xml_parser_repeated_tags() {
    let parser = XmlOutputParser::new();
    let result = parser
        .parse("<items><item>a</item><item>b</item></items>")
        .unwrap();
    assert_eq!(result, json!({"items": {"item": ["a", "b"]}}));
}

#[test]
fn test_xml_parser_with_fences() {
    let parser = XmlOutputParser::new();
    let input = "```xml\n<result>success</result>\n```";
    let result = parser.parse(input).unwrap();
    assert_eq!(result, json!({"result": "success"}));
}

#[test]
fn test_xml_parser_format_instructions_with_tags() {
    let parser = XmlOutputParser::with_tags(vec!["name".into(), "age".into()]);
    let instructions = parser.get_format_instructions().unwrap();
    assert!(instructions.contains("<name>"));
    assert!(instructions.contains("<age>"));
}

#[tokio::test]
async fn test_xml_parser_runnable() {
    let parser = XmlOutputParser::new();
    let result = parser
        .invoke(json!("<data>hello</data>"), None)
        .await
        .unwrap();
    assert_eq!(result, json!({"data": "hello"}));
}

// ─── TransformOutputParser Tests ───

#[test]
fn test_transform_output_parser_trait_exists() {
    // Just verify the trait and default_transform_stream are accessible
    fn _accepts_parser(_p: &dyn OutputParser) {}
    let parser = StrOutputParser;
    _accepts_parser(&parser);
}
