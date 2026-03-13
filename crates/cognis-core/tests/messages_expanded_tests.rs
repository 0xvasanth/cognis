use cognis_core::messages::*;
use serde_json::json;
use std::collections::HashMap;

// --- Task 6: OpenAI messages ---
#[test]
fn test_human_message_to_openai() {
    let msgs = vec![Message::Human(HumanMessage::new("Hello"))];
    let result = convert_to_openai_messages(&msgs);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0]["role"], "user");
    assert_eq!(result[0]["content"], "Hello");
}

#[test]
fn test_ai_message_to_openai() {
    let msgs = vec![Message::Ai(AIMessage::new("Hi there"))];
    let result = convert_to_openai_messages(&msgs);
    assert_eq!(result[0]["role"], "assistant");
    assert_eq!(result[0]["content"], "Hi there");
}

#[test]
fn test_system_message_to_openai() {
    let msgs = vec![Message::System(SystemMessage::new("Be helpful"))];
    let result = convert_to_openai_messages(&msgs);
    assert_eq!(result[0]["role"], "system");
}

#[test]
fn test_tool_message_to_openai() {
    let msgs = vec![Message::Tool(ToolMessage::new("result", "tc_1"))];
    let result = convert_to_openai_messages(&msgs);
    assert_eq!(result[0]["role"], "tool");
    assert_eq!(result[0]["content"], "result");
    assert_eq!(result[0]["tool_call_id"], "tc_1");
}

#[test]
fn test_ai_with_tool_calls_to_openai() {
    let mut ai = AIMessage::new("Let me search");
    ai.tool_calls = vec![ToolCall {
        name: "search".into(),
        args: {
            let mut m = HashMap::new();
            m.insert("q".into(), json!("rust"));
            m
        },
        id: Some("tc_1".into()),
    }];
    let msgs = vec![Message::Ai(ai)];
    let result = convert_to_openai_messages(&msgs);
    let tcs = result[0]["tool_calls"].as_array().unwrap();
    assert_eq!(tcs.len(), 1);
    assert_eq!(tcs[0]["type"], "function");
    assert_eq!(tcs[0]["id"], "tc_1");
    assert_eq!(tcs[0]["function"]["name"], "search");
}

#[test]
fn test_mixed_messages_to_openai() {
    let msgs = vec![
        Message::System(SystemMessage::new("Be helpful")),
        Message::Human(HumanMessage::new("Hi")),
        Message::Ai(AIMessage::new("Hello!")),
    ];
    let result = convert_to_openai_messages(&msgs);
    assert_eq!(result.len(), 3);
    assert_eq!(result[0]["role"], "system");
    assert_eq!(result[1]["role"], "user");
    assert_eq!(result[2]["role"], "assistant");
}

#[test]
fn test_count_tokens_approximately() {
    let msgs = vec![
        Message::Human(HumanMessage::new("Hello world")), // 11 chars
        Message::Ai(AIMessage::new("Hi")),                // 2 chars
    ];
    let tokens = count_tokens_approximately(&msgs, 4.0, 3.0);
    // (11/4 + 3) + (2/4 + 3) = 5.75 + 3.5 = 9.25 -> ceil = 10
    assert_eq!(tokens, 10);
}

// --- Task 7: Factory functions ---
#[test]
fn test_tool_call_factory() {
    let tc = tool_call(
        "search",
        {
            let mut m = HashMap::new();
            m.insert("q".into(), json!("rust"));
            m
        },
        Some("tc_1".into()),
    );
    assert_eq!(tc.name, "search");
    assert_eq!(tc.id, Some("tc_1".into()));
}

#[test]
fn test_tool_call_chunk_factory() {
    let tc = tool_call_chunk(
        Some("search".into()),
        Some(r#"{"q":"rust"}"#.into()),
        Some("tc_1".into()),
        Some(0),
    );
    assert_eq!(tc.name, Some("search".into()));
    assert_eq!(tc.index, Some(0));
}

#[test]
fn test_invalid_tool_call_factory() {
    let itc = invalid_tool_call(
        Some("bad".into()),
        Some("{invalid".into()),
        Some("tc_2".into()),
        Some("parse error".into()),
    );
    assert_eq!(itc.name, Some("bad".into()));
    assert_eq!(itc.error, Some("parse error".into()));
}

#[test]
fn test_default_tool_parser_success() {
    let raw = vec![json!({"name": "search", "args": {"q": "rust"}, "id": "tc_1"})];
    let (valid, invalid) = default_tool_parser(&raw);
    assert_eq!(valid.len(), 1);
    assert_eq!(invalid.len(), 0);
    assert_eq!(valid[0].name, "search");
}

#[test]
fn test_default_tool_parser_invalid_args() {
    let raw = vec![json!({"name": "bad", "args": "not-json-obj", "id": "tc_2"})];
    let (valid, invalid) = default_tool_parser(&raw);
    assert_eq!(valid.len(), 0);
    assert_eq!(invalid.len(), 1);
    assert!(invalid[0].error.is_some());
}

#[test]
fn test_default_tool_parser_string_args() {
    let raw = vec![json!({"name": "search", "args": "{\"q\": \"rust\"}", "id": "tc_3"})];
    let (valid, invalid) = default_tool_parser(&raw);
    assert_eq!(valid.len(), 1);
    assert_eq!(invalid.len(), 0);
    assert_eq!(valid[0].args["q"], json!("rust"));
}

#[test]
fn test_default_tool_chunk_parser() {
    let raw = vec![json!({"name": "search", "args": "{\"q\":\"rust\"}", "id": "tc_1", "index": 0})];
    let chunks = default_tool_chunk_parser(&raw);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].name, Some("search".into()));
    assert_eq!(chunks[0].index, Some(0));
}
