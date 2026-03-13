use cognis_core::messages::*;

#[test]
fn get_buffer_string_basic() {
    let messages = vec![
        Message::Human(HumanMessage::new("What is Rust?")),
        Message::Ai(AIMessage::new("Rust is a systems programming language.")),
    ];
    let result = get_buffer_string(&messages, "Human", "AI");
    assert_eq!(
        result,
        "Human: What is Rust?\nAI: Rust is a systems programming language."
    );
}

#[test]
fn get_buffer_string_custom_prefixes() {
    let messages = vec![
        Message::Human(HumanMessage::new("Hi")),
        Message::Ai(AIMessage::new("Hello")),
    ];
    let result = get_buffer_string(&messages, "User", "Assistant");
    assert_eq!(result, "User: Hi\nAssistant: Hello");
}

#[test]
fn get_buffer_string_system_and_tool() {
    let messages = vec![
        Message::System(SystemMessage::new("Be helpful")),
        Message::Tool(ToolMessage::new("result", "tc-1")),
    ];
    let result = get_buffer_string(&messages, "Human", "AI");
    assert_eq!(result, "System: Be helpful\nTool: result");
}

#[test]
fn convert_to_messages_basic() {
    let input = vec![
        ("human".into(), "Hello".into()),
        ("ai".into(), "Hi".into()),
        ("system".into(), "Be nice".into()),
    ];
    let messages = convert_to_messages(input);
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].message_type(), MessageType::Human);
    assert_eq!(messages[1].message_type(), MessageType::Ai);
    assert_eq!(messages[2].message_type(), MessageType::System);
}

#[test]
fn convert_to_messages_aliases() {
    let input = vec![
        ("user".into(), "Hello".into()),
        ("assistant".into(), "Hi".into()),
    ];
    let messages = convert_to_messages(input);
    assert_eq!(messages[0].message_type(), MessageType::Human);
    assert_eq!(messages[1].message_type(), MessageType::Ai);
}

#[test]
fn convert_to_messages_unknown_role() {
    let input = vec![("moderator".into(), "Welcome".into())];
    let messages = convert_to_messages(input);
    assert_eq!(messages[0].message_type(), MessageType::Chat);
}

#[test]
fn filter_messages_by_type() {
    let messages = vec![
        Message::Human(HumanMessage::new("Hi")),
        Message::Ai(AIMessage::new("Hello")),
        Message::System(SystemMessage::new("Be nice")),
    ];
    let filtered = filter_messages(
        &messages,
        None,
        Some(&[MessageType::Human, MessageType::Ai]),
        None,
        None,
        None,
    );
    assert_eq!(filtered.len(), 2);
}

#[test]
fn filter_messages_exclude_type() {
    let messages = vec![
        Message::Human(HumanMessage::new("Hi")),
        Message::System(SystemMessage::new("Ignore")),
    ];
    let filtered = filter_messages(
        &messages,
        None,
        None,
        None,
        Some(&[MessageType::System]),
        None,
    );
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].message_type(), MessageType::Human);
}

#[test]
fn filter_messages_by_id() {
    let mut msg = HumanMessage::new("Hello");
    msg.base.id = Some("msg-1".into());
    let messages = vec![Message::Human(msg), Message::Ai(AIMessage::new("Hi"))];
    let filtered = filter_messages(&messages, None, None, None, None, Some(&["msg-1"]));
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].message_type(), MessageType::Ai);
}

#[test]
fn merge_message_runs_basic() {
    let messages = vec![
        Message::Human(HumanMessage::new("Hello ")),
        Message::Human(HumanMessage::new("there")),
        Message::Ai(AIMessage::new("Hi")),
    ];
    let merged = merge_message_runs(&messages);
    assert_eq!(merged.len(), 2);
    assert_eq!(merged[0].content().text(), "Hello \nthere");
    assert_eq!(merged[1].content().text(), "Hi");
}

#[test]
fn merge_message_runs_no_consecutive() {
    let messages = vec![
        Message::Human(HumanMessage::new("A")),
        Message::Ai(AIMessage::new("B")),
        Message::Human(HumanMessage::new("C")),
    ];
    let merged = merge_message_runs(&messages);
    assert_eq!(merged.len(), 3);
}

#[test]
fn merge_message_runs_empty() {
    let merged = merge_message_runs(&[]);
    assert!(merged.is_empty());
}

#[test]
fn trim_messages_first_strategy() {
    let messages = vec![
        Message::Human(HumanMessage::new("short")),
        Message::Ai(AIMessage::new("a longer message here")),
        Message::Human(HumanMessage::new("end")),
    ];
    // Simple word count as token counter
    let counter = |s: &str| -> usize { s.split_whitespace().count() };
    let trimmed = trim_messages(&messages, 2, &counter, TrimStrategy::First);
    assert_eq!(trimmed.len(), 1);
    assert_eq!(trimmed[0].content().text(), "short");
}

#[test]
fn trim_messages_last_strategy() {
    let messages = vec![
        Message::Human(HumanMessage::new("first message")),
        Message::Ai(AIMessage::new("second")),
        Message::Human(HumanMessage::new("third")),
    ];
    let counter = |s: &str| -> usize { s.split_whitespace().count() };
    let trimmed = trim_messages(&messages, 2, &counter, TrimStrategy::Last);
    assert_eq!(trimmed.len(), 2);
    assert_eq!(trimmed[0].content().text(), "second");
    assert_eq!(trimmed[1].content().text(), "third");
}

#[test]
fn message_chunk_to_message_converts() {
    let chunk = Message::HumanChunk(HumanMessageChunk::new("Hello"));
    let msg = message_chunk_to_message(&chunk);
    assert_eq!(msg.message_type(), MessageType::Human);
    assert_eq!(msg.content().text(), "Hello");
}

#[test]
fn message_chunk_to_message_ai_chunk() {
    let mut ai_chunk = AIMessageChunk::new("Response");
    ai_chunk.usage_metadata = Some(UsageMetadata::new(5, 10, 15));
    let chunk = Message::AiChunk(ai_chunk);
    let msg = message_chunk_to_message(&chunk);
    assert_eq!(msg.message_type(), MessageType::Ai);
    if let Message::Ai(ai) = msg {
        assert!(ai.usage_metadata.is_some());
    } else {
        panic!("Expected Ai variant");
    }
}

#[test]
fn messages_to_dict_and_back() {
    let messages = vec![
        Message::Human(HumanMessage::new("Hello")),
        Message::Ai(AIMessage::new("Hi")),
    ];
    let dicts = messages_to_dict(&messages);
    assert_eq!(dicts.len(), 2);
    let restored = messages_from_dict(&dicts).unwrap();
    assert_eq!(restored.len(), 2);
    assert_eq!(restored[0].content().text(), "Hello");
    assert_eq!(restored[1].content().text(), "Hi");
}

// =========================================================================
// Expanded convert_to_messages tests
// =========================================================================

#[test]
fn convert_to_messages_tool_and_function_roles() {
    let input = vec![
        ("tool".into(), "result".into()),
        ("function".into(), "output".into()),
    ];
    let messages = convert_to_messages(input);
    assert_eq!(messages[0].message_type(), MessageType::Tool);
    assert_eq!(messages[1].message_type(), MessageType::Function);
}

#[test]
fn convert_to_messages_developer_alias() {
    let input = vec![("developer".into(), "System instructions".into())];
    let messages = convert_to_messages(input);
    assert_eq!(messages[0].message_type(), MessageType::System);
    assert_eq!(messages[0].content().text(), "System instructions");
}

#[test]
fn convert_to_messages_empty_input() {
    let input: Vec<(String, String)> = vec![];
    let messages = convert_to_messages(input);
    assert!(messages.is_empty());
}

#[test]
fn convert_to_messages_flex_with_strings() {
    use cognis_core::messages::utils::MessageLike;
    let input = vec![
        MessageLike::Text("Hello".into()),
        MessageLike::Text("World".into()),
    ];
    let messages = convert_to_messages_flex(input).unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].message_type(), MessageType::Human);
    assert_eq!(messages[1].message_type(), MessageType::Human);
}

#[test]
fn convert_to_messages_flex_with_message() {
    use cognis_core::messages::utils::MessageLike;
    let msg = Message::Ai(AIMessage::new("Response"));
    let input = vec![MessageLike::Msg(Box::new(msg))];
    let messages = convert_to_messages_flex(input).unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].message_type(), MessageType::Ai);
}

#[test]
fn convert_to_messages_flex_with_dict() {
    use cognis_core::messages::utils::MessageLike;
    use serde_json::json;
    let input = vec![MessageLike::Dict(json!({
        "role": "system",
        "content": "Be helpful"
    }))];
    let messages = convert_to_messages_flex(input).unwrap();
    assert_eq!(messages[0].message_type(), MessageType::System);
    assert_eq!(messages[0].content().text(), "Be helpful");
}

#[test]
fn convert_to_messages_flex_with_tuple() {
    use cognis_core::messages::utils::MessageLike;
    let input = vec![MessageLike::Tuple("assistant".into(), "Hello".into())];
    let messages = convert_to_messages_flex(input).unwrap();
    assert_eq!(messages[0].message_type(), MessageType::Ai);
}

#[test]
fn convert_to_messages_flex_mixed() {
    use cognis_core::messages::utils::MessageLike;
    use serde_json::json;
    let input = vec![
        MessageLike::Text("Hi".into()),
        MessageLike::Tuple("ai".into(), "Hello".into()),
        MessageLike::Dict(json!({"type": "system", "content": "Be nice"})),
        MessageLike::Msg(Box::new(Message::Tool(ToolMessage::new("result", "tc-1")))),
    ];
    let messages = convert_to_messages_flex(input).unwrap();
    assert_eq!(messages.len(), 4);
    assert_eq!(messages[0].message_type(), MessageType::Human);
    assert_eq!(messages[1].message_type(), MessageType::Ai);
    assert_eq!(messages[2].message_type(), MessageType::System);
    assert_eq!(messages[3].message_type(), MessageType::Tool);
}

// =========================================================================
// Expanded filter_messages tests
// =========================================================================

#[test]
fn filter_messages_include_names() {
    let mut msg1 = HumanMessage::new("Hello");
    msg1.base.name = Some("alice".into());
    let mut msg2 = HumanMessage::new("Goodbye");
    msg2.base.name = Some("bob".into());
    let messages = vec![Message::Human(msg1), Message::Human(msg2)];
    let filtered = filter_messages(&messages, Some(&["alice"]), None, None, None, None);
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].content().text(), "Hello");
}

#[test]
fn filter_messages_exclude_names() {
    let mut msg1 = HumanMessage::new("Hello");
    msg1.base.name = Some("alice".into());
    let mut msg2 = AIMessage::new("Response");
    msg2.base.name = Some("bot".into());
    let messages = vec![Message::Human(msg1), Message::Ai(msg2)];
    let filtered = filter_messages(&messages, None, None, Some(&["bot"]), None, None);
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].message_type(), MessageType::Human);
}

#[test]
fn filter_messages_exclude_tool_calls() {
    use std::collections::HashMap;
    let tc = ToolCall {
        name: "get_weather".into(),
        args: HashMap::new(),
        id: Some("call_1".into()),
    };
    let messages = vec![
        Message::Human(HumanMessage::new("Hi")),
        Message::Ai(AIMessage::new("Let me check").with_tool_calls(vec![tc])),
        Message::Tool(ToolMessage::new("sunny", "call_1")),
        Message::Ai(AIMessage::new("It's sunny")),
    ];
    let filtered = filter_messages_full(
        &messages, None, None, None, None, None, None, true, // exclude_tool_calls
    );
    assert_eq!(filtered.len(), 2);
    assert_eq!(filtered[0].message_type(), MessageType::Human);
    assert_eq!(filtered[1].content().text(), "It's sunny");
}

#[test]
fn filter_messages_include_and_exclude() {
    let messages = vec![
        Message::Human(HumanMessage::new("Hi")),
        Message::Ai(AIMessage::new("Hello")),
        Message::System(SystemMessage::new("System")),
        Message::Tool(ToolMessage::new("result", "tc-1")),
    ];
    // Include only Human and Ai, but also exclude Ai
    let filtered = filter_messages(
        &messages,
        None,
        Some(&[MessageType::Human, MessageType::Ai]),
        None,
        Some(&[MessageType::Ai]),
        None,
    );
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].message_type(), MessageType::Human);
}

#[test]
fn filter_messages_include_ids() {
    let mut msg1 = HumanMessage::new("Hi");
    msg1.base.id = Some("msg-1".into());
    let mut msg2 = AIMessage::new("Hello");
    msg2.base.id = Some("msg-2".into());
    let messages = vec![Message::Human(msg1), Message::Ai(msg2)];
    let filtered = filter_messages_full(
        &messages,
        None,
        None,
        Some(&["msg-2"]),
        None,
        None,
        None,
        false,
    );
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].message_type(), MessageType::Ai);
}

#[test]
fn filter_messages_empty_input() {
    let filtered = filter_messages(&[], None, None, None, None, None);
    assert!(filtered.is_empty());
}

// =========================================================================
// Expanded merge_message_runs tests
// =========================================================================

#[test]
fn merge_message_runs_three_consecutive() {
    let messages = vec![
        Message::Human(HumanMessage::new("A")),
        Message::Human(HumanMessage::new("B")),
        Message::Human(HumanMessage::new("C")),
    ];
    let merged = merge_message_runs(&messages);
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].content().text(), "A\nB\nC");
}

#[test]
fn merge_message_runs_tool_messages_never_merged() {
    let messages = vec![
        Message::Tool(ToolMessage::new("result1", "tc-1")),
        Message::Tool(ToolMessage::new("result2", "tc-2")),
    ];
    let merged = merge_message_runs(&messages);
    assert_eq!(merged.len(), 2);
}

#[test]
fn merge_message_runs_ai_consecutive() {
    let messages = vec![
        Message::Ai(AIMessage::new("part 1")),
        Message::Ai(AIMessage::new("part 2")),
    ];
    let merged = merge_message_runs(&messages);
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].content().text(), "part 1\npart 2");
}

#[test]
fn merge_message_runs_system_consecutive() {
    let messages = vec![
        Message::System(SystemMessage::new("rule 1")),
        Message::System(SystemMessage::new("rule 2")),
    ];
    let merged = merge_message_runs(&messages);
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].content().text(), "rule 1\nrule 2");
}

#[test]
fn merge_message_runs_custom_separator() {
    let messages = vec![
        Message::Human(HumanMessage::new("Hello")),
        Message::Human(HumanMessage::new("World")),
    ];
    let merged = merge_message_runs_with_separator(&messages, " ");
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].content().text(), "Hello World");
}

#[test]
fn merge_message_runs_preserves_interleaved() {
    let messages = vec![
        Message::Human(HumanMessage::new("Q1")),
        Message::Ai(AIMessage::new("A1")),
        Message::Human(HumanMessage::new("Q2")),
        Message::Ai(AIMessage::new("A2")),
    ];
    let merged = merge_message_runs(&messages);
    assert_eq!(merged.len(), 4);
}

#[test]
fn merge_message_runs_with_empty_content() {
    let messages = vec![
        Message::Human(HumanMessage::new("")),
        Message::Human(HumanMessage::new("Hello")),
    ];
    let merged = merge_message_runs(&messages);
    assert_eq!(merged.len(), 1);
    // Empty + non-empty: no separator inserted when left is empty
    assert_eq!(merged[0].content().text(), "Hello");
}

// =========================================================================
// Expanded trim_messages tests
// =========================================================================

#[test]
fn trim_messages_first_exact_budget() {
    let messages = vec![
        Message::Human(HumanMessage::new("one two")),
        Message::Ai(AIMessage::new("three")),
    ];
    let counter = |s: &str| -> usize { s.split_whitespace().count() };
    let trimmed = trim_messages(&messages, 3, &counter, TrimStrategy::First);
    assert_eq!(trimmed.len(), 2);
}

#[test]
fn trim_messages_last_with_include_system() {
    let messages = vec![
        Message::System(SystemMessage::new("sys")),
        Message::Human(HumanMessage::new("hello world")),
        Message::Ai(AIMessage::new("response")),
    ];
    let counter = |s: &str| -> usize { s.split_whitespace().count() };
    let trimmed = trim_messages_full(
        &messages,
        2,
        &counter,
        TrimStrategy::Last,
        true, // include_system
        None,
        None,
    );
    // Budget is 2, system takes 1, so 1 left for the rest.
    // "response" = 1 token, fits.
    assert_eq!(trimmed.len(), 2);
    assert_eq!(trimmed[0].message_type(), MessageType::System);
    assert_eq!(trimmed[1].content().text(), "response");
}

#[test]
fn trim_messages_first_with_end_on() {
    let messages = vec![
        Message::Human(HumanMessage::new("A")),
        Message::Ai(AIMessage::new("B")),
        Message::Human(HumanMessage::new("C")),
        Message::Tool(ToolMessage::new("D", "tc-1")),
    ];
    let counter = |s: &str| -> usize { s.split_whitespace().count() };
    // All fit in budget, but end_on says we must end on Ai
    let trimmed = trim_messages_full(
        &messages,
        100,
        &counter,
        TrimStrategy::First,
        false,
        None,
        Some(&[MessageType::Ai]),
    );
    assert_eq!(trimmed.len(), 2);
    assert_eq!(trimmed[1].message_type(), MessageType::Ai);
}

#[test]
fn trim_messages_last_with_start_on() {
    let messages = vec![
        Message::Tool(ToolMessage::new("tool result", "tc-1")),
        Message::Human(HumanMessage::new("Hello")),
        Message::Ai(AIMessage::new("Hi")),
    ];
    let counter = |s: &str| -> usize { s.split_whitespace().count() };
    // Budget fits all. start_on says we must start on Human
    let trimmed = trim_messages_full(
        &messages,
        100,
        &counter,
        TrimStrategy::Last,
        false,
        Some(&[MessageType::Human]),
        None,
    );
    assert_eq!(trimmed.len(), 2);
    assert_eq!(trimmed[0].message_type(), MessageType::Human);
    assert_eq!(trimmed[1].message_type(), MessageType::Ai);
}

#[test]
fn trim_messages_empty_input() {
    let counter = |s: &str| -> usize { s.split_whitespace().count() };
    let trimmed = trim_messages(&[], 100, &counter, TrimStrategy::First);
    assert!(trimmed.is_empty());
}

#[test]
fn trim_messages_zero_budget() {
    let messages = vec![Message::Human(HumanMessage::new("Hello"))];
    let counter = |s: &str| -> usize { s.split_whitespace().count() };
    let trimmed = trim_messages(&messages, 0, &counter, TrimStrategy::First);
    assert!(trimmed.is_empty());
}

// =========================================================================
// message_chunk_to_message expanded tests
// =========================================================================

#[test]
fn message_chunk_to_message_system_chunk() {
    let chunk = Message::SystemChunk(SystemMessageChunk::new("System prompt"));
    let msg = message_chunk_to_message(&chunk);
    assert_eq!(msg.message_type(), MessageType::System);
    assert_eq!(msg.content().text(), "System prompt");
}

#[test]
fn message_chunk_to_message_tool_chunk() {
    let chunk = Message::ToolChunk(ToolMessageChunk {
        base: cognis_core::messages::BaseMessageFields::new(MessageContent::Text(
            "result".into(),
        )),
        tool_call_id: "tc-1".into(),
        tool_call_chunks: Vec::new(),
        artifact: None,
        status: cognis_core::messages::ToolStatus::Success,
    });
    let msg = message_chunk_to_message(&chunk);
    assert_eq!(msg.message_type(), MessageType::Tool);
    if let Message::Tool(t) = msg {
        assert_eq!(t.tool_call_id, "tc-1");
    } else {
        panic!("Expected Tool variant");
    }
}

#[test]
fn message_chunk_to_message_non_chunk_passthrough() {
    let msg = Message::Human(HumanMessage::new("passthrough"));
    let result = message_chunk_to_message(&msg);
    assert_eq!(result.content().text(), "passthrough");
}

// =========================================================================
// get_buffer_string expanded tests
// =========================================================================

#[test]
fn get_buffer_string_with_ai_tool_calls() {
    use std::collections::HashMap;
    let tc = ToolCall {
        name: "search".into(),
        args: HashMap::new(),
        id: Some("call_1".into()),
    };
    let messages = vec![Message::Ai(
        AIMessage::new("Searching...").with_tool_calls(vec![tc]),
    )];
    let result = get_buffer_string(&messages, "Human", "AI");
    assert!(result.contains("AI: Searching..."));
    assert!(result.contains("search"));
}

#[test]
fn get_buffer_string_chat_message() {
    let messages = vec![Message::Chat(ChatMessage::new("moderator", "Welcome"))];
    let result = get_buffer_string(&messages, "Human", "AI");
    assert_eq!(result, "moderator: Welcome");
}

#[test]
fn get_buffer_string_remove_message_skipped() {
    let messages = vec![
        Message::Human(HumanMessage::new("Hi")),
        Message::Remove(RemoveMessage::new("id-to-remove")),
        Message::Ai(AIMessage::new("Hello")),
    ];
    let result = get_buffer_string(&messages, "Human", "AI");
    assert_eq!(result, "Human: Hi\nAI: Hello");
}

#[test]
fn get_buffer_string_full_custom_separator() {
    let messages = vec![
        Message::Human(HumanMessage::new("Hi")),
        Message::Ai(AIMessage::new("Hello")),
    ];
    let result = get_buffer_string_full(&messages, "User", "Bot", "Sys", "Fn", "Tool", " | ");
    assert_eq!(result, "User: Hi | Bot: Hello");
}
