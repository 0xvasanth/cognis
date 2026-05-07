use cognis_core::messages::{
    AIMessageChunk, ChatMessageChunk, FunctionMessageChunk, HumanMessageChunk, MessageChunkTrait,
    RemoveMessage, SystemMessageChunk, ToolCallChunk, ToolMessageChunk, UsageMetadata,
};

#[test]
fn human_message_chunk_add() {
    let a = HumanMessageChunk::new("Hello ");
    let b = HumanMessageChunk::new("World");
    let result = a.add(b);
    assert_eq!(result.base.content.text(), "Hello World");
}

#[test]
fn system_message_chunk_add() {
    let a = SystemMessageChunk::new("Be ");
    let b = SystemMessageChunk::new("helpful");
    let result = a.add(b);
    assert_eq!(result.base.content.text(), "Be helpful");
}

#[test]
fn chat_message_chunk_add_same_role() {
    let a = ChatMessageChunk::new("admin", "Hello ");
    let b = ChatMessageChunk::new("admin", "there");
    let result = a.add(b);
    assert_eq!(result.base.content.text(), "Hello there");
    assert_eq!(result.role, "admin");
}

#[test]
#[should_panic(expected = "Cannot concatenate ChatMessageChunks with different roles")]
fn chat_message_chunk_add_different_roles() {
    let a = ChatMessageChunk::new("admin", "Hello");
    let b = ChatMessageChunk::new("user", "World");
    let _ = a.add(b);
}

#[test]
fn function_message_chunk_add_same_name() {
    let a = FunctionMessageChunk::new("search", "result1");
    let b = FunctionMessageChunk::new("search", "result2");
    let result = a.add(b);
    assert_eq!(result.base.content.text(), "result1result2");
    assert_eq!(result.name(), Some("search"));
}

#[test]
#[should_panic(expected = "Cannot concatenate FunctionMessageChunks with different names")]
fn function_message_chunk_add_different_names() {
    let a = FunctionMessageChunk::new("search", "a");
    let b = FunctionMessageChunk::new("calculate", "b");
    let _ = a.add(b);
}

#[test]
fn tool_message_chunk_add_same_id() {
    let a = ToolMessageChunk::new("part1", "call-1");
    let b = ToolMessageChunk::new("part2", "call-1");
    let result = a.add(b);
    assert_eq!(result.base.content.text(), "part1part2");
    assert_eq!(result.tool_call_id, "call-1");
}

#[test]
#[should_panic(expected = "Cannot concatenate ToolMessageChunks with different tool_call_ids")]
fn tool_message_chunk_add_different_ids() {
    let a = ToolMessageChunk::new("a", "call-1");
    let b = ToolMessageChunk::new("b", "call-2");
    let _ = a.add(b);
}

#[test]
fn remove_message_creation() {
    let rm = RemoveMessage::new("msg-123");
    assert_eq!(rm.id, "msg-123");
}

#[test]
fn remove_message_serde() {
    let rm = RemoveMessage::new("msg-456");
    let json = serde_json::to_string(&rm).unwrap();
    let deserialized: RemoveMessage = serde_json::from_str(&json).unwrap();
    assert_eq!(rm, deserialized);
}

#[test]
fn ai_message_chunk_add_content() {
    let a = AIMessageChunk::new("Hello ");
    let b = AIMessageChunk::new("World");
    let result = a.add(b);
    assert_eq!(result.base.content.text(), "Hello World");
}

#[test]
fn ai_message_chunk_add_with_tool_call_chunks() {
    let mut a = AIMessageChunk::new("");
    a.tool_call_chunks = vec![ToolCallChunk {
        name: Some("search".into()),
        args: Some("{\"q\":".into()),
        id: Some("tc-1".into()),
        index: Some(0),
    }];
    let mut b = AIMessageChunk::new("");
    b.tool_call_chunks = vec![ToolCallChunk {
        name: None,
        args: Some("\"rust\"}".into()),
        id: None,
        index: Some(0),
    }];
    let result = a.add(b);
    assert_eq!(result.tool_call_chunks.len(), 2);
}

#[test]
fn ai_message_chunk_add_with_usage_metadata() {
    let mut a = AIMessageChunk::new("Hi");
    a.usage_metadata = Some(UsageMetadata::new(10, 5, 15));
    let mut b = AIMessageChunk::new(" there");
    b.usage_metadata = Some(UsageMetadata::new(0, 3, 3));
    let result = a.add(b);
    let usage = result.usage_metadata.unwrap();
    assert_eq!(usage.input_tokens, 10);
    assert_eq!(usage.output_tokens, 8);
    assert_eq!(usage.total_tokens, 18);
}

#[test]
fn ai_message_chunk_add_one_usage_none() {
    let mut a = AIMessageChunk::new("Hi");
    a.usage_metadata = Some(UsageMetadata::new(10, 5, 15));
    let b = AIMessageChunk::new(" there");
    let result = a.add(b);
    let usage = result.usage_metadata.unwrap();
    assert_eq!(usage.input_tokens, 10);
}
