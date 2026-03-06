use rustchain_core::messages::*;
use serde_json::json;
use std::collections::HashMap;

#[test]
fn human_message_text() {
    let msg = HumanMessage::new("Hello");
    assert_eq!(msg.base.content.text(), "Hello");
}

#[test]
fn human_message_with_blocks() {
    let blocks = vec![
        ContentBlock::Text {
            text: "Part 1".into(),
            id: None,
            annotations: None,
            index: None,
            extras: None,
        },
        ContentBlock::Text {
            text: " Part 2".into(),
            id: None,
            annotations: None,
            index: None,
            extras: None,
        },
    ];
    let msg = HumanMessage::with_blocks(blocks);
    assert_eq!(msg.base.content.text(), "Part 1 Part 2");
}

#[test]
fn system_message() {
    let msg = SystemMessage::new("You are helpful.");
    assert_eq!(msg.base.content.text(), "You are helpful.");
}

#[test]
fn ai_message_simple() {
    let msg = AIMessage::new("The answer is 42.");
    assert_eq!(msg.base.content.text(), "The answer is 42.");
    assert!(msg.tool_calls.is_empty());
    assert!(msg.usage_metadata.is_none());
}

#[test]
fn ai_message_with_tool_calls() {
    let tc = ToolCall {
        name: "get_weather".into(),
        args: {
            let mut m = HashMap::new();
            m.insert("city".into(), json!("Paris"));
            m
        },
        id: Some("call_1".into()),
    };
    let msg = AIMessage::new("Let me check.").with_tool_calls(vec![tc.clone()]);
    assert_eq!(msg.tool_calls.len(), 1);
    assert_eq!(msg.tool_calls[0].name, "get_weather");
    assert_eq!(msg.tool_calls[0].id, Some("call_1".into()));
}

#[test]
fn ai_message_with_usage() {
    let usage = UsageMetadata::new(10, 20, 30);
    let msg = AIMessage::new("Hi").with_usage(usage);
    let u = msg.usage_metadata.unwrap();
    assert_eq!(u.input_tokens, 10);
    assert_eq!(u.output_tokens, 20);
    assert_eq!(u.total_tokens, 30);
}

#[test]
fn usage_metadata_add() {
    let a = UsageMetadata {
        input_tokens: 5,
        output_tokens: 10,
        total_tokens: 15,
        input_token_details: Some(InputTokenDetails {
            audio: None,
            cache_creation: None,
            cache_read: Some(3),
        }),
        output_token_details: None,
    };
    let b = UsageMetadata {
        input_tokens: 2,
        output_tokens: 8,
        total_tokens: 10,
        input_token_details: None,
        output_token_details: Some(OutputTokenDetails {
            audio: None,
            reasoning: Some(4),
        }),
    };
    let sum = a.add(&b);
    assert_eq!(sum.input_tokens, 7);
    assert_eq!(sum.output_tokens, 18);
    assert_eq!(sum.total_tokens, 25);
    assert_eq!(sum.input_token_details.unwrap().cache_read, Some(3));
    assert_eq!(sum.output_token_details.unwrap().reasoning, Some(4));
}

#[test]
fn tool_message_basic() {
    let msg = ToolMessage::new("42", "call_1");
    assert_eq!(msg.tool_call_id, "call_1");
    assert_eq!(msg.base.content.text(), "42");
    assert_eq!(msg.status, ToolStatus::Success);
}

#[test]
fn tool_message_error() {
    let msg = ToolMessage::new("failed", "call_2").with_error();
    assert_eq!(msg.status, ToolStatus::Error);
}

#[test]
fn tool_message_artifact() {
    let msg = ToolMessage::new("ok", "call_3").with_artifact(json!({"data": [1, 2, 3]}));
    assert!(msg.artifact.is_some());
}

#[test]
fn function_message() {
    let msg = FunctionMessage::new("my_func", "result");
    assert_eq!(msg.base.name, Some("my_func".into()));
    assert_eq!(msg.base.content.text(), "result");
}

#[test]
fn chat_message() {
    let msg = ChatMessage::new("assistant", "Hello!");
    assert_eq!(msg.role, "assistant");
    assert_eq!(msg.base.content.text(), "Hello!");
}

#[test]
fn message_enum_dispatch() {
    let msg = Message::Human(HumanMessage::new("Hi"));
    assert_eq!(msg.message_type(), MessageType::Human);
    assert_eq!(msg.content().text(), "Hi");
}

#[test]
fn message_type_as_str() {
    assert_eq!(MessageType::Human.as_str(), "human");
    assert_eq!(MessageType::Ai.as_str(), "ai");
    assert_eq!(MessageType::System.as_str(), "system");
    assert_eq!(MessageType::Tool.as_str(), "tool");
    assert_eq!(MessageType::Function.as_str(), "function");
    assert_eq!(MessageType::Chat.as_str(), "chat");
    assert_eq!(MessageType::Remove.as_str(), "remove");
}

#[test]
fn ai_message_chunk_basic() {
    let chunk = AIMessageChunk::new("Hello");
    assert_eq!(chunk.base.content.text(), "Hello");
    assert!(chunk.tool_call_chunks.is_empty());
    assert!(chunk.chunk_position.is_none());
}

#[test]
fn content_block_serde_roundtrip() {
    let block = ContentBlock::Text {
        text: "hello".into(),
        id: None,
        annotations: None,
        index: None,
        extras: None,
    };
    let json = serde_json::to_string(&block).unwrap();
    let back: ContentBlock = serde_json::from_str(&json).unwrap();
    assert_eq!(block, back);
}

#[test]
fn tool_call_chunk_fields() {
    let chunk = ToolCallChunk {
        name: Some("search".into()),
        args: Some("{\"q\":\"rust\"}".into()),
        id: Some("tc_1".into()),
        index: Some(0),
    };
    assert_eq!(chunk.name, Some("search".into()));
    assert_eq!(chunk.index, Some(0));
}

#[test]
fn invalid_tool_call_fields() {
    let itc = InvalidToolCall {
        name: Some("bad_tool".into()),
        args: Some("not json".into()),
        id: Some("tc_2".into()),
        error: Some("parse error".into()),
    };
    assert_eq!(itc.error, Some("parse error".into()));
}

#[test]
fn base_message_fields_builder() {
    let fields = BaseMessageFields::new(MessageContent::Text("test".into()))
        .with_name("alice")
        .with_id("msg_1");
    assert_eq!(fields.name, Some("alice".into()));
    assert_eq!(fields.id, Some("msg_1".into()));
}

#[test]
fn message_content_default() {
    let content = MessageContent::default();
    assert_eq!(content.text(), "");
}

// --- UsageMetadata subtract tests ---

#[test]
fn usage_metadata_subtract() {
    let a = UsageMetadata::new(100, 50, 150);
    let b = UsageMetadata::new(30, 20, 50);
    let result = a.subtract(&b);
    assert_eq!(result.input_tokens, 70);
    assert_eq!(result.output_tokens, 30);
    assert_eq!(result.total_tokens, 100);
}

#[test]
fn usage_metadata_subtract_saturates() {
    let a = UsageMetadata::new(5, 3, 8);
    let b = UsageMetadata::new(10, 10, 20);
    let result = a.subtract(&b);
    assert_eq!(result.input_tokens, 0);
    assert_eq!(result.output_tokens, 0);
    assert_eq!(result.total_tokens, 0);
}

#[test]
fn usage_metadata_subtract_with_details() {
    let a = UsageMetadata {
        input_tokens: 50,
        output_tokens: 30,
        total_tokens: 80,
        input_token_details: Some(InputTokenDetails {
            audio: Some(10),
            cache_creation: Some(5),
            cache_read: Some(20),
        }),
        output_token_details: Some(OutputTokenDetails {
            audio: Some(8),
            reasoning: Some(15),
        }),
    };
    let b = UsageMetadata {
        input_tokens: 20,
        output_tokens: 10,
        total_tokens: 30,
        input_token_details: Some(InputTokenDetails {
            audio: Some(3),
            cache_creation: None,
            cache_read: Some(5),
        }),
        output_token_details: Some(OutputTokenDetails {
            audio: Some(2),
            reasoning: Some(100), // larger than a's
        }),
    };
    let result = a.subtract(&b);
    assert_eq!(result.input_tokens, 30);
    let itd = result.input_token_details.unwrap();
    assert_eq!(itd.audio, Some(7));
    assert_eq!(itd.cache_creation, Some(5)); // b has None
    assert_eq!(itd.cache_read, Some(15));
    let otd = result.output_token_details.unwrap();
    assert_eq!(otd.audio, Some(6));
    assert_eq!(otd.reasoning, Some(0)); // saturating sub
}

#[test]
fn add_usage_free_function() {
    let a = UsageMetadata::new(10, 20, 30);
    let b = UsageMetadata::new(5, 10, 15);
    let result = add_usage(&a, &b);
    assert_eq!(result.input_tokens, 15);
    assert_eq!(result.output_tokens, 30);
    assert_eq!(result.total_tokens, 45);
}

// --- Message enum chunk variant tests ---

#[test]
fn message_enum_chunk_variants() {
    let msg = Message::HumanChunk(HumanMessageChunk::new("Hello"));
    assert_eq!(msg.message_type(), MessageType::Human);
    assert_eq!(msg.content().text(), "Hello");

    let msg = Message::AiChunk(AIMessageChunk::new("Response"));
    assert_eq!(msg.message_type(), MessageType::Ai);

    let msg = Message::SystemChunk(SystemMessageChunk::new("Sys"));
    assert_eq!(msg.message_type(), MessageType::System);

    let msg = Message::Remove(RemoveMessage::new("id-1"));
    assert_eq!(msg.message_type(), MessageType::Remove);
}

#[test]
fn message_base_returns_none_for_remove() {
    let msg = Message::Remove(RemoveMessage::new("id-1"));
    assert!(msg.base().is_none());
}

#[test]
fn message_base_returns_some_for_chunk() {
    let msg = Message::HumanChunk(HumanMessageChunk::new("Hi"));
    assert!(msg.base().is_some());
}

// =========================================================================
// ContentBlock expanded tests — all block types
// =========================================================================

#[test]
fn content_block_text_with_annotations() {
    use rustchain_core::messages::content::{Annotation, Citation};

    let block = ContentBlock::Text {
        text: "Hello world".into(),
        id: Some("lc_123".into()),
        annotations: Some(vec![Annotation::Citation(Citation {
            url: Some("https://example.com".into()),
            title: Some("Example".into()),
            start_index: Some(0),
            end_index: Some(5),
            cited_text: Some("Hello".into()),
            extras: None,
        })]),
        index: Some(0),
        extras: None,
    };
    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["type"], "text");
    assert_eq!(json["text"], "Hello world");
    assert_eq!(json["id"], "lc_123");
    assert!(json["annotations"].is_array());
    let back: ContentBlock = serde_json::from_value(json).unwrap();
    assert_eq!(block, back);
}

#[test]
fn content_block_image_url_source() {
    let block = ContentBlock::Image {
        id: Some("img_1".into()),
        url: Some("https://example.com/img.png".into()),
        base64: None,
        file_id: None,
        mime_type: Some("image/png".into()),
        index: None,
        image: None,
        source_type: None,
        media_type: None,
        extras: None,
    };
    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["type"], "image");
    assert_eq!(json["url"], "https://example.com/img.png");
    let back: ContentBlock = serde_json::from_value(json).unwrap();
    assert_eq!(block, back);
}

#[test]
fn content_block_image_base64_source() {
    let block = ContentBlock::Image {
        id: None,
        url: None,
        base64: Some("iVBORw0KGgo=".into()),
        file_id: None,
        mime_type: Some("image/png".into()),
        index: None,
        image: None,
        source_type: None,
        media_type: None,
        extras: None,
    };
    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["type"], "image");
    assert_eq!(json["base64"], "iVBORw0KGgo=");
    let back: ContentBlock = serde_json::from_value(json).unwrap();
    assert_eq!(block, back);
}

#[test]
fn content_block_image_file_id_source() {
    let block = ContentBlock::Image {
        id: None,
        url: None,
        base64: None,
        file_id: Some("file-abc123".into()),
        mime_type: None,
        index: None,
        image: None,
        source_type: None,
        media_type: None,
        extras: None,
    };
    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["type"], "image");
    assert_eq!(json["file_id"], "file-abc123");
    let back: ContentBlock = serde_json::from_value(json).unwrap();
    assert_eq!(block, back);
}

#[test]
fn content_block_image_url_block() {
    use rustchain_core::messages::ImageUrlInfo;

    let block = ContentBlock::ImageUrl {
        image_url: ImageUrlInfo {
            url: "https://example.com/img.png".into(),
            detail: Some("high".into()),
        },
        extras: None,
    };
    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["type"], "image_url");
    assert_eq!(json["image_url"]["url"], "https://example.com/img.png");
    assert_eq!(json["image_url"]["detail"], "high");
    let back: ContentBlock = serde_json::from_value(json).unwrap();
    assert_eq!(block, back);
}

#[test]
fn content_block_audio() {
    let block = ContentBlock::Audio {
        id: Some("aud_1".into()),
        url: Some("https://example.com/audio.mp3".into()),
        base64: None,
        file_id: None,
        mime_type: Some("audio/mpeg".into()),
        index: None,
        audio: None,
        source_type: None,
        media_type: None,
        extras: None,
    };
    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["type"], "audio");
    assert_eq!(json["url"], "https://example.com/audio.mp3");
    let back: ContentBlock = serde_json::from_value(json).unwrap();
    assert_eq!(block, back);
}

#[test]
fn content_block_video() {
    let block = ContentBlock::Video {
        id: None,
        url: Some("https://example.com/video.mp4".into()),
        base64: None,
        file_id: None,
        mime_type: Some("video/mp4".into()),
        index: None,
        video: None,
        source_type: None,
        media_type: None,
        extras: None,
    };
    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["type"], "video");
    let back: ContentBlock = serde_json::from_value(json).unwrap();
    assert_eq!(block, back);
}

#[test]
fn content_block_file() {
    let block = ContentBlock::File {
        id: None,
        url: None,
        base64: Some("JVBERi0=".into()),
        file_id: None,
        mime_type: Some("application/pdf".into()),
        index: None,
        file: None,
        source_type: None,
        media_type: None,
        extras: None,
    };
    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["type"], "file");
    assert_eq!(json["base64"], "JVBERi0=");
    let back: ContentBlock = serde_json::from_value(json).unwrap();
    assert_eq!(block, back);
}

#[test]
fn content_block_plain_text() {
    let block = ContentBlock::PlainText {
        id: None,
        url: None,
        base64: None,
        file_id: None,
        mime_type: Some("text/plain".into()),
        text: Some("Document content here".into()),
        title: Some("My Document".into()),
        context: Some("Summary of the document".into()),
        index: None,
        extras: None,
    };
    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["type"], "text-plain");
    assert_eq!(json["text"], "Document content here");
    assert_eq!(json["title"], "My Document");
    assert_eq!(json["context"], "Summary of the document");
    let back: ContentBlock = serde_json::from_value(json).unwrap();
    assert_eq!(block, back);
}

#[test]
fn content_block_reasoning() {
    let block = ContentBlock::Reasoning {
        reasoning: Some("I need to think about this step by step...".into()),
        id: Some("reason_1".into()),
        index: None,
        extras: None,
    };
    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["type"], "reasoning");
    assert_eq!(json["reasoning"], "I need to think about this step by step...");
    let back: ContentBlock = serde_json::from_value(json).unwrap();
    assert_eq!(block, back);
}

#[test]
fn content_block_redacted_thinking() {
    let block = ContentBlock::RedactedThinking {
        data: Some("opaque_data_here".into()),
        extras: None,
    };
    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["type"], "redacted_thinking");
    assert_eq!(json["data"], "opaque_data_here");
    let back: ContentBlock = serde_json::from_value(json).unwrap();
    assert_eq!(block, back);
}

#[test]
fn content_block_cache_control() {
    let block = ContentBlock::CacheControl {
        cache_type: Some("ephemeral".into()),
        extras: None,
    };
    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["type"], "cache_control");
    assert_eq!(json["cache_type"], "ephemeral");
    let back: ContentBlock = serde_json::from_value(json).unwrap();
    assert_eq!(block, back);
}

#[test]
fn content_block_tool_call() {
    let block = ContentBlock::ToolCall {
        name: "get_weather".into(),
        args: json!({"city": "Paris"}),
        id: Some("call_1".into()),
        index: Some(0),
        extras: None,
    };
    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["type"], "tool_call");
    assert_eq!(json["name"], "get_weather");
    assert_eq!(json["args"]["city"], "Paris");
    let back: ContentBlock = serde_json::from_value(json).unwrap();
    assert_eq!(block, back);
}

#[test]
fn content_block_tool_call_chunk() {
    let block = ContentBlock::ToolCallChunk {
        name: Some("search".into()),
        args: Some("{\"q\":\"rust".into()),
        id: Some("tc_1".into()),
        index: Some(0),
        extras: None,
    };
    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["type"], "tool_call_chunk");
    let back: ContentBlock = serde_json::from_value(json).unwrap();
    assert_eq!(block, back);
}

#[test]
fn content_block_invalid_tool_call() {
    let block = ContentBlock::InvalidToolCall {
        name: Some("bad_tool".into()),
        args: Some("not json".into()),
        id: Some("tc_2".into()),
        error: Some("parse error".into()),
        index: None,
        extras: None,
    };
    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["type"], "invalid_tool_call");
    assert_eq!(json["error"], "parse error");
    let back: ContentBlock = serde_json::from_value(json).unwrap();
    assert_eq!(block, back);
}

#[test]
fn content_block_server_tool_call() {
    let block = ContentBlock::ServerToolCall {
        name: "web_search".into(),
        args: json!({"query": "rust lang"}),
        id: Some("stc_1".into()),
        index: None,
        extras: None,
    };
    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["type"], "server_tool_call");
    let back: ContentBlock = serde_json::from_value(json).unwrap();
    assert_eq!(block, back);
}

#[test]
fn content_block_server_tool_call_chunk() {
    let block = ContentBlock::ServerToolCallChunk {
        name: Some("web_search".into()),
        args: Some("{\"query\":".into()),
        id: Some("stc_1".into()),
        index: Some(0),
        extras: None,
    };
    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["type"], "server_tool_call_chunk");
    let back: ContentBlock = serde_json::from_value(json).unwrap();
    assert_eq!(block, back);
}

#[test]
fn content_block_server_tool_result() {
    let block = ContentBlock::ServerToolResult {
        tool_call_id: Some("stc_1".into()),
        status: Some("success".into()),
        output: Some(json!({"result": "Rust is a systems programming language"})),
        id: Some("str_1".into()),
        index: None,
        extras: None,
    };
    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["type"], "server_tool_result");
    assert_eq!(json["status"], "success");
    let back: ContentBlock = serde_json::from_value(json).unwrap();
    assert_eq!(block, back);
}

#[test]
fn content_block_tool_result() {
    let block = ContentBlock::ToolResult {
        tool_use_id: "toolu_1".into(),
        content: Some(json!("The weather is sunny")),
        is_error: Some(false),
        extras: None,
    };
    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["type"], "tool_result");
    assert_eq!(json["tool_use_id"], "toolu_1");
    let back: ContentBlock = serde_json::from_value(json).unwrap();
    assert_eq!(block, back);
}

#[test]
fn content_block_data() {
    let block = ContentBlock::Data {
        data: json!({"custom": [1, 2, 3]}),
        extras: None,
    };
    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["type"], "data");
    let back: ContentBlock = serde_json::from_value(json).unwrap();
    assert_eq!(block, back);
}

#[test]
fn content_block_non_standard() {
    let block = ContentBlock::NonStandard {
        id: Some("ns_1".into()),
        value: json!({"provider_specific": "data"}),
        index: None,
    };
    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["type"], "non_standard");
    let back: ContentBlock = serde_json::from_value(json).unwrap();
    assert_eq!(block, back);
}

#[test]
fn content_block_image_legacy_fields() {
    // Test backward compatibility with legacy image fields
    let block = ContentBlock::Image {
        id: None,
        url: None,
        base64: None,
        file_id: None,
        mime_type: None,
        index: None,
        image: Some("base64data".into()),
        source_type: Some("base64".into()),
        media_type: Some("image/png".into()),
        extras: None,
    };
    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["image"], "base64data");
    assert_eq!(json["source_type"], "base64");
    assert_eq!(json["media_type"], "image/png");
    let back: ContentBlock = serde_json::from_value(json).unwrap();
    assert_eq!(block, back);
}

#[test]
fn known_block_types_contains_all() {
    use rustchain_core::messages::KNOWN_BLOCK_TYPES;
    assert!(KNOWN_BLOCK_TYPES.contains(&"text"));
    assert!(KNOWN_BLOCK_TYPES.contains(&"image"));
    assert!(KNOWN_BLOCK_TYPES.contains(&"image_url"));
    assert!(KNOWN_BLOCK_TYPES.contains(&"audio"));
    assert!(KNOWN_BLOCK_TYPES.contains(&"video"));
    assert!(KNOWN_BLOCK_TYPES.contains(&"file"));
    assert!(KNOWN_BLOCK_TYPES.contains(&"text-plain"));
    assert!(KNOWN_BLOCK_TYPES.contains(&"reasoning"));
    assert!(KNOWN_BLOCK_TYPES.contains(&"redacted_thinking"));
    assert!(KNOWN_BLOCK_TYPES.contains(&"cache_control"));
    assert!(KNOWN_BLOCK_TYPES.contains(&"tool_call"));
    assert!(KNOWN_BLOCK_TYPES.contains(&"tool_call_chunk"));
    assert!(KNOWN_BLOCK_TYPES.contains(&"invalid_tool_call"));
    assert!(KNOWN_BLOCK_TYPES.contains(&"tool_result"));
    assert!(KNOWN_BLOCK_TYPES.contains(&"server_tool_call"));
    assert!(KNOWN_BLOCK_TYPES.contains(&"server_tool_call_chunk"));
    assert!(KNOWN_BLOCK_TYPES.contains(&"server_tool_result"));
    assert!(KNOWN_BLOCK_TYPES.contains(&"data"));
    assert!(KNOWN_BLOCK_TYPES.contains(&"non_standard"));
}

#[test]
fn is_data_content_block_type_check() {
    use rustchain_core::messages::is_data_content_block_type;
    assert!(is_data_content_block_type("image"));
    assert!(is_data_content_block_type("video"));
    assert!(is_data_content_block_type("audio"));
    assert!(is_data_content_block_type("text-plain"));
    assert!(is_data_content_block_type("file"));
    assert!(!is_data_content_block_type("text"));
    assert!(!is_data_content_block_type("tool_call"));
    assert!(!is_data_content_block_type("reasoning"));
}

#[test]
fn message_content_text_from_blocks_extracts_text() {
    let content = MessageContent::Blocks(vec![
        ContentBlock::Text {
            text: "Hello ".into(),
            id: None,
            annotations: None,
            index: None,
            extras: None,
        },
        ContentBlock::Image {
            id: None,
            url: Some("https://example.com/img.png".into()),
            base64: None,
            file_id: None,
            mime_type: None,
            index: None,
            image: None,
            source_type: None,
            media_type: None,
            extras: None,
        },
        ContentBlock::Text {
            text: "world".into(),
            id: None,
            annotations: None,
            index: None,
            extras: None,
        },
    ]);
    // text() should only extract text from Text blocks
    assert_eq!(content.text(), "Hello world");
}
