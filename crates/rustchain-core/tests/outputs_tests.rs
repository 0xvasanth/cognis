use rustchain_core::messages::{AIMessage, AIMessageChunk};
use rustchain_core::outputs::{
    ChatGeneration, ChatGenerationChunk, ChatResult, Generation, GenerationChunk, LLMResult,
    RunInfo, merge_chat_generation_chunks,
};
use serde_json::json;
use std::collections::HashMap;
use uuid::Uuid;

#[test]
fn generation_new() {
    let gen = Generation::new("Hello");
    assert_eq!(gen.text, "Hello");
    assert!(gen.generation_info.is_none());
}

#[test]
fn generation_with_info() {
    let mut info = HashMap::new();
    info.insert("finish_reason".into(), json!("stop"));
    let gen = Generation {
        text: "Hello".into(),
        generation_info: Some(info),
    };
    assert_eq!(
        gen.generation_info.unwrap()["finish_reason"],
        json!("stop")
    );
}

#[test]
fn chat_generation_new() {
    let msg = AIMessage::new("The answer is 42.");
    let gen = ChatGeneration::new(msg);
    assert_eq!(gen.text, "The answer is 42.");
    assert_eq!(gen.message.content().text(), "The answer is 42.");
}

#[test]
fn chat_result() {
    let gen = ChatGeneration::new(AIMessage::new("Hi"));
    let result = ChatResult {
        generations: vec![gen],
        llm_output: None,
    };
    assert_eq!(result.generations.len(), 1);
}

#[test]
fn run_info() {
    let id = Uuid::new_v4();
    let info = RunInfo { run_id: id };
    assert_eq!(info.run_id, id);
}

#[test]
fn llm_result_flatten_single() {
    let result = LLMResult {
        generations: vec![vec![Generation::new("a"), Generation::new("b")]],
        llm_output: Some({
            let mut m = HashMap::new();
            m.insert("token_usage".into(), json!({"total": 10}));
            m
        }),
        run: None,
    };
    let flat = result.flatten();
    assert_eq!(flat.len(), 1);
    assert_eq!(flat[0].generations[0].len(), 2);
    assert!(flat[0].llm_output.is_some());
}

#[test]
fn llm_result_flatten_multiple() {
    let result = LLMResult {
        generations: vec![
            vec![Generation::new("a")],
            vec![Generation::new("b")],
        ],
        llm_output: Some({
            let mut m = HashMap::new();
            m.insert("token_usage".into(), json!({"total": 10}));
            m
        }),
        run: None,
    };
    let flat = result.flatten();
    assert_eq!(flat.len(), 2);
    // First keeps original token_usage
    assert_eq!(
        flat[0].llm_output.as_ref().unwrap()["token_usage"],
        json!({"total": 10})
    );
    // Second has empty token_usage
    assert_eq!(
        flat[1].llm_output.as_ref().unwrap()["token_usage"],
        json!({})
    );
}

#[test]
fn generation_serde_roundtrip() {
    let gen = Generation::new("test");
    let json = serde_json::to_string(&gen).unwrap();
    let back: Generation = serde_json::from_str(&json).unwrap();
    assert_eq!(gen, back);
}

// --- GenerationChunk tests ---

#[test]
fn generation_chunk_new() {
    let chunk = GenerationChunk::new("Hello");
    assert_eq!(chunk.text, "Hello");
    assert!(chunk.generation_info.is_none());
}

#[test]
fn generation_chunk_add() {
    let a = GenerationChunk::new("Hello ");
    let b = GenerationChunk::new("World");
    let result = a.add(&b);
    assert_eq!(result.text, "Hello World");
}

#[test]
fn generation_chunk_add_merges_info() {
    let mut a = GenerationChunk::new("a");
    a.generation_info = Some({
        let mut m = HashMap::new();
        m.insert("key1".into(), json!("val1"));
        m
    });
    let mut b = GenerationChunk::new("b");
    b.generation_info = Some({
        let mut m = HashMap::new();
        m.insert("key2".into(), json!("val2"));
        m
    });
    let result = a.add(&b);
    let info = result.generation_info.unwrap();
    assert_eq!(info["key1"], json!("val1"));
    assert_eq!(info["key2"], json!("val2"));
}

// --- ChatGenerationChunk tests ---

#[test]
fn chat_generation_chunk_new() {
    let chunk = ChatGenerationChunk::new(AIMessageChunk::new("Hi"));
    assert_eq!(chunk.text, "Hi");
}

#[test]
fn chat_generation_chunk_add() {
    let a = ChatGenerationChunk::new(AIMessageChunk::new("Hello "));
    let b = ChatGenerationChunk::new(AIMessageChunk::new("World"));
    let result = a.add(&b);
    assert_eq!(result.text, "Hello World");
    assert_eq!(result.message.base.content.text(), "Hello World");
}

#[test]
fn merge_chat_generation_chunks_works() {
    let chunks = vec![
        ChatGenerationChunk::new(AIMessageChunk::new("a")),
        ChatGenerationChunk::new(AIMessageChunk::new("b")),
        ChatGenerationChunk::new(AIMessageChunk::new("c")),
    ];
    let merged = merge_chat_generation_chunks(chunks).unwrap();
    assert_eq!(merged.text, "abc");
}

#[test]
fn merge_chat_generation_chunks_empty() {
    let result = merge_chat_generation_chunks(vec![]);
    assert!(result.is_none());
}
