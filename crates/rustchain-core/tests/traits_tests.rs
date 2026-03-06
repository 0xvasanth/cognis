use rustchain_core::agents::{AgentAction, AgentActionMessageLog, AgentFinish, AgentStep};
use rustchain_core::caches::{BaseCache, InMemoryCache};
use rustchain_core::chat_history::{BaseChatMessageHistory, InMemoryChatMessageHistory};
use rustchain_core::messages::{HumanMessage, Message};
use rustchain_core::outputs::Generation;
use rustchain_core::prompt_values::{ChatPromptValue, ImagePromptValue, PromptValue, StringPromptValue};
use rustchain_core::stores::{BaseStore, ByteStore, InMemoryByteStore, InMemoryStore};
use serde_json::json;
use std::collections::HashMap;

// --- Store tests ---

#[tokio::test]
async fn in_memory_store_mget_mset() {
    let store = InMemoryStore::new();
    store
        .mset(vec![
            ("k1".into(), json!("v1")),
            ("k2".into(), json!("v2")),
        ])
        .await
        .unwrap();
    let vals = store.mget(&["k1".into(), "k2".into(), "k3".into()]).await.unwrap();
    assert_eq!(vals[0], Some(json!("v1")));
    assert_eq!(vals[1], Some(json!("v2")));
    assert_eq!(vals[2], None);
}

#[tokio::test]
async fn in_memory_store_mdelete() {
    let store = InMemoryStore::new();
    store.mset(vec![("k1".into(), json!(1))]).await.unwrap();
    store.mdelete(&["k1".into()]).await.unwrap();
    let vals = store.mget(&["k1".into()]).await.unwrap();
    assert_eq!(vals[0], None);
}

#[tokio::test]
async fn in_memory_store_yield_keys() {
    let store = InMemoryStore::new();
    store
        .mset(vec![
            ("abc".into(), json!(1)),
            ("abd".into(), json!(2)),
            ("xyz".into(), json!(3)),
        ])
        .await
        .unwrap();
    let keys = store.yield_keys(Some("ab")).await.unwrap();
    assert_eq!(keys.len(), 2);
    let all_keys = store.yield_keys(None).await.unwrap();
    assert_eq!(all_keys.len(), 3);
}

// --- Cache tests ---

#[tokio::test]
async fn in_memory_cache_lookup_update() {
    let cache = InMemoryCache::new();
    let result = cache.lookup("prompt", "model").await.unwrap();
    assert!(result.is_none());

    cache
        .update("prompt", "model", vec![Generation::new("Paris")])
        .await
        .unwrap();
    let result = cache.lookup("prompt", "model").await.unwrap();
    assert_eq!(result.unwrap()[0].text, "Paris");
}

#[tokio::test]
async fn in_memory_cache_clear() {
    let cache = InMemoryCache::new();
    cache
        .update("p", "m", vec![Generation::new("a")])
        .await
        .unwrap();
    cache.clear().await.unwrap();
    let result = cache.lookup("p", "m").await.unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn in_memory_cache_maxsize() {
    let cache = InMemoryCache::with_maxsize(2);
    cache.update("p1", "m", vec![Generation::new("a")]).await.unwrap();
    cache.update("p2", "m", vec![Generation::new("b")]).await.unwrap();
    cache.update("p3", "m", vec![Generation::new("c")]).await.unwrap();
    // One of p1 or p2 should have been evicted; only 2 remain
    let r1 = cache.lookup("p1", "m").await.unwrap();
    let r2 = cache.lookup("p2", "m").await.unwrap();
    let r3 = cache.lookup("p3", "m").await.unwrap();
    let present_count = [r1.is_some(), r2.is_some(), r3.is_some()]
        .iter()
        .filter(|&&b| b)
        .count();
    assert_eq!(present_count, 2);
    // p3 must be present (just inserted)
    assert!(r3.is_some());
}

// --- Chat history tests ---

#[tokio::test]
async fn in_memory_chat_history() {
    let history = InMemoryChatMessageHistory::new();
    history
        .add_messages(vec![
            Message::Human(HumanMessage::new("Hello")),
            Message::Human(HumanMessage::new("World")),
        ])
        .await
        .unwrap();
    let msgs = history.messages().await.unwrap();
    assert_eq!(msgs.len(), 2);
}

#[tokio::test]
async fn in_memory_chat_history_clear() {
    let history = InMemoryChatMessageHistory::new();
    history
        .add_messages(vec![Message::Human(HumanMessage::new("Hi"))])
        .await
        .unwrap();
    history.clear().await.unwrap();
    let msgs = history.messages().await.unwrap();
    assert!(msgs.is_empty());
}

// --- Prompt value tests ---

#[test]
fn string_prompt_value() {
    let pv = StringPromptValue::new("Hello");
    assert_eq!(PromptValue::to_string(&pv), "Hello");
    let msgs = pv.to_messages();
    assert_eq!(msgs.len(), 1);
}

#[test]
fn chat_prompt_value() {
    let msgs = vec![
        Message::Human(HumanMessage::new("Hi")),
    ];
    let pv = ChatPromptValue::new(msgs);
    let s = PromptValue::to_string(&pv);
    assert!(s.contains("human: Hi"));
    assert_eq!(pv.to_messages().len(), 1);
}

#[test]
fn image_prompt_value() {
    let pv = ImagePromptValue::new("https://example.com/img.png");
    assert_eq!(PromptValue::to_string(&pv), "https://example.com/img.png");
}

// --- Agent type tests ---

#[test]
fn agent_action() {
    let action = AgentAction::new("search", json!({"query": "rust"}), "Searching...");
    assert_eq!(action.tool, "search");
    assert_eq!(action.log, "Searching...");
}

#[test]
fn agent_finish() {
    let mut rv = HashMap::new();
    rv.insert("output".into(), json!("42"));
    let finish = AgentFinish::new(rv, "Final answer");
    assert_eq!(finish.return_values["output"], json!("42"));
}

#[test]
fn agent_step() {
    let action = AgentAction::new("tool", json!("input"), "log");
    let step = AgentStep::new(action, "result");
    assert_eq!(step.observation, "result");
    assert_eq!(step.action.tool, "tool");
}

#[test]
fn agent_action_serde_roundtrip() {
    let action = AgentAction::new("search", json!({"q": "test"}), "log");
    let json = serde_json::to_string(&action).unwrap();
    let back: AgentAction = serde_json::from_str(&json).unwrap();
    assert_eq!(action, back);
}

// --- AgentActionMessageLog tests ---

#[test]
fn agent_action_message_log() {
    let messages = vec![
        Message::Human(HumanMessage::new("What's the weather?")),
        Message::Ai(rustchain_core::messages::AIMessage::new("Let me check.")),
    ];
    let action = AgentActionMessageLog::new(
        "get_weather",
        json!({"city": "Paris"}),
        "Looking up weather",
        messages,
    );
    assert_eq!(action.tool, "get_weather");
    assert_eq!(action.message_log.len(), 2);
}

#[test]
fn agent_action_message_log_serde() {
    let action = AgentActionMessageLog::new(
        "search",
        json!("query"),
        "log",
        vec![Message::Human(HumanMessage::new("Hi"))],
    );
    let json_str = serde_json::to_string(&action).unwrap();
    let back: AgentActionMessageLog = serde_json::from_str(&json_str).unwrap();
    assert_eq!(action, back);
}

// --- ByteStore tests ---

#[tokio::test]
async fn in_memory_byte_store_mget_mset() {
    let store = InMemoryByteStore::new();
    store
        .mset(vec![
            ("k1".into(), b"hello".to_vec()),
            ("k2".into(), b"world".to_vec()),
        ])
        .await
        .unwrap();
    let vals = store.mget(&["k1".into(), "k2".into(), "k3".into()]).await.unwrap();
    assert_eq!(vals[0], Some(b"hello".to_vec()));
    assert_eq!(vals[1], Some(b"world".to_vec()));
    assert_eq!(vals[2], None);
}

#[tokio::test]
async fn in_memory_byte_store_mdelete() {
    let store = InMemoryByteStore::new();
    store.mset(vec![("k1".into(), vec![1, 2, 3])]).await.unwrap();
    store.mdelete(&["k1".into()]).await.unwrap();
    let vals = store.mget(&["k1".into()]).await.unwrap();
    assert_eq!(vals[0], None);
}

#[tokio::test]
async fn in_memory_byte_store_yield_keys() {
    let store = InMemoryByteStore::new();
    store
        .mset(vec![
            ("abc".into(), vec![1]),
            ("abd".into(), vec![2]),
            ("xyz".into(), vec![3]),
        ])
        .await
        .unwrap();
    let keys = store.yield_keys(Some("ab")).await.unwrap();
    assert_eq!(keys.len(), 2);
    let all = store.yield_keys(None).await.unwrap();
    assert_eq!(all.len(), 3);
}
