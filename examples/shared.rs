//! Shared utilities for examples.
//!
//! Provides `get_chat_model()` which auto-detects a local Ollama server.
//! If Ollama is running, returns a real `ChatOllama` model.
//! Otherwise, falls back to `FakeListChatModel` with predefined responses.

#![allow(dead_code)]

use std::sync::Arc;

use cognis::chat_models::ollama::ChatOllama;
use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::language_models::{FakeListChatModel, FakeMessagesListChatModel};
use cognis_core::messages::Message;

/// Check if Ollama is reachable at localhost:11434.
pub fn is_ollama_available() -> bool {
    std::net::TcpStream::connect_timeout(
        &"127.0.0.1:11434".parse().unwrap(),
        std::time::Duration::from_secs(1),
    )
    .is_ok()
}

/// Returns a chat model: real Ollama if available, otherwise a fake model.
///
/// The `fake_responses` are used as fallback when Ollama is not running.
/// When Ollama is available, these are ignored and real LLM responses are used.
pub fn get_chat_model(fake_responses: Vec<String>) -> Arc<dyn BaseChatModel> {
    if is_ollama_available() {
        let model = ChatOllama::builder()
            .model("llama3.2")
            .temperature(0.3)
            .num_predict(256)
            .build()
            .expect("Failed to build ChatOllama");
        println!("[Using Ollama llama3.2]\n");
        Arc::new(model)
    } else {
        println!("[Ollama not detected — using fake model]\n");
        Arc::new(FakeListChatModel::new(fake_responses))
    }
}

/// Returns a streaming-capable chat model.
/// Same auto-detection logic as `get_chat_model`.
pub fn get_streaming_model(fake_responses: Vec<String>) -> Arc<dyn BaseChatModel> {
    get_chat_model(fake_responses)
}

/// Returns a chat model for tool-calling examples.
///
/// When Ollama is available, returns a real model. Otherwise uses
/// `FakeMessagesListChatModel` with the provided `Message` sequence
/// to simulate a tool-calling conversation (e.g., AI message with
/// tool calls followed by a final AI answer).
pub fn get_tool_calling_model(fake_messages: Vec<Message>) -> Arc<dyn BaseChatModel> {
    if is_ollama_available() {
        let model = ChatOllama::builder()
            .model("llama3.2")
            .temperature(0.3)
            .num_predict(256)
            .build()
            .expect("Failed to build ChatOllama");
        println!("[Using Ollama llama3.2]\n");
        Arc::new(model)
    } else {
        println!("[Ollama not detected — using fake tool-calling model]\n");
        Arc::new(FakeMessagesListChatModel::new(fake_messages))
    }
}
