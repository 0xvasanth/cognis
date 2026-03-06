//! Block translators for converting between LangChain message formats
//! and provider-specific formats.
//!
//! Mirrors Python `langchain_core.messages.block_translators`.

use serde_json::{json, Value};

use crate::error::{Result, RustChainError};

/// Trait for translating between LangChain and provider message formats.
#[allow(clippy::wrong_self_convention)]
pub trait BlockTranslator: Send + Sync {
    /// Provider name (e.g., "openai", "anthropic").
    fn provider(&self) -> &str;

    /// Convert a LangChain message to provider format.
    fn to_provider_message(&self, message: &Value) -> Result<Value>;

    /// Convert a provider message to LangChain format.
    fn from_provider_message(&self, message: &Value) -> Result<Value>;

    /// Convert a content block to provider format.
    fn to_provider_content_block(&self, block: &Value) -> Result<Value>;

    /// Convert a provider content block to LangChain format.
    fn from_provider_content_block(&self, block: &Value) -> Result<Value>;
}

// ============================================================
// OpenAI Block Translator
// ============================================================

/// Translates between LangChain message formats and OpenAI API formats.
pub struct OpenAIBlockTranslator;

impl BlockTranslator for OpenAIBlockTranslator {
    fn provider(&self) -> &str {
        "openai"
    }

    fn to_provider_message(&self, message: &Value) -> Result<Value> {
        let role = message
            .get("type")
            .or_else(|| message.get("role"))
            .and_then(|r| r.as_str())
            .unwrap_or("user");

        let openai_role = match role {
            "human" | "user" => "user",
            "ai" | "assistant" => "assistant",
            "system" => "system",
            "tool" => "tool",
            "function" => "function",
            other => other,
        };

        let mut result = json!({ "role": openai_role });

        // Handle content
        if let Some(content) = message.get("content") {
            match content {
                Value::String(s) => {
                    result["content"] = Value::String(s.clone());
                }
                Value::Array(blocks) => {
                    let translated: Vec<Value> = blocks
                        .iter()
                        .filter_map(|b| self.to_provider_content_block(b).ok())
                        .collect();
                    result["content"] = Value::Array(translated);
                }
                other => {
                    result["content"] = other.clone();
                }
            }
        }

        // Handle tool calls
        if let Some(tool_calls) = message.get("tool_calls") {
            result["tool_calls"] = tool_calls.clone();
        }

        // Handle tool_call_id for tool messages
        if openai_role == "tool" {
            if let Some(id) = message.get("tool_call_id") {
                result["tool_call_id"] = id.clone();
            }
        }

        // Handle name
        if let Some(name) = message.get("name") {
            result["name"] = name.clone();
        }

        Ok(result)
    }

    fn from_provider_message(&self, message: &Value) -> Result<Value> {
        let role = message
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("user");

        let lc_type = match role {
            "user" => "human",
            "assistant" => "ai",
            "system" => "system",
            "tool" => "tool",
            "function" => "function",
            other => other,
        };

        let mut result = json!({ "type": lc_type });

        if let Some(content) = message.get("content") {
            match content {
                Value::Array(blocks) => {
                    let translated: Vec<Value> = blocks
                        .iter()
                        .filter_map(|b| self.from_provider_content_block(b).ok())
                        .collect();
                    result["content"] = Value::Array(translated);
                }
                other => {
                    result["content"] = other.clone();
                }
            }
        }

        if let Some(tool_calls) = message.get("tool_calls") {
            result["tool_calls"] = tool_calls.clone();
        }
        if let Some(id) = message.get("tool_call_id") {
            result["tool_call_id"] = id.clone();
        }
        if let Some(name) = message.get("name") {
            result["name"] = name.clone();
        }

        Ok(result)
    }

    fn to_provider_content_block(&self, block: &Value) -> Result<Value> {
        let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("text");
        match block_type {
            "text" => Ok(json!({
                "type": "text",
                "text": block.get("text").cloned().unwrap_or(Value::String(String::new())),
            })),
            "image_url" | "image" => {
                let url = block
                    .get("image_url")
                    .or_else(|| block.get("url"))
                    .or_else(|| block.get("data"))
                    .cloned()
                    .unwrap_or(Value::Null);
                let url_obj = if url.is_string() {
                    json!({ "url": url })
                } else {
                    url
                };
                Ok(json!({
                    "type": "image_url",
                    "image_url": url_obj,
                }))
            }
            _ => Ok(block.clone()),
        }
    }

    fn from_provider_content_block(&self, block: &Value) -> Result<Value> {
        let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("text");
        match block_type {
            "text" => Ok(json!({
                "type": "text",
                "text": block.get("text").cloned().unwrap_or(Value::String(String::new())),
            })),
            "image_url" => {
                let url = block
                    .get("image_url")
                    .and_then(|u| u.get("url"))
                    .cloned()
                    .unwrap_or(Value::Null);
                Ok(json!({
                    "type": "image_url",
                    "image_url": { "url": url },
                }))
            }
            _ => Ok(block.clone()),
        }
    }
}

// ============================================================
// Anthropic Block Translator
// ============================================================

/// Translates between LangChain message formats and Anthropic API formats.
pub struct AnthropicBlockTranslator;

impl BlockTranslator for AnthropicBlockTranslator {
    fn provider(&self) -> &str {
        "anthropic"
    }

    fn to_provider_message(&self, message: &Value) -> Result<Value> {
        let role = message
            .get("type")
            .or_else(|| message.get("role"))
            .and_then(|r| r.as_str())
            .unwrap_or("user");

        let anthropic_role = match role {
            "human" | "user" => "user",
            "ai" | "assistant" => "assistant",
            _ => {
                return Err(RustChainError::Other(format!(
                    "Anthropic doesn't support role: {}",
                    role
                )))
            }
        };

        let mut result = json!({ "role": anthropic_role });

        if let Some(content) = message.get("content") {
            match content {
                Value::String(s) => {
                    result["content"] = Value::String(s.clone());
                }
                Value::Array(blocks) => {
                    let translated: Vec<Value> = blocks
                        .iter()
                        .filter_map(|b| self.to_provider_content_block(b).ok())
                        .collect();
                    result["content"] = Value::Array(translated);
                }
                other => result["content"] = other.clone(),
            }
        }

        Ok(result)
    }

    fn from_provider_message(&self, message: &Value) -> Result<Value> {
        let role = message
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("user");
        let lc_type = match role {
            "user" => "human",
            "assistant" => "ai",
            other => other,
        };

        let mut result = json!({ "type": lc_type });

        if let Some(content) = message.get("content") {
            match content {
                Value::Array(blocks) => {
                    let translated: Vec<Value> = blocks
                        .iter()
                        .filter_map(|b| self.from_provider_content_block(b).ok())
                        .collect();
                    result["content"] = Value::Array(translated);
                }
                other => result["content"] = other.clone(),
            }
        }

        // Handle tool_use blocks in content -> tool_calls
        if let Some(Value::Array(blocks)) = message.get("content") {
            let tool_calls: Vec<Value> = blocks
                .iter()
                .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_use"))
                .map(|b| {
                    json!({
                        "id": b.get("id").cloned().unwrap_or(Value::Null),
                        "name": b.get("name").cloned().unwrap_or(Value::Null),
                        "args": b.get("input").cloned().unwrap_or(json!({})),
                    })
                })
                .collect();
            if !tool_calls.is_empty() {
                result["tool_calls"] = Value::Array(tool_calls);
            }
        }

        Ok(result)
    }

    fn to_provider_content_block(&self, block: &Value) -> Result<Value> {
        let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("text");
        match block_type {
            "text" => Ok(json!({
                "type": "text",
                "text": block.get("text").cloned().unwrap_or(Value::String(String::new())),
            })),
            "image_url" | "image" => {
                let url = block
                    .get("image_url")
                    .and_then(|u| {
                        if u.is_string() {
                            Some(u.clone())
                        } else {
                            u.get("url").cloned()
                        }
                    })
                    .or_else(|| block.get("url").cloned())
                    .or_else(|| block.get("data").cloned());

                if let Some(Value::String(data_uri)) = &url {
                    if data_uri.starts_with("data:") {
                        if let Some((mime, data)) =
                            crate::language_models::utils::parse_data_uri(data_uri)
                        {
                            return Ok(json!({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime,
                                    "data": data,
                                }
                            }));
                        }
                    }
                    return Ok(json!({
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": data_uri,
                        }
                    }));
                }
                Ok(block.clone())
            }
            "tool_use" => Ok(json!({
                "type": "tool_use",
                "id": block.get("id").cloned().unwrap_or(Value::Null),
                "name": block.get("name").cloned().unwrap_or(Value::Null),
                "input": block.get("input").or_else(|| block.get("args")).cloned().unwrap_or(json!({})),
            })),
            "tool_result" => Ok(json!({
                "type": "tool_result",
                "tool_use_id": block.get("tool_use_id").cloned().unwrap_or(Value::Null),
                "content": block.get("content").cloned().unwrap_or(Value::Null),
            })),
            _ => Ok(block.clone()),
        }
    }

    fn from_provider_content_block(&self, block: &Value) -> Result<Value> {
        let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("text");
        match block_type {
            "text" => Ok(json!({
                "type": "text",
                "text": block.get("text").cloned().unwrap_or(Value::String(String::new())),
            })),
            "image" => {
                let source = block.get("source");
                if let Some(src) = source {
                    let src_type = src.get("type").and_then(|t| t.as_str()).unwrap_or("url");
                    match src_type {
                        "base64" => {
                            let mime = src
                                .get("media_type")
                                .and_then(|m| m.as_str())
                                .unwrap_or("image/png");
                            let data = src.get("data").and_then(|d| d.as_str()).unwrap_or("");
                            let data_uri =
                                crate::language_models::utils::build_data_uri(mime, data);
                            Ok(json!({
                                "type": "image_url",
                                "image_url": { "url": data_uri },
                            }))
                        }
                        _ => {
                            let url = src.get("url").cloned().unwrap_or(Value::Null);
                            Ok(json!({
                                "type": "image_url",
                                "image_url": { "url": url },
                            }))
                        }
                    }
                } else {
                    Ok(block.clone())
                }
            }
            "tool_use" => Ok(json!({
                "type": "tool_use",
                "id": block.get("id").cloned().unwrap_or(Value::Null),
                "name": block.get("name").cloned().unwrap_or(Value::Null),
                "input": block.get("input").cloned().unwrap_or(json!({})),
            })),
            _ => Ok(block.clone()),
        }
    }
}

// ============================================================
// Google GenAI Block Translator
// ============================================================

/// Translates between LangChain message formats and Google GenAI API formats.
pub struct GoogleGenAIBlockTranslator;

impl BlockTranslator for GoogleGenAIBlockTranslator {
    fn provider(&self) -> &str {
        "google_genai"
    }

    fn to_provider_message(&self, message: &Value) -> Result<Value> {
        let role = message
            .get("type")
            .or_else(|| message.get("role"))
            .and_then(|r| r.as_str())
            .unwrap_or("user");

        let google_role = match role {
            "human" | "user" => "user",
            "ai" | "assistant" => "model",
            "system" => "user", // Google uses system_instruction separately
            _ => "user",
        };

        let mut parts = Vec::new();
        if let Some(content) = message.get("content") {
            match content {
                Value::String(s) => {
                    parts.push(json!({ "text": s }));
                }
                Value::Array(blocks) => {
                    for block in blocks {
                        if let Ok(translated) = self.to_provider_content_block(block) {
                            parts.push(translated);
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(json!({
            "role": google_role,
            "parts": parts,
        }))
    }

    fn from_provider_message(&self, message: &Value) -> Result<Value> {
        let role = message
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("user");
        let lc_type = match role {
            "user" => "human",
            "model" => "ai",
            other => other,
        };

        let mut content_blocks = Vec::new();
        if let Some(Value::Array(parts)) = message.get("parts") {
            for part in parts {
                if let Ok(translated) = self.from_provider_content_block(part) {
                    content_blocks.push(translated);
                }
            }
        }

        let content = if content_blocks.len() == 1 {
            if let Some(text) = content_blocks[0].get("text") {
                text.clone()
            } else {
                Value::Array(content_blocks)
            }
        } else {
            Value::Array(content_blocks)
        };

        Ok(json!({
            "type": lc_type,
            "content": content,
        }))
    }

    fn to_provider_content_block(&self, block: &Value) -> Result<Value> {
        let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("text");
        match block_type {
            "text" => Ok(json!({
                "text": block.get("text").cloned().unwrap_or(Value::String(String::new())),
            })),
            "image_url" | "image" => {
                let url = block
                    .get("image_url")
                    .and_then(|u| {
                        if u.is_string() {
                            Some(u.clone())
                        } else {
                            u.get("url").cloned()
                        }
                    })
                    .or_else(|| block.get("url").cloned());
                Ok(json!({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": url.unwrap_or(Value::Null),
                    }
                }))
            }
            _ => Ok(json!({ "text": block.to_string() })),
        }
    }

    fn from_provider_content_block(&self, block: &Value) -> Result<Value> {
        if let Some(text) = block.get("text") {
            return Ok(json!({ "type": "text", "text": text }));
        }
        if let Some(inline) = block.get("inline_data") {
            let mime = inline
                .get("mime_type")
                .and_then(|m| m.as_str())
                .unwrap_or("image/png");
            let data = inline.get("data").and_then(|d| d.as_str()).unwrap_or("");
            let data_uri = crate::language_models::utils::build_data_uri(mime, data);
            return Ok(json!({
                "type": "image_url",
                "image_url": { "url": data_uri },
            }));
        }
        if let Some(fc) = block.get("functionCall") {
            return Ok(json!({
                "type": "tool_use",
                "name": fc.get("name").cloned().unwrap_or(Value::Null),
                "input": fc.get("args").cloned().unwrap_or(json!({})),
            }));
        }
        Ok(block.clone())
    }
}

/// Get a block translator by provider name.
pub fn get_translator(provider: &str) -> Option<Box<dyn BlockTranslator>> {
    match provider {
        "openai" => Some(Box::new(OpenAIBlockTranslator)),
        "anthropic" => Some(Box::new(AnthropicBlockTranslator)),
        "google_genai" | "google" => Some(Box::new(GoogleGenAIBlockTranslator)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ========== OpenAI Tests ==========

    #[test]
    fn test_openai_provider_name() {
        let translator = OpenAIBlockTranslator;
        assert_eq!(translator.provider(), "openai");
    }

    #[test]
    fn test_openai_to_provider_human_message() {
        let translator = OpenAIBlockTranslator;
        let msg = json!({ "type": "human", "content": "Hello" });
        let result = translator.to_provider_message(&msg).unwrap();
        assert_eq!(result["role"], "user");
        assert_eq!(result["content"], "Hello");
    }

    #[test]
    fn test_openai_to_provider_ai_message() {
        let translator = OpenAIBlockTranslator;
        let msg = json!({ "type": "ai", "content": "Hi there" });
        let result = translator.to_provider_message(&msg).unwrap();
        assert_eq!(result["role"], "assistant");
        assert_eq!(result["content"], "Hi there");
    }

    #[test]
    fn test_openai_to_provider_system_message() {
        let translator = OpenAIBlockTranslator;
        let msg = json!({ "type": "system", "content": "You are helpful." });
        let result = translator.to_provider_message(&msg).unwrap();
        assert_eq!(result["role"], "system");
        assert_eq!(result["content"], "You are helpful.");
    }

    #[test]
    fn test_openai_to_provider_tool_message() {
        let translator = OpenAIBlockTranslator;
        let msg = json!({
            "type": "tool",
            "content": "result",
            "tool_call_id": "call_123"
        });
        let result = translator.to_provider_message(&msg).unwrap();
        assert_eq!(result["role"], "tool");
        assert_eq!(result["tool_call_id"], "call_123");
    }

    #[test]
    fn test_openai_to_provider_with_tool_calls() {
        let translator = OpenAIBlockTranslator;
        let msg = json!({
            "type": "ai",
            "content": "",
            "tool_calls": [{"id": "call_1", "function": {"name": "get_weather"}}]
        });
        let result = translator.to_provider_message(&msg).unwrap();
        assert!(result.get("tool_calls").is_some());
    }

    #[test]
    fn test_openai_to_provider_with_name() {
        let translator = OpenAIBlockTranslator;
        let msg = json!({ "type": "human", "content": "Hi", "name": "Alice" });
        let result = translator.to_provider_message(&msg).unwrap();
        assert_eq!(result["name"], "Alice");
    }

    #[test]
    fn test_openai_to_provider_content_blocks() {
        let translator = OpenAIBlockTranslator;
        let msg = json!({
            "type": "human",
            "content": [
                {"type": "text", "text": "Look at this image"},
                {"type": "image_url", "image_url": "https://example.com/img.png"}
            ]
        });
        let result = translator.to_provider_message(&msg).unwrap();
        let content = result["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image_url");
    }

    #[test]
    fn test_openai_from_provider_user_message() {
        let translator = OpenAIBlockTranslator;
        let msg = json!({ "role": "user", "content": "Hello" });
        let result = translator.from_provider_message(&msg).unwrap();
        assert_eq!(result["type"], "human");
        assert_eq!(result["content"], "Hello");
    }

    #[test]
    fn test_openai_from_provider_assistant_message() {
        let translator = OpenAIBlockTranslator;
        let msg = json!({ "role": "assistant", "content": "Hi" });
        let result = translator.from_provider_message(&msg).unwrap();
        assert_eq!(result["type"], "ai");
    }

    #[test]
    fn test_openai_from_provider_tool_message() {
        let translator = OpenAIBlockTranslator;
        let msg = json!({
            "role": "tool",
            "content": "result",
            "tool_call_id": "call_123"
        });
        let result = translator.from_provider_message(&msg).unwrap();
        assert_eq!(result["type"], "tool");
        assert_eq!(result["tool_call_id"], "call_123");
    }

    #[test]
    fn test_openai_content_block_text() {
        let translator = OpenAIBlockTranslator;
        let block = json!({"type": "text", "text": "hello"});
        let result = translator.to_provider_content_block(&block).unwrap();
        assert_eq!(result["type"], "text");
        assert_eq!(result["text"], "hello");
    }

    #[test]
    fn test_openai_content_block_image_url_string() {
        let translator = OpenAIBlockTranslator;
        let block = json!({"type": "image_url", "image_url": "https://example.com/img.png"});
        let result = translator.to_provider_content_block(&block).unwrap();
        assert_eq!(result["type"], "image_url");
        assert_eq!(result["image_url"]["url"], "https://example.com/img.png");
    }

    #[test]
    fn test_openai_from_content_block_image_url() {
        let translator = OpenAIBlockTranslator;
        let block = json!({
            "type": "image_url",
            "image_url": {"url": "https://example.com/img.png"}
        });
        let result = translator.from_provider_content_block(&block).unwrap();
        assert_eq!(result["type"], "image_url");
        assert_eq!(result["image_url"]["url"], "https://example.com/img.png");
    }

    // ========== Anthropic Tests ==========

    #[test]
    fn test_anthropic_provider_name() {
        let translator = AnthropicBlockTranslator;
        assert_eq!(translator.provider(), "anthropic");
    }

    #[test]
    fn test_anthropic_to_provider_human_message() {
        let translator = AnthropicBlockTranslator;
        let msg = json!({ "type": "human", "content": "Hello" });
        let result = translator.to_provider_message(&msg).unwrap();
        assert_eq!(result["role"], "user");
        assert_eq!(result["content"], "Hello");
    }

    #[test]
    fn test_anthropic_to_provider_ai_message() {
        let translator = AnthropicBlockTranslator;
        let msg = json!({ "type": "ai", "content": "Hi" });
        let result = translator.to_provider_message(&msg).unwrap();
        assert_eq!(result["role"], "assistant");
    }

    #[test]
    fn test_anthropic_to_provider_unsupported_role() {
        let translator = AnthropicBlockTranslator;
        let msg = json!({ "type": "system", "content": "You are helpful." });
        let result = translator.to_provider_message(&msg);
        assert!(result.is_err());
    }

    #[test]
    fn test_anthropic_from_provider_user_message() {
        let translator = AnthropicBlockTranslator;
        let msg = json!({ "role": "user", "content": "Hello" });
        let result = translator.from_provider_message(&msg).unwrap();
        assert_eq!(result["type"], "human");
    }

    #[test]
    fn test_anthropic_from_provider_assistant_message() {
        let translator = AnthropicBlockTranslator;
        let msg = json!({ "role": "assistant", "content": "Hi" });
        let result = translator.from_provider_message(&msg).unwrap();
        assert_eq!(result["type"], "ai");
    }

    #[test]
    fn test_anthropic_from_provider_with_tool_use_blocks() {
        let translator = AnthropicBlockTranslator;
        let msg = json!({
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "NYC"}}
            ]
        });
        let result = translator.from_provider_message(&msg).unwrap();
        assert_eq!(result["type"], "ai");
        let tool_calls = result["tool_calls"].as_array().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0]["name"], "get_weather");
        assert_eq!(tool_calls[0]["id"], "toolu_1");
        assert_eq!(tool_calls[0]["args"]["city"], "NYC");
    }

    #[test]
    fn test_anthropic_to_provider_content_block_text() {
        let translator = AnthropicBlockTranslator;
        let block = json!({"type": "text", "text": "hello"});
        let result = translator.to_provider_content_block(&block).unwrap();
        assert_eq!(result["type"], "text");
        assert_eq!(result["text"], "hello");
    }

    #[test]
    fn test_anthropic_to_provider_content_block_image_data_uri() {
        let translator = AnthropicBlockTranslator;
        let block = json!({
            "type": "image_url",
            "image_url": "data:image/png;base64,iVBORw0KGgo="
        });
        let result = translator.to_provider_content_block(&block).unwrap();
        assert_eq!(result["type"], "image");
        assert_eq!(result["source"]["type"], "base64");
        assert_eq!(result["source"]["media_type"], "image/png");
        assert_eq!(result["source"]["data"], "iVBORw0KGgo=");
    }

    #[test]
    fn test_anthropic_to_provider_content_block_image_url() {
        let translator = AnthropicBlockTranslator;
        let block = json!({
            "type": "image_url",
            "image_url": "https://example.com/img.png"
        });
        let result = translator.to_provider_content_block(&block).unwrap();
        assert_eq!(result["type"], "image");
        assert_eq!(result["source"]["type"], "url");
        assert_eq!(result["source"]["url"], "https://example.com/img.png");
    }

    #[test]
    fn test_anthropic_to_provider_content_block_tool_use() {
        let translator = AnthropicBlockTranslator;
        let block = json!({
            "type": "tool_use",
            "id": "toolu_1",
            "name": "get_weather",
            "input": {"city": "NYC"}
        });
        let result = translator.to_provider_content_block(&block).unwrap();
        assert_eq!(result["type"], "tool_use");
        assert_eq!(result["id"], "toolu_1");
        assert_eq!(result["name"], "get_weather");
        assert_eq!(result["input"]["city"], "NYC");
    }

    #[test]
    fn test_anthropic_to_provider_content_block_tool_result() {
        let translator = AnthropicBlockTranslator;
        let block = json!({
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": "Sunny, 72F"
        });
        let result = translator.to_provider_content_block(&block).unwrap();
        assert_eq!(result["type"], "tool_result");
        assert_eq!(result["tool_use_id"], "toolu_1");
        assert_eq!(result["content"], "Sunny, 72F");
    }

    #[test]
    fn test_anthropic_from_provider_content_block_base64_image() {
        let translator = AnthropicBlockTranslator;
        let block = json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": "abc123"
            }
        });
        let result = translator.from_provider_content_block(&block).unwrap();
        assert_eq!(result["type"], "image_url");
        assert_eq!(result["image_url"]["url"], "data:image/jpeg;base64,abc123");
    }

    #[test]
    fn test_anthropic_from_provider_content_block_url_image() {
        let translator = AnthropicBlockTranslator;
        let block = json!({
            "type": "image",
            "source": {
                "type": "url",
                "url": "https://example.com/img.png"
            }
        });
        let result = translator.from_provider_content_block(&block).unwrap();
        assert_eq!(result["type"], "image_url");
        assert_eq!(result["image_url"]["url"], "https://example.com/img.png");
    }

    // ========== Google GenAI Tests ==========

    #[test]
    fn test_google_provider_name() {
        let translator = GoogleGenAIBlockTranslator;
        assert_eq!(translator.provider(), "google_genai");
    }

    #[test]
    fn test_google_to_provider_human_message() {
        let translator = GoogleGenAIBlockTranslator;
        let msg = json!({ "type": "human", "content": "Hello" });
        let result = translator.to_provider_message(&msg).unwrap();
        assert_eq!(result["role"], "user");
        let parts = result["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["text"], "Hello");
    }

    #[test]
    fn test_google_to_provider_ai_message() {
        let translator = GoogleGenAIBlockTranslator;
        let msg = json!({ "type": "ai", "content": "Hi" });
        let result = translator.to_provider_message(&msg).unwrap();
        assert_eq!(result["role"], "model");
    }

    #[test]
    fn test_google_to_provider_system_message() {
        let translator = GoogleGenAIBlockTranslator;
        let msg = json!({ "type": "system", "content": "Be helpful." });
        let result = translator.to_provider_message(&msg).unwrap();
        // Google maps system to user (system_instruction handled separately)
        assert_eq!(result["role"], "user");
    }

    #[test]
    fn test_google_to_provider_content_blocks() {
        let translator = GoogleGenAIBlockTranslator;
        let msg = json!({
            "type": "human",
            "content": [
                {"type": "text", "text": "Describe this:"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
            ]
        });
        let result = translator.to_provider_message(&msg).unwrap();
        let parts = result["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["text"], "Describe this:");
        assert!(parts[1].get("inline_data").is_some());
    }

    #[test]
    fn test_google_from_provider_user_message() {
        let translator = GoogleGenAIBlockTranslator;
        let msg = json!({
            "role": "user",
            "parts": [{"text": "Hello"}]
        });
        let result = translator.from_provider_message(&msg).unwrap();
        assert_eq!(result["type"], "human");
        assert_eq!(result["content"], "Hello");
    }

    #[test]
    fn test_google_from_provider_model_message() {
        let translator = GoogleGenAIBlockTranslator;
        let msg = json!({
            "role": "model",
            "parts": [{"text": "Hi there"}]
        });
        let result = translator.from_provider_message(&msg).unwrap();
        assert_eq!(result["type"], "ai");
        assert_eq!(result["content"], "Hi there");
    }

    #[test]
    fn test_google_from_provider_multiple_parts() {
        let translator = GoogleGenAIBlockTranslator;
        let msg = json!({
            "role": "model",
            "parts": [
                {"text": "Part 1"},
                {"text": "Part 2"}
            ]
        });
        let result = translator.from_provider_message(&msg).unwrap();
        assert_eq!(result["type"], "ai");
        let content = result["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
    }

    #[test]
    fn test_google_from_provider_inline_data() {
        let translator = GoogleGenAIBlockTranslator;
        let block = json!({
            "inline_data": {
                "mime_type": "image/png",
                "data": "iVBORw0KGgo="
            }
        });
        let result = translator.from_provider_content_block(&block).unwrap();
        assert_eq!(result["type"], "image_url");
        assert_eq!(
            result["image_url"]["url"],
            "data:image/png;base64,iVBORw0KGgo="
        );
    }

    #[test]
    fn test_google_from_provider_function_call() {
        let translator = GoogleGenAIBlockTranslator;
        let block = json!({
            "functionCall": {
                "name": "get_weather",
                "args": {"city": "NYC"}
            }
        });
        let result = translator.from_provider_content_block(&block).unwrap();
        assert_eq!(result["type"], "tool_use");
        assert_eq!(result["name"], "get_weather");
        assert_eq!(result["input"]["city"], "NYC");
    }

    #[test]
    fn test_google_to_provider_text_block() {
        let translator = GoogleGenAIBlockTranslator;
        let block = json!({"type": "text", "text": "hello"});
        let result = translator.to_provider_content_block(&block).unwrap();
        assert_eq!(result["text"], "hello");
        assert!(result.get("type").is_none());
    }

    // ========== get_translator Tests ==========

    #[test]
    fn test_get_translator_openai() {
        let t = get_translator("openai");
        assert!(t.is_some());
        assert_eq!(t.unwrap().provider(), "openai");
    }

    #[test]
    fn test_get_translator_anthropic() {
        let t = get_translator("anthropic");
        assert!(t.is_some());
        assert_eq!(t.unwrap().provider(), "anthropic");
    }

    #[test]
    fn test_get_translator_google_genai() {
        let t = get_translator("google_genai");
        assert!(t.is_some());
        assert_eq!(t.unwrap().provider(), "google_genai");
    }

    #[test]
    fn test_get_translator_google_alias() {
        let t = get_translator("google");
        assert!(t.is_some());
        assert_eq!(t.unwrap().provider(), "google_genai");
    }

    #[test]
    fn test_get_translator_unknown() {
        let t = get_translator("unknown_provider");
        assert!(t.is_none());
    }
}
