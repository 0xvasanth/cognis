//! Prompt caching middleware — injects cache control metadata into agent state
//! before model calls, enabling Anthropic-style prompt caching for repeated
//! prefixes (system prompts, conversation history).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::middleware::{AgentState, Middleware, Result};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the [`PromptCachingMiddleware`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptCachingConfig {
    /// The cache type identifier (e.g. `"ephemeral"`).
    pub cache_type: String,
    /// Minimum number of messages required before cache control is injected.
    pub min_messages: usize,
    /// Whether to mark the system prompt for caching.
    pub cache_system_prompt: bool,
}

impl Default for PromptCachingConfig {
    fn default() -> Self {
        Self {
            cache_type: "ephemeral".to_string(),
            min_messages: 2,
            cache_system_prompt: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------

/// Middleware that injects `_cache_control` metadata into agent state before
/// model calls, signalling the provider to cache the prompt prefix.
///
/// **Note:** Chat model providers (e.g. `ChatAnthropic`) must read the
/// `_cache_control` key from the state and include the appropriate cache
/// headers in their API requests for this middleware to have effect.
pub struct PromptCachingMiddleware {
    config: PromptCachingConfig,
}

impl PromptCachingMiddleware {
    /// Create a new middleware with default configuration.
    pub fn new() -> Self {
        Self {
            config: PromptCachingConfig::default(),
        }
    }

    /// Create a new middleware with the given configuration.
    pub fn with_config(config: PromptCachingConfig) -> Self {
        Self { config }
    }

    /// Create a new middleware with a custom minimum message threshold.
    pub fn with_min_messages(min: usize) -> Self {
        Self {
            config: PromptCachingConfig {
                min_messages: min,
                ..Default::default()
            },
        }
    }
}

impl Default for PromptCachingMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Middleware for PromptCachingMiddleware {
    fn name(&self) -> &str {
        "prompt_caching"
    }

    async fn before_model(&self, state: &mut AgentState) -> Result<()> {
        let message_count = state
            .get("messages")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);

        if message_count < self.config.min_messages {
            return Ok(());
        }

        let cache_control = json!({
            "type": self.config.cache_type,
            "cache_system_prompt": self.config.cache_system_prompt,
            "message_count": message_count,
            "_injected_by": "prompt_caching_middleware",
        });

        if let Some(obj) = state.as_object_mut() {
            obj.insert("_cache_control".to_string(), cache_control);
        }

        Ok(())
    }

    async fn after_model(&self, state: &mut AgentState) -> Result<()> {
        // Only remove _cache_control if it was injected by this middleware
        // (identified by our private sentinel). This avoids clobbering
        // caller-provided cache control state.
        let was_injected_by_us = state
            .get("_cache_control")
            .and_then(|v| v.get("_injected_by"))
            .and_then(|v| v.as_str())
            == Some("prompt_caching_middleware");

        if was_injected_by_us {
            if let Some(obj) = state.as_object_mut() {
                obj.remove("_cache_control");
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_inject_when_enough_messages() {
        let mw = PromptCachingMiddleware::new();
        let mut state = json!({
            "messages": [
                {"type": "system", "content": "You are helpful."},
                {"type": "human", "content": "Hello!"}
            ]
        });

        mw.before_model(&mut state).await.unwrap();

        let cache = state
            .get("_cache_control")
            .expect("should have _cache_control");
        assert_eq!(cache["type"], "ephemeral");
        assert_eq!(cache["cache_system_prompt"], true);
        assert_eq!(cache["message_count"], 2);
    }

    #[tokio::test]
    async fn test_skip_when_too_few_messages() {
        let mw = PromptCachingMiddleware::with_min_messages(3);
        let mut state = json!({
            "messages": [
                {"type": "system", "content": "You are helpful."},
                {"type": "human", "content": "Hello!"}
            ]
        });

        mw.before_model(&mut state).await.unwrap();

        assert!(
            state.get("_cache_control").is_none(),
            "should not inject _cache_control when message count is below threshold"
        );
    }

    #[tokio::test]
    async fn test_cleanup_after_model() {
        let mw = PromptCachingMiddleware::new();
        let mut state = json!({
            "messages": [
                {"type": "system", "content": "You are helpful."},
                {"type": "human", "content": "Hello!"}
            ]
        });

        mw.before_model(&mut state).await.unwrap();
        assert!(state.get("_cache_control").is_some());

        mw.after_model(&mut state).await.unwrap();
        assert!(
            state.get("_cache_control").is_none(),
            "_cache_control should be removed after model call"
        );
    }

    #[tokio::test]
    async fn test_custom_config() {
        let config = PromptCachingConfig {
            cache_type: "persistent".to_string(),
            min_messages: 1,
            cache_system_prompt: false,
        };
        let mw = PromptCachingMiddleware::with_config(config);

        let mut state = json!({
            "messages": [
                {"type": "human", "content": "Hello!"}
            ]
        });

        mw.before_model(&mut state).await.unwrap();

        let cache = state
            .get("_cache_control")
            .expect("should have _cache_control");
        assert_eq!(cache["type"], "persistent");
        assert_eq!(cache["cache_system_prompt"], false);
        assert_eq!(cache["message_count"], 1);
    }

    #[test]
    fn test_default_config() {
        let config = PromptCachingConfig::default();
        assert_eq!(config.cache_type, "ephemeral");
        assert_eq!(config.min_messages, 2);
        assert!(config.cache_system_prompt);

        let mw = PromptCachingMiddleware::default();
        assert_eq!(mw.name(), "prompt_caching");
        assert_eq!(mw.config.cache_type, "ephemeral");
    }
}
