//! LLM response caching layer for chat models.
//!
//! Provides [`CachedChatModel`], a chat model wrapper that caches responses
//! from any [`BaseChatModel`] implementation using a pluggable [`LlmCache`]
//! backend.
//!
//! See the [`crate::cache`] module for available cache backends.

use std::sync::Arc;

use async_trait::async_trait;

use cognis_core::error::Result;
use cognis_core::language_models::chat_model::{
    BaseChatModel, ChatStream, ModelProfile, ToolChoice,
};
use cognis_core::messages::Message;
use cognis_core::outputs::ChatResult;
use cognis_core::tools::ToolSchema;

use crate::cache::{compute_cache_key, LlmCache};

/// A chat model wrapper that caches responses from an inner model.
///
/// `_generate` checks the cache before calling the inner model. On a miss the
/// result is stored for future calls with identical inputs. Streaming requests
/// are passed through to the inner model without caching.
///
/// # Example
///
/// ```rust,ignore
/// use cognis::cache::InMemoryCache;
/// use cognis::chat_models::cached::CachedChatModel;
/// use std::sync::Arc;
///
/// let cache = Arc::new(InMemoryCache::builder().max_size(100).build());
/// let cached_model = CachedChatModel::new(Box::new(my_model), cache);
/// ```
pub struct CachedChatModel {
    inner: Box<dyn BaseChatModel>,
    cache: Arc<dyn LlmCache>,
}

impl CachedChatModel {
    /// Wrap an existing chat model with a cache backend.
    pub fn new(inner: Box<dyn BaseChatModel>, cache: Arc<dyn LlmCache>) -> Self {
        Self { inner, cache }
    }

    /// Return a reference to the underlying cache.
    pub fn cache(&self) -> &Arc<dyn LlmCache> {
        &self.cache
    }
}

#[async_trait]
impl BaseChatModel for CachedChatModel {
    async fn _generate(&self, messages: &[Message], stop: Option<&[String]>) -> Result<ChatResult> {
        let key = compute_cache_key(messages, stop);

        // Return cached result on hit.
        if let Some(cached) = self.cache.get(&key).await {
            return Ok(cached);
        }

        // Miss — call through and cache the result.
        let result = self.inner._generate(messages, stop).await?;
        self.cache.put(&key, &result).await;
        Ok(result)
    }

    fn llm_type(&self) -> &str {
        // Leak is acceptable for a type identifier that lives for the program's
        // duration.
        let s = format!("cached({})", self.inner.llm_type());
        Box::leak(s.into_boxed_str())
    }

    async fn _stream(&self, messages: &[Message], stop: Option<&[String]>) -> Result<ChatStream> {
        // Streams are not cached; delegate directly.
        self.inner._stream(messages, stop).await
    }

    fn bind_tools(
        &self,
        tools: &[ToolSchema],
        tool_choice: Option<ToolChoice>,
    ) -> Result<Box<dyn BaseChatModel>> {
        let inner_with_tools = self.inner.bind_tools(tools, tool_choice)?;
        Ok(Box::new(CachedChatModel {
            inner: inner_with_tools,
            cache: Arc::clone(&self.cache),
        }))
    }

    fn profile(&self) -> ModelProfile {
        self.inner.profile()
    }

    fn get_num_tokens_from_messages(&self, messages: &[Message]) -> usize {
        self.inner.get_num_tokens_from_messages(messages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::InMemoryCache;
    use cognis_core::messages::{AIMessage, HumanMessage};
    use cognis_core::outputs::ChatGeneration;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// A fake chat model that counts how many times `_generate` is called.
    struct FakeModel {
        call_count: Arc<AtomicUsize>,
    }

    impl FakeModel {
        fn new() -> (Self, Arc<AtomicUsize>) {
            let count = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    call_count: Arc::clone(&count),
                },
                count,
            )
        }
    }

    #[async_trait]
    impl BaseChatModel for FakeModel {
        async fn _generate(
            &self,
            _messages: &[Message],
            _stop: Option<&[String]>,
        ) -> Result<ChatResult> {
            let n = self.call_count.fetch_add(1, Ordering::SeqCst);
            let text = format!("response #{n}");
            Ok(ChatResult {
                generations: vec![ChatGeneration::new(AIMessage::new(&text))],
                llm_output: None,
            })
        }

        fn llm_type(&self) -> &str {
            "fake"
        }
    }

    fn human(text: &str) -> Message {
        Message::Human(HumanMessage::new(text))
    }

    #[tokio::test]
    async fn test_cache_hit_returns_cached_result() {
        let (model, call_count) = FakeModel::new();
        let cache = Arc::new(InMemoryCache::new());
        let cached = CachedChatModel::new(Box::new(model), cache);

        let msgs = vec![human("hello")];
        let r1 = cached._generate(&msgs, None).await.unwrap();
        let r2 = cached._generate(&msgs, None).await.unwrap();

        assert_eq!(r1, r2);
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            1,
            "inner model called only once"
        );
    }

    #[tokio::test]
    async fn test_cache_miss_different_messages() {
        let (model, call_count) = FakeModel::new();
        let cache = Arc::new(InMemoryCache::new());
        let cached = CachedChatModel::new(Box::new(model), cache);

        let r1 = cached._generate(&[human("hello")], None).await.unwrap();
        let r2 = cached._generate(&[human("world")], None).await.unwrap();

        assert_ne!(r1, r2);
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_cached_model_llm_type() {
        let (model, _) = FakeModel::new();
        let cache = Arc::new(InMemoryCache::new());
        let cached = CachedChatModel::new(Box::new(model), cache);
        assert_eq!(cached.llm_type(), "cached(fake)");
    }

    #[tokio::test]
    async fn test_cached_model_with_ttl_expiry() {
        let (model, call_count) = FakeModel::new();
        let cache = Arc::new(
            InMemoryCache::builder()
                .ttl(Duration::from_millis(50))
                .build(),
        );
        let cached = CachedChatModel::new(Box::new(model), cache);

        let msgs = vec![human("hi")];
        let _ = cached._generate(&msgs, None).await.unwrap();
        assert_eq!(call_count.load(Ordering::SeqCst), 1);

        // Wait for expiry.
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Should call the inner model again.
        let _ = cached._generate(&msgs, None).await.unwrap();
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_cached_model_with_lru_eviction() {
        let (model, call_count) = FakeModel::new();
        let cache = Arc::new(InMemoryCache::builder().max_size(2).build());
        let cached = CachedChatModel::new(Box::new(model), cache);

        // Fill cache with 2 entries.
        let _ = cached._generate(&[human("a")], None).await.unwrap();
        let _ = cached._generate(&[human("b")], None).await.unwrap();
        assert_eq!(call_count.load(Ordering::SeqCst), 2);

        // Access "a" to make it recently used.
        let _ = cached._generate(&[human("a")], None).await.unwrap();
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            2,
            "a should be a cache hit"
        );

        // Insert "c" — should evict "b".
        let _ = cached._generate(&[human("c")], None).await.unwrap();
        assert_eq!(call_count.load(Ordering::SeqCst), 3);

        // "b" should now be a miss.
        let _ = cached._generate(&[human("b")], None).await.unwrap();
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            4,
            "b should be evicted and re-fetched"
        );
    }

    #[tokio::test]
    async fn test_stop_sequences_produce_different_cache_keys() {
        let (model, call_count) = FakeModel::new();
        let cache = Arc::new(InMemoryCache::new());
        let cached = CachedChatModel::new(Box::new(model), cache);

        let msgs = vec![human("hello")];
        let _ = cached._generate(&msgs, None).await.unwrap();
        let _ = cached
            ._generate(&msgs, Some(&["stop".to_string()]))
            .await
            .unwrap();

        assert_eq!(
            call_count.load(Ordering::SeqCst),
            2,
            "different stop seqs = different keys"
        );
    }
}
