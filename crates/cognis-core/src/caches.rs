use async_trait::async_trait;
use std::collections::HashMap;

use crate::error::Result;
use crate::outputs::Generation;

/// Interface for a caching layer for LLMs.
#[async_trait]
pub trait BaseCache: Send + Sync {
    /// Look up based on prompt and llm_string.
    async fn lookup(&self, prompt: &str, llm_string: &str) -> Result<Option<Vec<Generation>>>;

    /// Update cache based on prompt and llm_string.
    async fn update(
        &self,
        prompt: &str,
        llm_string: &str,
        return_val: Vec<Generation>,
    ) -> Result<()>;

    /// Clear the cache.
    async fn clear(&self) -> Result<()>;
}

/// In-memory cache implementation.
#[derive(Debug, Default)]
pub struct InMemoryCache {
    cache: tokio::sync::RwLock<HashMap<(String, String), Vec<Generation>>>,
    maxsize: Option<usize>,
}

impl InMemoryCache {
    pub fn new() -> Self {
        Self {
            cache: tokio::sync::RwLock::new(HashMap::new()),
            maxsize: None,
        }
    }

    pub fn with_maxsize(maxsize: usize) -> Self {
        Self {
            cache: tokio::sync::RwLock::new(HashMap::new()),
            maxsize: Some(maxsize),
        }
    }
}

#[async_trait]
impl BaseCache for InMemoryCache {
    async fn lookup(&self, prompt: &str, llm_string: &str) -> Result<Option<Vec<Generation>>> {
        let cache = self.cache.read().await;
        Ok(cache
            .get(&(prompt.to_string(), llm_string.to_string()))
            .cloned())
    }

    async fn update(
        &self,
        prompt: &str,
        llm_string: &str,
        return_val: Vec<Generation>,
    ) -> Result<()> {
        let mut cache = self.cache.write().await;
        if let Some(max) = self.maxsize {
            if cache.len() >= max {
                if let Some(first_key) = cache.keys().next().cloned() {
                    cache.remove(&first_key);
                }
            }
        }
        cache.insert((prompt.to_string(), llm_string.to_string()), return_val);
        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        let mut cache = self.cache.write().await;
        cache.clear();
        Ok(())
    }
}
