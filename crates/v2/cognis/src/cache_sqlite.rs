//! Sqlite-backed [`CacheBackend`] (feature `cache-sqlite`).
//!
//! Pairs with [`cognis2_core::wrappers::Cache`] for persistent caching of
//! any `Runnable` output across runs. Useful for caching expensive LLM
//! calls in development / batch runs.
//!
//! ```ignore
//! use std::sync::Arc;
//! use cognis2::{cache_sqlite::SqliteCache, RunnableExt};
//!
//! let cache = Arc::new(SqliteCache::connect("sqlite::memory:").await?);
//! let cached = client.with_disk_cache(cache, |msgs| stable_key(msgs));
//! ```

use std::marker::PhantomData;

use async_trait::async_trait;
use serde::{de::DeserializeOwned, Serialize};
use sqlx::sqlite::{SqlitePool, SqlitePoolOptions};
use sqlx::Row;

use cognis2_core::wrappers::CacheBackend;
use cognis2_core::{CognisError, Result};

/// Sqlite-backed cache. Generic over key (`ToString`) and value
/// (`Serialize + DeserializeOwned`). Values are stored as JSON text.
pub struct SqliteCache<K, V> {
    pool: SqlitePool,
    table: String,
    _phantom: PhantomData<fn(K) -> V>,
}

impl<K, V> SqliteCache<K, V>
where
    K: ToString + Send + Sync + 'static,
    V: Serialize + DeserializeOwned + Send + Sync + 'static,
{
    /// Connect to a sqlite database (file path or `sqlite::memory:`) and
    /// ensure the cache table exists.
    pub async fn connect(url: &str) -> Result<Self> {
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(url)
            .await
            .map_err(|e| CognisError::Configuration(format!("sqlite cache connect: {e}")))?;
        let cache = Self {
            pool,
            table: "cognis2_cache".into(),
            _phantom: PhantomData,
        };
        cache.ensure_table().await?;
        Ok(cache)
    }

    /// Override the table name (default `cognis2_cache`).
    pub fn with_table(mut self, table: impl Into<String>) -> Self {
        self.table = table.into();
        self
    }

    async fn ensure_table(&self) -> Result<()> {
        let stmt = format!(
            "CREATE TABLE IF NOT EXISTS {table} (
                 key   TEXT PRIMARY KEY,
                 value TEXT NOT NULL
             )",
            table = self.table,
        );
        sqlx::query(&stmt)
            .execute(&self.pool)
            .await
            .map_err(|e| CognisError::Internal(format!("sqlite cache create table: {e}")))?;
        Ok(())
    }
}

#[async_trait]
impl<K, V> CacheBackend<K, V> for SqliteCache<K, V>
where
    K: ToString + Send + Sync + 'static,
    V: Serialize + DeserializeOwned + Clone + Send + Sync + 'static,
{
    async fn get(&self, key: &K) -> Option<V> {
        let stmt = format!(
            "SELECT value FROM {table} WHERE key = ?",
            table = self.table
        );
        let row = sqlx::query(&stmt)
            .bind(key.to_string())
            .fetch_optional(&self.pool)
            .await
            .ok()??;
        let json: String = row.try_get("value").ok()?;
        serde_json::from_str(&json).ok()
    }

    async fn set(&self, key: K, value: V) {
        let json = match serde_json::to_string(&value) {
            Ok(j) => j,
            Err(_) => return,
        };
        let stmt = format!(
            "INSERT OR REPLACE INTO {table} (key, value) VALUES (?, ?)",
            table = self.table,
        );
        let _ = sqlx::query(&stmt)
            .bind(key.to_string())
            .bind(json)
            .execute(&self.pool)
            .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn roundtrip() {
        let cache: SqliteCache<String, String> =
            SqliteCache::connect("sqlite::memory:").await.unwrap();
        cache.set("k1".into(), "v1".into()).await;
        assert_eq!(cache.get(&"k1".to_string()).await, Some("v1".into()));
        assert_eq!(cache.get(&"missing".to_string()).await, None);
    }

    #[tokio::test]
    async fn overwrite_replaces() {
        let cache: SqliteCache<String, u64> =
            SqliteCache::connect("sqlite::memory:").await.unwrap();
        cache.set("k".into(), 1).await;
        cache.set("k".into(), 2).await;
        assert_eq!(cache.get(&"k".to_string()).await, Some(2));
    }
}
