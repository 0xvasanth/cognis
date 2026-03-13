//! SQLite-backed LLM response cache.
//!
//! [`SqliteCache`] persists [`ChatResult`] entries in a SQLite database using
//! `sqlx`. The schema is created automatically on construction.

use async_trait::async_trait;
use sqlx::sqlite::{SqlitePool, SqlitePoolOptions};

use cognis_core::outputs::ChatResult;

use super::LlmCache;

/// Persistent LLM response cache backed by SQLite.
///
/// The table `llm_cache` is created automatically when the cache is
/// constructed via [`SqliteCache::new`].
///
/// # Example
///
/// ```rust,ignore
/// use cognis::cache::SqliteCache;
///
/// let cache = SqliteCache::new("sqlite::memory:").await.unwrap();
/// ```
pub struct SqliteCache {
    pool: SqlitePool,
}

impl SqliteCache {
    /// Create a new SQLite cache, connecting to the given database URL.
    ///
    /// The `llm_cache` table is created if it does not already exist.
    ///
    /// # Arguments
    /// * `database_url` — A SQLite connection string such as
    ///   `"sqlite::memory:"` or `"sqlite:///tmp/cache.db"`.
    pub async fn new(database_url: &str) -> Result<Self, sqlx::Error> {
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(database_url)
            .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key   TEXT PRIMARY KEY NOT NULL,
                result_json TEXT NOT NULL,
                created_at  INTEGER NOT NULL DEFAULT (unixepoch())
            )
            "#,
        )
        .execute(&pool)
        .await?;

        Ok(Self { pool })
    }

    /// Create a new SQLite cache from an existing connection pool.
    ///
    /// The `llm_cache` table is created if it does not already exist.
    pub async fn from_pool(pool: SqlitePool) -> Result<Self, sqlx::Error> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key   TEXT PRIMARY KEY NOT NULL,
                result_json TEXT NOT NULL,
                created_at  INTEGER NOT NULL DEFAULT (unixepoch())
            )
            "#,
        )
        .execute(&pool)
        .await?;

        Ok(Self { pool })
    }
}

#[async_trait]
impl LlmCache for SqliteCache {
    async fn get(&self, key: &str) -> Option<ChatResult> {
        let row: Option<(String,)> =
            sqlx::query_as("SELECT result_json FROM llm_cache WHERE cache_key = ?")
                .bind(key)
                .fetch_optional(&self.pool)
                .await
                .ok()?;

        let (json,) = row?;
        serde_json::from_str(&json).ok()
    }

    async fn put(&self, key: &str, result: &ChatResult) {
        let json = match serde_json::to_string(result) {
            Ok(j) => j,
            Err(_) => return,
        };

        let _ = sqlx::query(
            r#"
            INSERT INTO llm_cache (cache_key, result_json)
            VALUES (?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET result_json = excluded.result_json
            "#,
        )
        .bind(key)
        .bind(&json)
        .execute(&self.pool)
        .await;
    }

    async fn clear(&self) {
        let _ = sqlx::query("DELETE FROM llm_cache")
            .execute(&self.pool)
            .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cognis_core::messages::AIMessage;
    use cognis_core::outputs::{ChatGeneration, ChatResult};

    fn make_result(text: &str) -> ChatResult {
        ChatResult {
            generations: vec![ChatGeneration::new(AIMessage::new(text))],
            llm_output: None,
        }
    }

    #[tokio::test]
    async fn test_sqlite_put_and_get() {
        let cache = SqliteCache::new("sqlite::memory:").await.unwrap();
        let result = make_result("cached response");

        cache.put("key1", &result).await;
        let got = cache.get("key1").await;

        assert!(got.is_some());
        assert_eq!(got.unwrap(), result);
    }

    #[tokio::test]
    async fn test_sqlite_get_miss() {
        let cache = SqliteCache::new("sqlite::memory:").await.unwrap();
        assert!(cache.get("nonexistent").await.is_none());
    }

    #[tokio::test]
    async fn test_sqlite_clear() {
        let cache = SqliteCache::new("sqlite::memory:").await.unwrap();
        cache.put("a", &make_result("a")).await;
        cache.put("b", &make_result("b")).await;

        cache.clear().await;

        assert!(cache.get("a").await.is_none());
        assert!(cache.get("b").await.is_none());
    }

    #[tokio::test]
    async fn test_sqlite_upsert() {
        let cache = SqliteCache::new("sqlite::memory:").await.unwrap();

        cache.put("k", &make_result("v1")).await;
        cache.put("k", &make_result("v2")).await;

        let got = cache.get("k").await.unwrap();
        assert_eq!(got.generations[0].text, "v2");
    }
}
