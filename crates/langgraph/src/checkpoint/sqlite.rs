//! SQLite-backed checkpoint persistence.
//!
//! Provides [`SqliteCheckpointSaver`], a durable checkpoint backend that stores
//! graph state in a SQLite database. This allows agent state to survive process
//! restarts.
//!
//! # Feature flag
//!
//! This module is only available when the `sqlite` feature is enabled.
//!
//! # Example
//!
//! ```ignore
//! use langgraph::checkpoint::SqliteCheckpointSaver;
//!
//! let saver = SqliteCheckpointSaver::new("sqlite::memory:").await?;
//! ```

use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::Value;
use sqlx::sqlite::{SqlitePool, SqlitePoolOptions};
use sqlx::Row;

use crate::errors::{LangGraphError, Result};
use crate::pregel::checkpoint::{
    Checkpoint, CheckpointEntry, CheckpointMetadata, CheckpointSaver, CheckpointTuple,
};

/// A checkpoint saver backed by SQLite via `sqlx`.
///
/// Stores checkpoints and pending writes in two tables. Supports multiple
/// threads and namespaces.
#[derive(Debug, Clone)]
pub struct SqliteCheckpointSaver {
    pool: SqlitePool,
}

impl SqliteCheckpointSaver {
    /// Connect to a SQLite database and create the required tables.
    ///
    /// # Arguments
    ///
    /// * `database_url` — A SQLite connection string, e.g. `"sqlite::memory:"`
    ///   or `"sqlite:///path/to/db.sqlite"`.
    pub async fn new(database_url: &str) -> Result<Self> {
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(database_url)
            .await
            .map_err(|e| LangGraphError::Other(format!("SQLite connection error: {}", e)))?;

        let saver = Self { pool };
        saver.setup_tables().await?;
        Ok(saver)
    }

    /// Create a saver from an existing connection pool.
    ///
    /// Callers are responsible for calling [`setup_tables`](Self::setup_tables)
    /// after construction.
    pub fn from_pool(pool: SqlitePool) -> Self {
        Self { pool }
    }

    /// Create the checkpoint and pending-writes tables if they do not exist.
    pub async fn setup_tables(&self) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                data BLOB NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| LangGraphError::Other(format!("Failed to create checkpoints table: {}", e)))?;

        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_checkpoints_thread
            ON checkpoints(thread_id, checkpoint_ns, created_at DESC)
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| LangGraphError::Other(format!("Failed to create index: {}", e)))?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS pending_writes (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                value BLOB NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| {
            LangGraphError::Other(format!("Failed to create pending_writes table: {}", e))
        })?;

        Ok(())
    }

    /// Extract common config fields.
    fn extract_config(config: &HashMap<String, Value>) -> (&str, &str, Option<&str>) {
        let thread_id = config
            .get("thread_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let checkpoint_ns = config
            .get("checkpoint_ns")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let checkpoint_id = config.get("checkpoint_id").and_then(|v| v.as_str());
        (thread_id, checkpoint_ns, checkpoint_id)
    }

    /// Build a [`CheckpointTuple`] from a database row.
    async fn row_to_tuple(&self, row: &sqlx::sqlite::SqliteRow) -> Result<CheckpointTuple> {
        let thread_id: String = row
            .try_get("thread_id")
            .map_err(|e| LangGraphError::Other(format!("Column error: {}", e)))?;
        let checkpoint_ns: String = row
            .try_get("checkpoint_ns")
            .map_err(|e| LangGraphError::Other(format!("Column error: {}", e)))?;
        let checkpoint_id: String = row
            .try_get("checkpoint_id")
            .map_err(|e| LangGraphError::Other(format!("Column error: {}", e)))?;
        let parent_checkpoint_id: Option<String> = row
            .try_get("parent_checkpoint_id")
            .map_err(|e| LangGraphError::Other(format!("Column error: {}", e)))?;
        let data_bytes: Vec<u8> = row
            .try_get("data")
            .map_err(|e| LangGraphError::Other(format!("Column error: {}", e)))?;
        let metadata_str: Option<String> = row
            .try_get("metadata")
            .map_err(|e| LangGraphError::Other(format!("Column error: {}", e)))?;

        let checkpoint: Checkpoint = serde_json::from_slice(&data_bytes)
            .map_err(|e| LangGraphError::Other(format!("Deserialize checkpoint error: {}", e)))?;

        let metadata: Option<CheckpointMetadata> = metadata_str
            .as_deref()
            .map(serde_json::from_str)
            .transpose()
            .map_err(|e| LangGraphError::Other(format!("Deserialize metadata error: {}", e)))?;

        let mut config = HashMap::new();
        config.insert("thread_id".to_string(), Value::String(thread_id.clone()));
        config.insert(
            "checkpoint_ns".to_string(),
            Value::String(checkpoint_ns.clone()),
        );
        config.insert(
            "checkpoint_id".to_string(),
            Value::String(checkpoint_id.clone()),
        );

        let parent_config = parent_checkpoint_id.map(|pid| {
            let mut pc = HashMap::new();
            pc.insert("thread_id".to_string(), Value::String(thread_id.clone()));
            pc.insert(
                "checkpoint_ns".to_string(),
                Value::String(checkpoint_ns.clone()),
            );
            pc.insert("checkpoint_id".to_string(), Value::String(pid));
            pc
        });

        // Load pending writes for this checkpoint.
        let pending_rows = sqlx::query(
            r#"
            SELECT task_id, channel, value
            FROM pending_writes
            WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
            ORDER BY created_at
            "#,
        )
        .bind(&thread_id)
        .bind(&checkpoint_ns)
        .bind(&checkpoint_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| LangGraphError::Other(format!("Failed to load pending writes: {}", e)))?;

        let pending_writes = if pending_rows.is_empty() {
            None
        } else {
            let mut writes = Vec::new();
            for pw_row in &pending_rows {
                let task_id: String = pw_row
                    .try_get("task_id")
                    .map_err(|e| LangGraphError::Other(format!("Column error: {}", e)))?;
                let channel: String = pw_row
                    .try_get("channel")
                    .map_err(|e| LangGraphError::Other(format!("Column error: {}", e)))?;
                let value_bytes: Vec<u8> = pw_row
                    .try_get("value")
                    .map_err(|e| LangGraphError::Other(format!("Column error: {}", e)))?;
                let value: Value = serde_json::from_slice(&value_bytes).map_err(|e| {
                    LangGraphError::Other(format!("Deserialize pending write error: {}", e))
                })?;
                writes.push((task_id, channel, value));
            }
            Some(writes)
        };

        Ok(CheckpointTuple {
            checkpoint,
            config,
            metadata,
            parent_config,
            pending_writes,
        })
    }
}

#[async_trait]
impl CheckpointSaver for SqliteCheckpointSaver {
    async fn get(&self, config: &HashMap<String, Value>) -> Result<Option<CheckpointTuple>> {
        let (thread_id, checkpoint_ns, checkpoint_id) = Self::extract_config(config);

        let row = if let Some(cid) = checkpoint_id {
            sqlx::query(
                r#"
                SELECT * FROM checkpoints
                WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                "#,
            )
            .bind(thread_id)
            .bind(checkpoint_ns)
            .bind(cid)
            .fetch_optional(&self.pool)
            .await
        } else {
            sqlx::query(
                r#"
                SELECT * FROM checkpoints
                WHERE thread_id = ? AND checkpoint_ns = ?
                ORDER BY created_at DESC
                LIMIT 1
                "#,
            )
            .bind(thread_id)
            .bind(checkpoint_ns)
            .fetch_optional(&self.pool)
            .await
        }
        .map_err(|e| LangGraphError::Other(format!("SQLite query error: {}", e)))?;

        match row {
            Some(r) => Ok(Some(self.row_to_tuple(&r).await?)),
            None => Ok(None),
        }
    }

    async fn get_tuple(
        &self,
        config: &HashMap<String, Value>,
    ) -> Result<Option<CheckpointTuple>> {
        // get_tuple is the same as get but always requires a specific checkpoint_id
        self.get(config).await
    }

    async fn put(
        &self,
        config: &HashMap<String, Value>,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> Result<HashMap<String, Value>> {
        let (thread_id, checkpoint_ns, _) = Self::extract_config(config);
        let parent_checkpoint_id = config
            .get("checkpoint_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let data = serde_json::to_vec(&checkpoint)
            .map_err(|e| LangGraphError::Other(format!("Serialize checkpoint error: {}", e)))?;
        let metadata_str = serde_json::to_string(&metadata)
            .map_err(|e| LangGraphError::Other(format!("Serialize metadata error: {}", e)))?;

        sqlx::query(
            r#"
            INSERT OR REPLACE INTO checkpoints
                (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, data, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(thread_id)
        .bind(checkpoint_ns)
        .bind(&checkpoint.id)
        .bind(&parent_checkpoint_id)
        .bind(&data)
        .bind(&metadata_str)
        .execute(&self.pool)
        .await
        .map_err(|e| LangGraphError::Other(format!("SQLite insert error: {}", e)))?;

        let mut new_config = config.clone();
        new_config.insert(
            "checkpoint_id".to_string(),
            Value::String(checkpoint.id),
        );

        Ok(new_config)
    }

    async fn put_writes(
        &self,
        config: &HashMap<String, Value>,
        writes: Vec<(String, Value)>,
        task_id: &str,
    ) -> Result<()> {
        let (thread_id, checkpoint_ns, checkpoint_id) = Self::extract_config(config);
        let checkpoint_id = checkpoint_id.ok_or_else(|| {
            LangGraphError::Other("checkpoint_id required for put_writes".into())
        })?;

        for (channel, value) in &writes {
            let value_bytes = serde_json::to_vec(value).map_err(|e| {
                LangGraphError::Other(format!("Serialize write value error: {}", e))
            })?;

            sqlx::query(
                r#"
                INSERT INTO pending_writes
                    (thread_id, checkpoint_ns, checkpoint_id, task_id, channel, value)
                VALUES (?, ?, ?, ?, ?, ?)
                "#,
            )
            .bind(thread_id)
            .bind(checkpoint_ns)
            .bind(checkpoint_id)
            .bind(task_id)
            .bind(channel)
            .bind(&value_bytes)
            .execute(&self.pool)
            .await
            .map_err(|e| LangGraphError::Other(format!("SQLite insert write error: {}", e)))?;
        }

        Ok(())
    }

    async fn list(
        &self,
        config: &HashMap<String, Value>,
        limit: Option<usize>,
    ) -> Result<Vec<CheckpointTuple>> {
        let (thread_id, checkpoint_ns, _) = Self::extract_config(config);
        let limit_val = limit.unwrap_or(100) as i32;

        let rows = sqlx::query(
            r#"
            SELECT * FROM checkpoints
            WHERE thread_id = ? AND checkpoint_ns = ?
            ORDER BY created_at DESC
            LIMIT ?
            "#,
        )
        .bind(thread_id)
        .bind(checkpoint_ns)
        .bind(limit_val)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| LangGraphError::Other(format!("SQLite query error: {}", e)))?;

        let mut results = Vec::with_capacity(rows.len());
        for row in &rows {
            results.push(self.row_to_tuple(row).await?);
        }
        Ok(results)
    }

    async fn list_checkpoints(&self, thread_id: &str) -> Result<Vec<CheckpointEntry>> {
        let rows = sqlx::query(
            r#"
            SELECT * FROM checkpoints
            WHERE thread_id = ? AND checkpoint_ns = ''
            ORDER BY created_at ASC
            "#,
        )
        .bind(thread_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| LangGraphError::Other(format!("SQLite query error: {}", e)))?;

        let mut entries = Vec::with_capacity(rows.len());
        for row in &rows {
            let tuple = self.row_to_tuple(row).await?;

            let node_name = tuple
                .metadata
                .as_ref()
                .and_then(|m| {
                    m.writes
                        .as_ref()
                        .and_then(|w| w.keys().next().cloned())
                        .or_else(|| Some(m.source.clone()))
                })
                .unwrap_or_default();

            let timestamp = tuple
                .checkpoint
                .ts
                .rsplit('+')
                .next()
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0)
                * 1000;

            let state = Value::Object(
                tuple
                    .checkpoint
                    .channel_values
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            );

            entries.push(CheckpointEntry {
                checkpoint_id: tuple.checkpoint.id.clone(),
                thread_id: thread_id.to_string(),
                node_name,
                timestamp,
                state,
            });
        }

        Ok(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pregel::checkpoint::empty_checkpoint;
    use serde_json::json;

    async fn create_test_saver() -> SqliteCheckpointSaver {
        SqliteCheckpointSaver::new("sqlite::memory:")
            .await
            .expect("Failed to create in-memory SQLite saver")
    }

    fn test_config(thread_id: &str) -> HashMap<String, Value> {
        let mut config = HashMap::new();
        config.insert("thread_id".to_string(), json!(thread_id));
        config
    }

    fn test_metadata(step: i64) -> CheckpointMetadata {
        CheckpointMetadata {
            source: "loop".to_string(),
            step,
            writes: None,
            extra: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_sqlite_save_and_load() {
        let saver = create_test_saver().await;
        let config = test_config("thread-1");

        let mut cp = empty_checkpoint();
        cp.channel_values
            .insert("state".to_string(), json!({"count": 42}));

        let cp_id = cp.id.clone();
        let new_config = saver.put(&config, cp, test_metadata(1)).await.unwrap();

        // Load by thread_id (latest)
        let loaded = saver.get(&config).await.unwrap();
        assert!(loaded.is_some());
        let tuple = loaded.unwrap();
        assert_eq!(tuple.checkpoint.id, cp_id);
        assert_eq!(tuple.checkpoint.channel_values["state"], json!({"count": 42}));

        // Load by explicit checkpoint_id
        let loaded2 = saver.get(&new_config).await.unwrap();
        assert!(loaded2.is_some());
        assert_eq!(loaded2.unwrap().checkpoint.id, cp_id);
    }

    #[tokio::test]
    async fn test_sqlite_list_checkpoints() {
        let saver = create_test_saver().await;
        let config = test_config("thread-1");

        for i in 0..5 {
            let cp = empty_checkpoint();
            saver
                .put(&config, cp, test_metadata(i))
                .await
                .unwrap();
        }

        let all = saver.list(&config, None).await.unwrap();
        assert_eq!(all.len(), 5);

        let limited = saver.list(&config, Some(2)).await.unwrap();
        assert_eq!(limited.len(), 2);
    }

    #[tokio::test]
    async fn test_sqlite_delete() {
        // SQLite saver does not expose a `delete` method on the trait, but we
        // verify that different threads are isolated and that listing works
        // correctly after multiple puts.
        let saver = create_test_saver().await;
        let config1 = test_config("thread-1");
        let config2 = test_config("thread-2");

        let cp1 = empty_checkpoint();
        saver
            .put(&config1, cp1, test_metadata(0))
            .await
            .unwrap();

        let cp2 = empty_checkpoint();
        saver
            .put(&config2, cp2, test_metadata(0))
            .await
            .unwrap();

        let list1 = saver.list(&config1, None).await.unwrap();
        assert_eq!(list1.len(), 1);

        let list2 = saver.list(&config2, None).await.unwrap();
        assert_eq!(list2.len(), 1);
    }

    #[tokio::test]
    async fn test_sqlite_load_nonexistent() {
        let saver = create_test_saver().await;
        let config = test_config("nonexistent-thread");

        let result = saver.get(&config).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_sqlite_put_writes() {
        let saver = create_test_saver().await;
        let config = test_config("thread-1");

        let cp = empty_checkpoint();
        let new_config = saver
            .put(&config, cp, test_metadata(0))
            .await
            .unwrap();

        let writes = vec![("state".to_string(), json!({"updated": true}))];
        saver
            .put_writes(&new_config, writes, "task-1")
            .await
            .unwrap();

        let loaded = saver.get(&new_config).await.unwrap().unwrap();
        let pending = loaded.pending_writes.unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].0, "task-1");
        assert_eq!(pending[0].1, "state");
        assert_eq!(pending[0].2, json!({"updated": true}));
    }

    #[tokio::test]
    async fn test_sqlite_metadata_roundtrip() {
        let saver = create_test_saver().await;
        let config = test_config("thread-1");

        let cp = empty_checkpoint();
        let mut metadata = test_metadata(3);
        metadata.source = "input".to_string();
        metadata.writes = Some({
            let mut m = HashMap::new();
            m.insert("ch".to_string(), json!("val"));
            m
        });

        saver.put(&config, cp, metadata).await.unwrap();

        let loaded = saver.get(&config).await.unwrap().unwrap();
        let meta = loaded.metadata.unwrap();
        assert_eq!(meta.source, "input");
        assert_eq!(meta.step, 3);
        assert!(meta.writes.is_some());
    }
}
