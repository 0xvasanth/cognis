//! Postgres-backed checkpoint persistence.
//!
//! Provides [`PostgresCheckpointSaver`], a durable checkpoint backend that stores
//! graph state in a PostgreSQL database using JSONB columns. This allows agent
//! state to survive process restarts and be shared across distributed workers.
//!
//! # Feature flag
//!
//! This module is only available when the `postgres` feature is enabled.
//!
//! # Example
//!
//! ```ignore
//! use langgraph::checkpoint::PostgresCheckpointSaver;
//!
//! let saver = PostgresCheckpointSaver::new("postgres://user:pass@localhost/db").await?;
//! ```

use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::Value;
use sqlx::postgres::{PgPool, PgPoolOptions};
use sqlx::Row;

use crate::errors::{LangGraphError, Result};
use crate::pregel::checkpoint::{
    Checkpoint, CheckpointEntry, CheckpointMetadata, CheckpointSaver, CheckpointTuple,
};

/// A checkpoint saver backed by PostgreSQL via `sqlx`.
///
/// Stores checkpoints as JSONB and pending writes in two tables. Supports
/// multiple threads and namespaces with concurrent access.
#[derive(Debug, Clone)]
pub struct PostgresCheckpointSaver {
    pool: PgPool,
}

impl PostgresCheckpointSaver {
    /// Connect to a PostgreSQL database and create the required tables.
    ///
    /// # Arguments
    ///
    /// * `database_url` — A PostgreSQL connection string, e.g.
    ///   `"postgres://user:password@localhost:5432/mydb"`.
    pub async fn new(database_url: &str) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(database_url)
            .await
            .map_err(|e| LangGraphError::Other(format!("Postgres connection error: {}", e)))?;

        let saver = Self { pool };
        saver.setup_tables().await?;
        Ok(saver)
    }

    /// Create a saver from an existing connection pool.
    ///
    /// Callers are responsible for calling [`setup_tables`](Self::setup_tables)
    /// after construction.
    pub fn from_pool(pool: PgPool) -> Self {
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
                data JSONB NOT NULL,
                metadata JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
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
                value JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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
    async fn row_to_tuple(&self, row: &sqlx::postgres::PgRow) -> Result<CheckpointTuple> {
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

        // Data is stored as JSONB, which sqlx returns as serde_json::Value.
        let data_value: Value = row
            .try_get("data")
            .map_err(|e| LangGraphError::Other(format!("Column error: {}", e)))?;
        let checkpoint: Checkpoint = serde_json::from_value(data_value)
            .map_err(|e| LangGraphError::Other(format!("Deserialize checkpoint error: {}", e)))?;

        let metadata_value: Option<Value> = row
            .try_get("metadata")
            .map_err(|e| LangGraphError::Other(format!("Column error: {}", e)))?;
        let metadata: Option<CheckpointMetadata> = metadata_value
            .map(serde_json::from_value)
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
            WHERE thread_id = $1 AND checkpoint_ns = $2 AND checkpoint_id = $3
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
                let value: Value = pw_row
                    .try_get("value")
                    .map_err(|e| LangGraphError::Other(format!("Column error: {}", e)))?;
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
impl CheckpointSaver for PostgresCheckpointSaver {
    async fn get(&self, config: &HashMap<String, Value>) -> Result<Option<CheckpointTuple>> {
        let (thread_id, checkpoint_ns, checkpoint_id) = Self::extract_config(config);

        let row = if let Some(cid) = checkpoint_id {
            sqlx::query(
                r#"
                SELECT * FROM checkpoints
                WHERE thread_id = $1 AND checkpoint_ns = $2 AND checkpoint_id = $3
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
                WHERE thread_id = $1 AND checkpoint_ns = $2
                ORDER BY created_at DESC
                LIMIT 1
                "#,
            )
            .bind(thread_id)
            .bind(checkpoint_ns)
            .fetch_optional(&self.pool)
            .await
        }
        .map_err(|e| LangGraphError::Other(format!("Postgres query error: {}", e)))?;

        match row {
            Some(r) => Ok(Some(self.row_to_tuple(&r).await?)),
            None => Ok(None),
        }
    }

    async fn get_tuple(&self, config: &HashMap<String, Value>) -> Result<Option<CheckpointTuple>> {
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

        let data = serde_json::to_value(&checkpoint)
            .map_err(|e| LangGraphError::Other(format!("Serialize checkpoint error: {}", e)))?;
        let metadata_value = serde_json::to_value(&metadata)
            .map_err(|e| LangGraphError::Other(format!("Serialize metadata error: {}", e)))?;

        sqlx::query(
            r#"
            INSERT INTO checkpoints
                (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, data, metadata)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id)
            DO UPDATE SET
                parent_checkpoint_id = EXCLUDED.parent_checkpoint_id,
                data = EXCLUDED.data,
                metadata = EXCLUDED.metadata,
                created_at = NOW()
            "#,
        )
        .bind(thread_id)
        .bind(checkpoint_ns)
        .bind(&checkpoint.id)
        .bind(&parent_checkpoint_id)
        .bind(&data)
        .bind(&metadata_value)
        .execute(&self.pool)
        .await
        .map_err(|e| LangGraphError::Other(format!("Postgres insert error: {}", e)))?;

        let mut new_config = config.clone();
        new_config.insert("checkpoint_id".to_string(), Value::String(checkpoint.id));

        Ok(new_config)
    }

    async fn put_writes(
        &self,
        config: &HashMap<String, Value>,
        writes: Vec<(String, Value)>,
        task_id: &str,
    ) -> Result<()> {
        let (thread_id, checkpoint_ns, checkpoint_id) = Self::extract_config(config);
        let checkpoint_id = checkpoint_id
            .ok_or_else(|| LangGraphError::Other("checkpoint_id required for put_writes".into()))?;

        for (channel, value) in &writes {
            let value_json = serde_json::to_value(value).map_err(|e| {
                LangGraphError::Other(format!("Serialize write value error: {}", e))
            })?;

            sqlx::query(
                r#"
                INSERT INTO pending_writes
                    (thread_id, checkpoint_ns, checkpoint_id, task_id, channel, value)
                VALUES ($1, $2, $3, $4, $5, $6)
                "#,
            )
            .bind(thread_id)
            .bind(checkpoint_ns)
            .bind(checkpoint_id)
            .bind(task_id)
            .bind(channel)
            .bind(&value_json)
            .execute(&self.pool)
            .await
            .map_err(|e| LangGraphError::Other(format!("Postgres insert write error: {}", e)))?;
        }

        Ok(())
    }

    async fn list(
        &self,
        config: &HashMap<String, Value>,
        limit: Option<usize>,
    ) -> Result<Vec<CheckpointTuple>> {
        let (thread_id, checkpoint_ns, _) = Self::extract_config(config);
        let limit_val = limit.unwrap_or(100) as i64;

        let rows = sqlx::query(
            r#"
            SELECT * FROM checkpoints
            WHERE thread_id = $1 AND checkpoint_ns = $2
            ORDER BY created_at DESC
            LIMIT $3
            "#,
        )
        .bind(thread_id)
        .bind(checkpoint_ns)
        .bind(limit_val)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| LangGraphError::Other(format!("Postgres query error: {}", e)))?;

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
            WHERE thread_id = $1 AND checkpoint_ns = ''
            ORDER BY created_at ASC
            "#,
        )
        .bind(thread_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| LangGraphError::Other(format!("Postgres query error: {}", e)))?;

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
    use crate::pregel::checkpoint::{empty_checkpoint, CheckpointMetadata, LATEST_VERSION};
    use serde_json::json;

    /// Test that a checkpoint round-trips through JSON serialization correctly.
    /// This validates the JSONB storage path without needing a real database.
    #[test]
    fn test_checkpoint_jsonb_roundtrip() {
        let mut cp = empty_checkpoint();
        cp.channel_values.insert(
            "state".to_string(),
            json!({"count": 42, "items": [1, 2, 3]}),
        );
        cp.channel_versions.insert("state".to_string(), 5);
        cp.versions_seen.insert("agent".to_string(), {
            let mut m = HashMap::new();
            m.insert("state".to_string(), 4);
            m
        });

        // Simulate JSONB storage: serialize to Value, then back.
        let as_value = serde_json::to_value(&cp).unwrap();
        let restored: Checkpoint = serde_json::from_value(as_value).unwrap();

        assert_eq!(restored.v, LATEST_VERSION);
        assert_eq!(restored.id, cp.id);
        assert_eq!(restored.ts, cp.ts);
        assert_eq!(restored.channel_values, cp.channel_values);
        assert_eq!(restored.channel_versions, cp.channel_versions);
        assert_eq!(restored.versions_seen, cp.versions_seen);
    }

    /// Test metadata serialization through serde_json::Value (JSONB path).
    #[test]
    fn test_metadata_jsonb_roundtrip() {
        let metadata = CheckpointMetadata {
            source: "loop".to_string(),
            step: 7,
            writes: Some({
                let mut m = HashMap::new();
                m.insert("agent".to_string(), json!({"action": "search"}));
                m
            }),
            extra: {
                let mut m = HashMap::new();
                m.insert("run_id".to_string(), json!("run-abc-123"));
                m
            },
        };

        let as_value = serde_json::to_value(&metadata).unwrap();
        let restored: CheckpointMetadata = serde_json::from_value(as_value).unwrap();

        assert_eq!(restored.source, "loop");
        assert_eq!(restored.step, 7);
        assert!(restored.writes.is_some());
        let writes = restored.writes.unwrap();
        assert_eq!(writes["agent"], json!({"action": "search"}));
        assert_eq!(restored.extra["run_id"], json!("run-abc-123"));
    }

    /// Test that CheckpointTuple serializes/deserializes correctly through JSON,
    /// covering the full data structure stored in Postgres.
    #[test]
    fn test_checkpoint_tuple_jsonb_roundtrip() {
        let cp = empty_checkpoint();
        let metadata = CheckpointMetadata {
            source: "input".to_string(),
            step: 0,
            writes: None,
            extra: HashMap::new(),
        };

        let mut config = HashMap::new();
        config.insert("thread_id".to_string(), json!("thread-1"));
        config.insert("checkpoint_ns".to_string(), json!(""));
        config.insert("checkpoint_id".to_string(), json!(cp.id.clone()));

        let tuple = CheckpointTuple {
            checkpoint: cp,
            config: config.clone(),
            metadata: Some(metadata),
            parent_config: None,
            pending_writes: Some(vec![(
                "task-1".to_string(),
                "state".to_string(),
                json!({"key": "value"}),
            )]),
        };

        let as_value = serde_json::to_value(&tuple).unwrap();
        let restored: CheckpointTuple = serde_json::from_value(as_value).unwrap();

        assert_eq!(restored.checkpoint.id, tuple.checkpoint.id);
        assert_eq!(restored.config["thread_id"], json!("thread-1"));
        assert!(restored.metadata.is_some());
        assert_eq!(restored.metadata.unwrap().source, "input");
        assert!(restored.pending_writes.is_some());
        let pw = restored.pending_writes.unwrap();
        assert_eq!(pw.len(), 1);
        assert_eq!(pw[0].0, "task-1");
        assert_eq!(pw[0].1, "state");
    }

    /// Test config extraction helper.
    #[test]
    fn test_extract_config() {
        let mut config = HashMap::new();
        config.insert("thread_id".to_string(), json!("t-1"));
        config.insert("checkpoint_ns".to_string(), json!("ns-1"));
        config.insert("checkpoint_id".to_string(), json!("cp-1"));

        let (tid, ns, cid) = PostgresCheckpointSaver::extract_config(&config);
        assert_eq!(tid, "t-1");
        assert_eq!(ns, "ns-1");
        assert_eq!(cid, Some("cp-1"));
    }

    /// Test config extraction with missing fields defaults properly.
    #[test]
    fn test_extract_config_defaults() {
        let config = HashMap::new();
        let (tid, ns, cid) = PostgresCheckpointSaver::extract_config(&config);
        assert_eq!(tid, "");
        assert_eq!(ns, "");
        assert_eq!(cid, None);
    }

    /// Test that a checkpoint with updated_channels serializes without the field
    /// when it is None (due to skip_serializing_if).
    #[test]
    fn test_checkpoint_optional_updated_channels() {
        let cp = empty_checkpoint();
        assert!(cp.updated_channels.is_none());

        let as_value = serde_json::to_value(&cp).unwrap();
        assert!(as_value.get("updated_channels").is_none());

        // Now with Some
        let mut cp2 = empty_checkpoint();
        cp2.updated_channels = Some(vec!["a".to_string(), "b".to_string()]);
        let as_value2 = serde_json::to_value(&cp2).unwrap();
        assert_eq!(as_value2["updated_channels"], json!(["a", "b"]));

        // Round-trip
        let restored: Checkpoint = serde_json::from_value(as_value2).unwrap();
        assert_eq!(
            restored.updated_channels,
            Some(vec!["a".to_string(), "b".to_string()])
        );
    }
}
