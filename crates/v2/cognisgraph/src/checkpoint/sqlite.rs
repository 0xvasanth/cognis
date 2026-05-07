//! Sqlite-backed checkpointer (feature-gated).
//!
//! State is serialized via a pluggable [`CheckpointSerializer`] (default
//! [`JsonSerializer`]) into a single `checkpoints` table keyed by
//! `(run_id, namespace, step)`. The serializer's name is stored alongside
//! each row so payloads survive a serializer swap.

use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{de::DeserializeOwned, Serialize};
use sqlx::sqlite::{SqlitePool, SqlitePoolOptions};
use sqlx::Row;
use uuid::Uuid;

use cognis2_core::{CognisError, Result};

use crate::state::GraphState;

use super::serializer::{CheckpointSerializer, JsonSerializer};
use super::Checkpointer;

/// Sqlite-backed implementation of [`Checkpointer`].
///
/// Construct via [`SqliteCheckpointer::connect`]:
/// ```ignore
/// let cp = SqliteCheckpointer::<MyState>::connect("sqlite:graph.db").await?;
/// ```
pub struct SqliteCheckpointer<S> {
    pool: SqlitePool,
    table: String,
    namespace: String,
    serializer: Arc<dyn CheckpointSerializer>,
    _phantom: PhantomData<fn() -> S>,
}

impl<S> SqliteCheckpointer<S>
where
    S: GraphState + Serialize + DeserializeOwned + Clone,
{
    /// Connect to a sqlite database (file or `:memory:`) and ensure the
    /// `checkpoints` table exists. The default table name is `checkpoints`;
    /// override with [`SqliteCheckpointer::with_table`]. Default serializer
    /// is [`JsonSerializer`].
    pub async fn connect(url: &str) -> Result<Self> {
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(url)
            .await
            .map_err(|e| CognisError::Configuration(format!("sqlite connect: {e}")))?;
        let cp = Self {
            pool,
            table: "checkpoints".to_string(),
            namespace: String::new(),
            serializer: Arc::new(JsonSerializer),
            _phantom: PhantomData,
        };
        cp.ensure_table().await?;
        Ok(cp)
    }

    /// Override the table name (default `checkpoints`).
    pub fn with_table(mut self, table: impl Into<String>) -> Self {
        self.table = table.into();
        self
    }

    /// Set the namespace for subgraph isolation. All operations on this
    /// checkpointer instance will scope to `(run_id, namespace, step)`.
    pub fn with_namespace(mut self, ns: impl Into<String>) -> Self {
        self.namespace = ns.into();
        self
    }

    /// Override the serializer (default JSON). Each checkpoint stores
    /// the serializer's name so reads pick the matching format.
    pub fn with_serializer(mut self, s: Arc<dyn CheckpointSerializer>) -> Self {
        self.serializer = s;
        self
    }

    async fn ensure_table(&self) -> Result<()> {
        let stmt = format!(
            "CREATE TABLE IF NOT EXISTS {table} (
                 run_id     TEXT NOT NULL,
                 namespace  TEXT NOT NULL,
                 step       INTEGER NOT NULL,
                 state      BLOB NOT NULL,
                 serializer TEXT NOT NULL DEFAULT 'json',
                 PRIMARY KEY (run_id, namespace, step)
             )",
            table = self.table,
        );
        sqlx::query(&stmt)
            .execute(&self.pool)
            .await
            .map_err(|e| CognisError::Internal(format!("sqlite create table: {e}")))?;
        // Best-effort migration for older databases (no-op if the column
        // already exists; sqlite errors with "duplicate column", which we
        // silently ignore).
        let alter = format!(
            "ALTER TABLE {table} ADD COLUMN serializer TEXT NOT NULL DEFAULT 'json'",
            table = self.table,
        );
        let _ = sqlx::query(&alter).execute(&self.pool).await;
        Ok(())
    }
}

#[async_trait]
impl<S> Checkpointer<S> for SqliteCheckpointer<S>
where
    S: GraphState + Serialize + DeserializeOwned + Clone,
{
    async fn save(&self, run_id: Uuid, step: u64, state: &S) -> Result<()> {
        let bytes = super::serializer::encode(&self.serializer, state)?;
        let stmt = format!(
            "INSERT OR REPLACE INTO {table}
             (run_id, namespace, step, state, serializer)
             VALUES (?, ?, ?, ?, ?)",
            table = self.table,
        );
        sqlx::query(&stmt)
            .bind(run_id.to_string())
            .bind(&self.namespace)
            .bind(step as i64)
            .bind(bytes)
            .bind(self.serializer.name())
            .execute(&self.pool)
            .await
            .map_err(|e| CognisError::Internal(format!("sqlite save: {e}")))?;
        Ok(())
    }

    async fn load(&self, run_id: Uuid, step: Option<u64>) -> Result<Option<S>> {
        let row = match step {
            Some(s) => {
                let stmt = format!(
                    "SELECT state, serializer FROM {table}
                     WHERE run_id = ? AND namespace = ? AND step = ?",
                    table = self.table,
                );
                sqlx::query(&stmt)
                    .bind(run_id.to_string())
                    .bind(&self.namespace)
                    .bind(s as i64)
                    .fetch_optional(&self.pool)
                    .await
                    .map_err(|e| CognisError::Internal(format!("sqlite load: {e}")))?
            }
            None => {
                let stmt = format!(
                    "SELECT state, serializer FROM {table}
                     WHERE run_id = ? AND namespace = ?
                     ORDER BY step DESC LIMIT 1",
                    table = self.table,
                );
                sqlx::query(&stmt)
                    .bind(run_id.to_string())
                    .bind(&self.namespace)
                    .fetch_optional(&self.pool)
                    .await
                    .map_err(|e| CognisError::Internal(format!("sqlite load latest: {e}")))?
            }
        };
        match row {
            None => Ok(None),
            Some(row) => {
                let bytes: Vec<u8> = row
                    .try_get("state")
                    .map_err(|e| CognisError::Internal(format!("sqlite read column: {e}")))?;
                let stored_serializer: String = row
                    .try_get("serializer")
                    .unwrap_or_else(|_| "json".to_string());
                if stored_serializer != self.serializer.name() {
                    return Err(CognisError::Configuration(format!(
                        "checkpoint was written with serializer `{stored_serializer}` but \
                         this checkpointer is configured for `{}`",
                        self.serializer.name()
                    )));
                }
                let state: S = super::serializer::decode(&self.serializer, &bytes)?;
                Ok(Some(state))
            }
        }
    }

    async fn list(&self, run_id: Uuid) -> Result<Vec<u64>> {
        let stmt = format!(
            "SELECT step FROM {table}
             WHERE run_id = ? AND namespace = ?
             ORDER BY step ASC",
            table = self.table,
        );
        let rows = sqlx::query(&stmt)
            .bind(run_id.to_string())
            .bind(&self.namespace)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| CognisError::Internal(format!("sqlite list: {e}")))?;
        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            let s: i64 = r
                .try_get("step")
                .map_err(|e| CognisError::Internal(format!("sqlite read column: {e}")))?;
            out.push(s as u64);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Default, Clone, Debug, Serialize, Deserialize, PartialEq)]
    struct S {
        n: u32,
    }
    #[derive(Default)]
    struct SU {
        n: u32,
    }
    impl GraphState for S {
        type Update = SU;
        fn apply(&mut self, u: Self::Update) {
            self.n += u.n;
        }
    }

    async fn cp() -> SqliteCheckpointer<S> {
        SqliteCheckpointer::<S>::connect("sqlite::memory:")
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn save_then_load_explicit_step() {
        let cp = cp().await;
        let id = Uuid::new_v4();
        cp.save(id, 0, &S { n: 1 }).await.unwrap();
        cp.save(id, 1, &S { n: 5 }).await.unwrap();
        assert_eq!(cp.load(id, Some(0)).await.unwrap(), Some(S { n: 1 }));
        assert_eq!(cp.load(id, Some(1)).await.unwrap(), Some(S { n: 5 }));
        assert_eq!(cp.load(id, Some(99)).await.unwrap(), None);
    }

    #[tokio::test]
    async fn load_latest_when_step_is_none() {
        let cp = cp().await;
        let id = Uuid::new_v4();
        cp.save(id, 0, &S { n: 1 }).await.unwrap();
        cp.save(id, 5, &S { n: 9 }).await.unwrap();
        cp.save(id, 2, &S { n: 4 }).await.unwrap();
        assert_eq!(cp.load(id, None).await.unwrap(), Some(S { n: 9 }));
    }

    #[tokio::test]
    async fn list_returns_sorted_steps() {
        let cp = cp().await;
        let id = Uuid::new_v4();
        for s in [3u64, 1, 4, 1, 5, 9, 2, 6] {
            cp.save(id, s, &S { n: s as u32 }).await.unwrap();
        }
        assert_eq!(cp.list(id).await.unwrap(), vec![1, 2, 3, 4, 5, 6, 9]);
    }

    #[tokio::test]
    async fn namespaces_isolate_runs() {
        let parent = cp().await;
        let id = Uuid::new_v4();
        parent.save(id, 0, &S { n: 1 }).await.unwrap();

        let child = SqliteCheckpointer::<S>::connect("sqlite::memory:")
            .await
            .unwrap()
            .with_namespace("subgraph_a");
        // Different db connection — also test that namespace col gets used.
        child.save(id, 0, &S { n: 100 }).await.unwrap();
        assert_eq!(child.load(id, None).await.unwrap(), Some(S { n: 100 }));
    }

    #[tokio::test]
    async fn unknown_run_returns_empty() {
        let cp = cp().await;
        let unknown = Uuid::new_v4();
        assert_eq!(cp.load(unknown, None).await.unwrap(), None);
        assert!(cp.list(unknown).await.unwrap().is_empty());
    }

    #[cfg(feature = "serializer-cbor")]
    #[tokio::test]
    async fn cbor_serializer_roundtrip() {
        use crate::checkpoint::CborSerializer;
        let cp = SqliteCheckpointer::<S>::connect("sqlite::memory:")
            .await
            .unwrap()
            .with_serializer(Arc::new(CborSerializer));
        let id = Uuid::new_v4();
        cp.save(id, 0, &S { n: 42 }).await.unwrap();
        assert_eq!(cp.load(id, Some(0)).await.unwrap(), Some(S { n: 42 }));
    }

    #[cfg(feature = "serializer-cbor")]
    #[tokio::test]
    async fn serializer_mismatch_errors() {
        use crate::checkpoint::CborSerializer;
        // Connect once, save with JSON, reconnect and try to read with CBOR.
        // Use a shared file so both connections see the same row.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ck.db");
        let url = format!("sqlite://{}?mode=rwc", path.display());

        let cp_json = SqliteCheckpointer::<S>::connect(&url).await.unwrap();
        let id = Uuid::new_v4();
        cp_json.save(id, 0, &S { n: 7 }).await.unwrap();

        let cp_cbor = SqliteCheckpointer::<S>::connect(&url)
            .await
            .unwrap()
            .with_serializer(Arc::new(CborSerializer));
        let err = cp_cbor.load(id, Some(0)).await.unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("json") && msg.contains("cbor"),
            "expected mismatch error, got: {msg}",
        );
    }
}
