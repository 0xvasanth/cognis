//! Postgres-backed checkpointer (feature-gated).
//!
//! Schema parallels [`super::sqlite::SqliteCheckpointer`]: a single
//! `checkpoints` table keyed by `(run_id, namespace, step)`. State is
//! stored as JSONB.

use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{de::DeserializeOwned, Serialize};
use sqlx::postgres::{PgPool, PgPoolOptions};
use sqlx::Row;
use uuid::Uuid;

use cognis_core::{CognisError, Result};

use crate::state::GraphState;

use super::serializer::{CheckpointSerializer, JsonSerializer};
use super::Checkpointer;

/// Postgres-backed [`Checkpointer`].
pub struct PostgresCheckpointer<S> {
    pool: PgPool,
    table: String,
    namespace: String,
    serializer: Arc<dyn CheckpointSerializer>,
    _phantom: PhantomData<fn() -> S>,
}

impl<S> PostgresCheckpointer<S>
where
    S: GraphState + Serialize + DeserializeOwned + Clone,
{
    /// Connect to a Postgres database. Connection string is the standard
    /// `postgres://user:pass@host/db` form.
    pub async fn connect(url: &str) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(url)
            .await
            .map_err(|e| CognisError::Configuration(format!("postgres connect: {e}")))?;
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

    /// Override the table name.
    pub fn with_table(mut self, table: impl Into<String>) -> Self {
        self.table = table.into();
        self
    }

    /// Set the namespace for subgraph isolation.
    pub fn with_namespace(mut self, ns: impl Into<String>) -> Self {
        self.namespace = ns.into();
        self
    }

    /// Override the serializer (default JSON). The serializer's name is
    /// stored alongside each row.
    pub fn with_serializer(mut self, s: Arc<dyn CheckpointSerializer>) -> Self {
        self.serializer = s;
        self
    }

    async fn ensure_table(&self) -> Result<()> {
        // `state` was historically JSONB; we keep it JSONB and store
        // bytes from non-JSON serializers as a base64-encoded string
        // wrapped in a JSON object. This avoids a destructive schema
        // migration on existing deployments while still letting users
        // opt in to alternative formats.
        let stmt = format!(
            "CREATE TABLE IF NOT EXISTS {table} (
                 run_id     UUID NOT NULL,
                 namespace  TEXT NOT NULL,
                 step       BIGINT NOT NULL,
                 state      JSONB NOT NULL,
                 serializer TEXT NOT NULL DEFAULT 'json',
                 PRIMARY KEY (run_id, namespace, step)
             )",
            table = self.table,
        );
        sqlx::query(&stmt)
            .execute(&self.pool)
            .await
            .map_err(|e| CognisError::Internal(format!("postgres create table: {e}")))?;
        // Best-effort migration for older databases.
        let alter = format!(
            "ALTER TABLE {table} ADD COLUMN IF NOT EXISTS serializer TEXT NOT NULL DEFAULT 'json'",
            table = self.table,
        );
        let _ = sqlx::query(&alter).execute(&self.pool).await;
        Ok(())
    }
}

#[async_trait]
impl<S> Checkpointer<S> for PostgresCheckpointer<S>
where
    S: GraphState + Serialize + DeserializeOwned + Clone,
{
    async fn save(&self, run_id: Uuid, step: u64, state: &S) -> Result<()> {
        // For JSON serializer, store the value as JSONB directly. For
        // non-JSON serializers, base64-wrap the bytes inside a JSON object
        // so the JSONB column stays well-formed.
        let json: serde_json::Value = if self.serializer.name() == "json" {
            serde_json::to_value(state)
                .map_err(|e| CognisError::Serialization(format!("checkpoint serialize: {e}")))?
        } else {
            let bytes = super::serializer::encode(&self.serializer, state)?;
            let b64 = cognis_core::base64_encode(&bytes);
            serde_json::json!({ "__cognis_serialized__": b64 })
        };
        let stmt = format!(
            "INSERT INTO {table} (run_id, namespace, step, state, serializer)
             VALUES ($1, $2, $3, $4, $5)
             ON CONFLICT (run_id, namespace, step)
             DO UPDATE SET state = EXCLUDED.state, serializer = EXCLUDED.serializer",
            table = self.table,
        );
        sqlx::query(&stmt)
            .bind(run_id)
            .bind(&self.namespace)
            .bind(step as i64)
            .bind(json)
            .bind(self.serializer.name())
            .execute(&self.pool)
            .await
            .map_err(|e| CognisError::Internal(format!("postgres save: {e}")))?;
        Ok(())
    }

    async fn load(&self, run_id: Uuid, step: Option<u64>) -> Result<Option<S>> {
        let row = match step {
            Some(s) => {
                let stmt = format!(
                    "SELECT state, serializer FROM {table}
                     WHERE run_id = $1 AND namespace = $2 AND step = $3",
                    table = self.table,
                );
                sqlx::query(&stmt)
                    .bind(run_id)
                    .bind(&self.namespace)
                    .bind(s as i64)
                    .fetch_optional(&self.pool)
                    .await
                    .map_err(|e| CognisError::Internal(format!("postgres load: {e}")))?
            }
            None => {
                let stmt = format!(
                    "SELECT state, serializer FROM {table}
                     WHERE run_id = $1 AND namespace = $2
                     ORDER BY step DESC LIMIT 1",
                    table = self.table,
                );
                sqlx::query(&stmt)
                    .bind(run_id)
                    .bind(&self.namespace)
                    .fetch_optional(&self.pool)
                    .await
                    .map_err(|e| CognisError::Internal(format!("postgres load latest: {e}")))?
            }
        };
        match row {
            None => Ok(None),
            Some(row) => {
                let json: serde_json::Value = row
                    .try_get("state")
                    .map_err(|e| CognisError::Internal(format!("postgres read column: {e}")))?;
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
                let state: S = if stored_serializer == "json" {
                    serde_json::from_value(json).map_err(|e| {
                        CognisError::Serialization(format!("checkpoint deserialize: {e}"))
                    })?
                } else {
                    let b64 = json["__cognis_serialized__"].as_str().ok_or_else(|| {
                        CognisError::Serialization(
                            "non-json checkpoint missing __cognis_serialized__ wrapper".into(),
                        )
                    })?;
                    let bytes = cognis_core::base64_decode(b64)?;
                    super::serializer::decode(&self.serializer, &bytes)?
                };
                Ok(Some(state))
            }
        }
    }

    async fn list(&self, run_id: Uuid) -> Result<Vec<u64>> {
        let stmt = format!(
            "SELECT step FROM {table}
             WHERE run_id = $1 AND namespace = $2
             ORDER BY step ASC",
            table = self.table,
        );
        let rows = sqlx::query(&stmt)
            .bind(run_id)
            .bind(&self.namespace)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| CognisError::Internal(format!("postgres list: {e}")))?;
        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            let s: i64 = r
                .try_get("step")
                .map_err(|e| CognisError::Internal(format!("postgres read column: {e}")))?;
            out.push(s as u64);
        }
        Ok(out)
    }
}
