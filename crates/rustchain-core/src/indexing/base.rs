//! Record manager and document index abstractions.
//!
//! Mirrors Python `langchain_core.indexing.base`.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::documents::Document;
use crate::error::Result;

/// Response from an upsert operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertResponse {
    /// Keys that were successfully upserted.
    pub succeeded: Vec<String>,
    /// Keys that failed to upsert.
    pub failed: Vec<String>,
}

/// Response from a delete operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteResponse {
    /// Number of records deleted (if known).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_deleted: Option<usize>,
    /// Keys that were successfully deleted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub succeeded: Option<Vec<String>>,
    /// Keys that failed to delete.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failed: Option<Vec<String>>,
}

/// A record stored in the record manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub group_id: Option<String>,
    pub updated_at: f64,
}

/// Abstract interface for tracking which documents have been indexed.
///
/// The record manager keeps track of document keys and their associated
/// metadata (group ID, timestamps) so that the indexing pipeline can
/// detect new, updated, and deleted documents.
#[async_trait]
pub trait RecordManager: Send + Sync {
    /// Create the database schema or table if it doesn't exist.
    async fn create_schema(&self) -> Result<()>;

    /// Get the current server time as a high-resolution timestamp.
    async fn get_time(&self) -> Result<f64>;

    /// Upsert records. Each key maps to an optional group_id.
    /// The `time_at_least` parameter ensures the recorded timestamp is
    /// at least this value.
    async fn update(
        &self,
        keys: &[String],
        group_ids: &[Option<String>],
        time_at_least: Option<f64>,
    ) -> Result<()>;

    /// Check which keys exist in the record manager.
    async fn exists(&self, keys: &[String]) -> Result<Vec<bool>>;

    /// List keys matching optional filters.
    ///
    /// * `before` - Only keys updated before this timestamp.
    /// * `after` - Only keys updated after this timestamp.
    /// * `group_ids` - Only keys belonging to these groups.
    /// * `limit` - Maximum number of keys to return.
    async fn list_keys(
        &self,
        before: Option<f64>,
        after: Option<f64>,
        group_ids: Option<&[String]>,
        limit: Option<usize>,
    ) -> Result<Vec<String>>;

    /// Delete the specified keys from the record manager.
    async fn delete_keys(&self, keys: &[String]) -> Result<()>;
}

/// In-memory record manager for testing and lightweight usage.
pub struct InMemoryRecordManager {
    namespace: String,
    records: Mutex<HashMap<String, Record>>,
}

impl InMemoryRecordManager {
    pub fn new(namespace: impl Into<String>) -> Self {
        Self {
            namespace: namespace.into(),
            records: Mutex::new(HashMap::new()),
        }
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    fn current_time() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64()
    }
}

#[async_trait]
impl RecordManager for InMemoryRecordManager {
    async fn create_schema(&self) -> Result<()> {
        // No-op for in-memory.
        Ok(())
    }

    async fn get_time(&self) -> Result<f64> {
        Ok(Self::current_time())
    }

    async fn update(
        &self,
        keys: &[String],
        group_ids: &[Option<String>],
        time_at_least: Option<f64>,
    ) -> Result<()> {
        let mut records = self.records.lock().unwrap();
        let now = Self::current_time();
        let timestamp = match time_at_least {
            Some(t) if t > now => t,
            _ => now,
        };

        for (key, group_id) in keys.iter().zip(group_ids.iter()) {
            records.insert(
                key.clone(),
                Record {
                    group_id: group_id.clone(),
                    updated_at: timestamp,
                },
            );
        }
        Ok(())
    }

    async fn exists(&self, keys: &[String]) -> Result<Vec<bool>> {
        let records = self.records.lock().unwrap();
        Ok(keys.iter().map(|k| records.contains_key(k)).collect())
    }

    async fn list_keys(
        &self,
        before: Option<f64>,
        after: Option<f64>,
        group_ids: Option<&[String]>,
        limit: Option<usize>,
    ) -> Result<Vec<String>> {
        let records = self.records.lock().unwrap();
        let mut keys: Vec<String> = records
            .iter()
            .filter(|(_, record)| {
                if let Some(b) = before {
                    if record.updated_at >= b {
                        return false;
                    }
                }
                if let Some(a) = after {
                    if record.updated_at <= a {
                        return false;
                    }
                }
                if let Some(gids) = group_ids {
                    match &record.group_id {
                        Some(gid) => {
                            if !gids.contains(gid) {
                                return false;
                            }
                        }
                        None => return false,
                    }
                }
                true
            })
            .map(|(k, _)| k.clone())
            .collect();

        keys.sort();
        if let Some(lim) = limit {
            keys.truncate(lim);
        }
        Ok(keys)
    }

    async fn delete_keys(&self, keys: &[String]) -> Result<()> {
        let mut records = self.records.lock().unwrap();
        for key in keys {
            records.remove(key);
        }
        Ok(())
    }
}

/// Abstract interface for a document index (e.g., vectorstore wrapper).
///
/// Provides CRUD operations for documents in a storage backend.
#[async_trait]
pub trait DocumentIndex: Send + Sync {
    /// Upsert documents into the index.
    async fn upsert(&self, docs: Vec<Document>) -> Result<UpsertResponse>;

    /// Delete documents by IDs.
    async fn delete(&self, ids: &[String]) -> Result<DeleteResponse>;

    /// Get documents by IDs.
    async fn get(&self, ids: &[String]) -> Result<Vec<Document>>;
}
