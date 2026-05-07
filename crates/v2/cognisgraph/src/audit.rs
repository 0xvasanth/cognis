//! Structured audit trail of graph node transitions.
//!
//! Distinct from `Observer`: an audit log is durable and queryable,
//! while observers are firehose event subscribers. Wire an
//! [`AuditLogObserver`] to bridge cognis2 events into your audit log.

use std::sync::Mutex;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use cognis2_core::{Event, Observer, Result};

/// One audit row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Run correlation ID.
    pub run_id: Uuid,
    /// Superstep counter.
    pub step: u64,
    /// Node name.
    pub node: String,
    /// What happened.
    pub kind: AuditKind,
    /// UTC seconds since epoch.
    pub ts: u64,
}

/// What an audit entry records.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AuditKind {
    /// Node started.
    Start,
    /// Node finished.
    End,
    /// Node errored.
    Error,
}

/// Pluggable audit log.
#[async_trait]
pub trait AuditLog: Send + Sync {
    /// Record one entry.
    async fn record(&self, entry: AuditEntry) -> Result<()>;
    /// All entries for `run_id` (oldest first).
    async fn entries(&self, run_id: Uuid) -> Result<Vec<AuditEntry>>;
}

/// In-process audit log.
#[derive(Default)]
pub struct InMemoryAuditLog {
    inner: Mutex<Vec<AuditEntry>>,
}

impl InMemoryAuditLog {
    /// Empty log.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl AuditLog for InMemoryAuditLog {
    async fn record(&self, entry: AuditEntry) -> Result<()> {
        let mut v = self
            .inner
            .lock()
            .map_err(|e| cognis2_core::CognisError::Internal(format!("audit mutex: {e}")))?;
        v.push(entry);
        Ok(())
    }
    async fn entries(&self, run_id: Uuid) -> Result<Vec<AuditEntry>> {
        let v = self
            .inner
            .lock()
            .map_err(|e| cognis2_core::CognisError::Internal(format!("audit mutex: {e}")))?;
        Ok(v.iter().filter(|e| e.run_id == run_id).cloned().collect())
    }
}

/// Bridges cognis2 [`Event`]s into an [`AuditLog`].
pub struct AuditLogObserver {
    log: std::sync::Arc<dyn AuditLog>,
}

impl AuditLogObserver {
    /// Build with a target log.
    pub fn new(log: std::sync::Arc<dyn AuditLog>) -> Self {
        Self { log }
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl Observer for AuditLogObserver {
    fn on_event(&self, event: &Event) {
        let entry = match event {
            Event::OnNodeStart { node, step, run_id } => AuditEntry {
                run_id: *run_id,
                step: *step,
                node: node.clone(),
                kind: AuditKind::Start,
                ts: now_secs(),
            },
            Event::OnNodeEnd {
                node, step, run_id, ..
            } => AuditEntry {
                run_id: *run_id,
                step: *step,
                node: node.clone(),
                kind: AuditKind::End,
                ts: now_secs(),
            },
            Event::OnError { error, run_id } => AuditEntry {
                run_id: *run_id,
                step: 0,
                node: error.clone(),
                kind: AuditKind::Error,
                ts: now_secs(),
            },
            _ => return,
        };
        let log = self.log.clone();
        tokio::spawn(async move {
            let _ = log.record(entry).await;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[tokio::test]
    async fn record_and_query() {
        let log = InMemoryAuditLog::new();
        let id = Uuid::new_v4();
        log.record(AuditEntry {
            run_id: id,
            step: 0,
            node: "a".into(),
            kind: AuditKind::Start,
            ts: 1000,
        })
        .await
        .unwrap();
        log.record(AuditEntry {
            run_id: Uuid::new_v4(),
            step: 0,
            node: "x".into(),
            kind: AuditKind::Start,
            ts: 1001,
        })
        .await
        .unwrap();
        let got = log.entries(id).await.unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].node, "a");
    }

    #[tokio::test]
    async fn observer_bridges_events() {
        let log: Arc<dyn AuditLog> = Arc::new(InMemoryAuditLog::new());
        let obs = AuditLogObserver::new(log.clone());
        let id = Uuid::new_v4();
        obs.on_event(&Event::OnNodeStart {
            node: "n".into(),
            step: 1,
            run_id: id,
        });
        // Wait for the spawned task.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        let entries = log.entries(id).await.unwrap();
        assert_eq!(entries.len(), 1);
    }
}
