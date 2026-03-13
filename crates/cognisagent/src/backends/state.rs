//! In-memory state backend.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Mutex;

use super::{Backend, Result};

/// Simple in-memory backend backed by a `HashMap`.
///
/// State is lost when the process exits. Suitable for testing,
/// short-lived sessions, and development.
#[derive(Clone)]
pub struct StateBackend {
    store: Arc<Mutex<HashMap<String, Value>>>,
}

impl StateBackend {
    /// Create a new empty in-memory backend.
    pub fn new() -> Self {
        Self {
            store: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl Default for StateBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Backend for StateBackend {
    async fn save_state(&self, session_id: &str, state: &Value) -> Result<()> {
        let mut store = self.store.lock().await;
        store.insert(session_id.to_string(), state.clone());
        Ok(())
    }

    async fn load_state(&self, session_id: &str) -> Result<Option<Value>> {
        let store = self.store.lock().await;
        Ok(store.get(session_id).cloned())
    }

    async fn list_sessions(&self) -> Result<Vec<String>> {
        let store = self.store.lock().await;
        Ok(store.keys().cloned().collect())
    }
}
