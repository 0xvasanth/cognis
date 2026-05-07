//! Per-agent health snapshot.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use cognis_llm::chat::HealthStatus;

/// Aggregated agent health.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentHealth {
    /// Status of the underlying LLM provider.
    pub llm: HealthStatus,
    /// Per-tool reachability (`true` = ping ok). Empty if no tools support
    /// `ping()`.
    pub tools: HashMap<String, bool>,
    /// Whether the workspace backend (if any) is reachable.
    pub backend: Option<bool>,
}

impl AgentHealth {
    /// True iff every component reports healthy.
    pub fn is_healthy(&self) -> bool {
        let llm_ok = matches!(self.llm, HealthStatus::Healthy { .. });
        let tools_ok = self.tools.values().all(|v| *v);
        let backend_ok = self.backend.unwrap_or(true);
        llm_ok && tools_ok && backend_ok
    }
}
