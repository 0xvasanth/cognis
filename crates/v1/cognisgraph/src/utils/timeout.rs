//! Node execution timeout, cancellation, and budget management utilities.
//!
//! Provides mechanisms for controlling node execution time within a graph,
//! including per-node timeouts, graph-level execution budgets, and
//! cooperative cancellation.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::errors::{LangGraphError, Result};

/// Cooperative cancellation token, re-exported from `cognis_core`.
///
/// Promoted to core so that every crate in the workspace (including
/// `cognis` and `cognisagent`) can share the same cancellation primitive.
pub use cognis_core::CancellationToken;

/// Default timeout duration for node execution (30 seconds).
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Action to take when a node times out.
#[derive(Debug, Clone)]
pub enum TimeoutAction {
    /// Return an error on timeout.
    Error,
    /// Return a partial/default value on timeout.
    ReturnPartial(Value),
    /// Skip the node and continue graph execution.
    Skip,
    /// Retry the node up to `max_retries` times.
    Retry { max_retries: usize },
}

/// Error returned when a node exceeds its timeout.
#[derive(Debug, Clone)]
pub struct TimeoutError {
    /// Name of the node that timed out.
    pub node: String,
    /// Actual elapsed time before timeout was triggered.
    pub elapsed: Duration,
    /// The configured timeout limit.
    pub limit: Duration,
}

impl std::fmt::Display for TimeoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Node '{}' timed out after {:?} (limit: {:?})",
            self.node, self.elapsed, self.limit
        )
    }
}

impl std::error::Error for TimeoutError {}

/// Configuration for timeout behavior.
#[derive(Debug, Clone)]
pub struct TimeoutConfig {
    /// Per-node timeout override.
    pub node_timeout: Option<Duration>,
    /// Total graph execution timeout.
    pub graph_timeout: Option<Duration>,
    /// Default timeout when no specific timeout is configured.
    pub default_timeout: Duration,
    /// Action to take on timeout.
    pub on_timeout: TimeoutAction,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            node_timeout: None,
            graph_timeout: None,
            default_timeout: DEFAULT_TIMEOUT,
            on_timeout: TimeoutAction::Error,
        }
    }
}

impl TimeoutConfig {
    /// Create a new `TimeoutConfig` with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the per-node timeout.
    pub fn with_node_timeout(mut self, timeout: Duration) -> Self {
        self.node_timeout = Some(timeout);
        self
    }

    /// Set the graph-level timeout.
    pub fn with_graph_timeout(mut self, timeout: Duration) -> Self {
        self.graph_timeout = Some(timeout);
        self
    }

    /// Set the default timeout.
    pub fn with_default_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = timeout;
        self
    }

    /// Set the action to take on timeout.
    pub fn with_timeout_action(mut self, action: TimeoutAction) -> Self {
        self.on_timeout = action;
        self
    }

    /// Get the effective timeout duration.
    pub fn effective_timeout(&self) -> Duration {
        self.node_timeout.unwrap_or(self.default_timeout)
    }
}

/// Wraps a node function with timeout behavior.
#[derive(Debug, Clone)]
pub struct NodeTimeout {
    /// The timeout duration.
    pub timeout: Duration,
    /// Action to take on timeout.
    pub action: TimeoutAction,
}

impl NodeTimeout {
    /// Create a new `NodeTimeout` with the given timeout and action.
    pub fn new(timeout: Duration, action: TimeoutAction) -> Self {
        Self { timeout, action }
    }

    /// Execute a future with timeout and apply the configured action on timeout.
    pub async fn execute<F, Fut>(&self, node_name: &str, f: F) -> Result<Value>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<Value>>,
    {
        let start = Instant::now();
        match tokio::time::timeout(self.timeout, f()).await {
            Ok(result) => result,
            Err(_elapsed) => {
                let timeout_err = TimeoutError {
                    node: node_name.to_string(),
                    elapsed: start.elapsed(),
                    limit: self.timeout,
                };
                self.handle_timeout(node_name, &timeout_err)
            }
        }
    }

    /// Handle a timeout according to the configured action.
    ///
    /// For `Retry`, this returns an error directing the caller to use
    /// `execute_with_retries` instead.
    fn handle_timeout(&self, node_name: &str, err: &TimeoutError) -> Result<Value> {
        match &self.action {
            TimeoutAction::Error => Err(LangGraphError::Other(err.to_string())),
            TimeoutAction::ReturnPartial(value) => Ok(value.clone()),
            TimeoutAction::Skip => Ok(Value::Null),
            TimeoutAction::Retry { .. } => Err(LangGraphError::Other(format!(
                "Node '{}' timed out and retry requires execute_with_retries",
                node_name
            ))),
        }
    }

    /// Execute a function with retry support on timeout.
    ///
    /// The factory function is called on each attempt, allowing retries.
    pub async fn execute_with_retries<F, Fut>(&self, node_name: &str, factory: F) -> Result<Value>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<Value>>,
    {
        let max_attempts = match &self.action {
            TimeoutAction::Retry { max_retries } => *max_retries + 1,
            _ => 1,
        };

        let mut last_err = None;
        for attempt in 0..max_attempts {
            let start = Instant::now();
            match tokio::time::timeout(self.timeout, factory()).await {
                Ok(result) => return result,
                Err(_elapsed) => {
                    let timeout_err = TimeoutError {
                        node: node_name.to_string(),
                        elapsed: start.elapsed(),
                        limit: self.timeout,
                    };
                    if attempt + 1 < max_attempts {
                        // Will retry
                        last_err = Some(timeout_err);
                        continue;
                    }
                    // Last attempt — apply action
                    match &self.action {
                        TimeoutAction::Retry { .. } => {
                            return Err(LangGraphError::Other(format!(
                                "Node '{}' timed out after {} retries: {}",
                                node_name,
                                max_attempts - 1,
                                timeout_err
                            )));
                        }
                        TimeoutAction::ReturnPartial(value) => return Ok(value.clone()),
                        TimeoutAction::Skip => return Ok(Value::Null),
                        TimeoutAction::Error => {
                            return Err(LangGraphError::Other(timeout_err.to_string()));
                        }
                    }
                }
            }
        }

        Err(LangGraphError::Other(
            last_err
                .map(|e| e.to_string())
                .unwrap_or_else(|| format!("Node '{}' failed with no attempts", node_name)),
        ))
    }
}

/// Tracks a time budget across multiple node executions.
///
/// Useful for enforcing a total graph execution time limit while dynamically
/// allocating remaining time to subsequent nodes.
#[derive(Debug, Clone)]
pub struct ExecutionBudget {
    /// Total budget duration.
    total: Duration,
    /// When the budget was created.
    start: Instant,
    /// Execution records per node.
    records: Arc<Mutex<HashMap<String, Duration>>>,
}

impl ExecutionBudget {
    /// Create a new execution budget with the given total duration.
    pub fn new(total: Duration) -> Self {
        Self {
            total,
            start: Instant::now(),
            records: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Get the remaining time in the budget.
    pub fn remaining(&self) -> Duration {
        let elapsed = self.start.elapsed();
        self.total.saturating_sub(elapsed)
    }

    /// Check if the budget has been exhausted.
    pub fn is_exhausted(&self) -> bool {
        self.remaining() == Duration::ZERO
    }

    /// Allocate a timeout for the next node based on remaining budget.
    ///
    /// Returns the remaining time, which can be used as the timeout for
    /// the next node execution.
    pub fn allocate(&self, _node: &str) -> Duration {
        self.remaining()
    }

    /// Record how long a node took to execute.
    pub fn record_execution(&self, node: &str, elapsed: Duration) {
        let mut records = self.records.lock().unwrap();
        records.insert(node.to_string(), elapsed);
    }

    /// Get the execution time recorded for a specific node.
    pub fn get_execution_time(&self, node: &str) -> Option<Duration> {
        let records = self.records.lock().unwrap();
        records.get(node).copied()
    }

    /// Get the total elapsed time since the budget was created.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

/// Manages timeouts for graph node execution.
///
/// Combines a base `TimeoutConfig` with per-node overrides and provides
/// convenience methods for executing nodes with appropriate timeouts.
#[derive(Debug, Clone)]
pub struct TimeoutManager {
    /// Base configuration.
    config: TimeoutConfig,
    /// Per-node timeout overrides.
    node_timeouts: HashMap<String, Duration>,
}

impl TimeoutManager {
    /// Create a new `TimeoutManager` with the given configuration.
    pub fn new(config: TimeoutConfig) -> Self {
        Self {
            config,
            node_timeouts: HashMap::new(),
        }
    }

    /// Get the effective timeout for a specific node.
    ///
    /// Priority: per-node override > config node_timeout > config default_timeout.
    pub fn get_timeout_for_node(&self, node: &str) -> Duration {
        if let Some(&timeout) = self.node_timeouts.get(node) {
            return timeout;
        }
        self.config.effective_timeout()
    }

    /// Set a timeout override for a specific node.
    pub fn set_node_timeout(&mut self, node: &str, timeout: Duration) {
        self.node_timeouts.insert(node.to_string(), timeout);
    }

    /// Remove a per-node timeout override.
    pub fn remove_node_timeout(&mut self, node: &str) {
        self.node_timeouts.remove(node);
    }

    /// Execute a function with the appropriate timeout for the given node.
    pub async fn execute_with_timeout<F, Fut>(&self, node: &str, f: F) -> Result<Value>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<Value>>,
    {
        let timeout_duration = self.get_timeout_for_node(node);
        let node_timeout = NodeTimeout::new(timeout_duration, self.config.on_timeout.clone());
        node_timeout.execute(node, f).await
    }

    /// Execute a function with retry support using the appropriate timeout.
    pub async fn execute_with_retries<F, Fut>(&self, node: &str, factory: F) -> Result<Value>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<Value>>,
    {
        let timeout_duration = self.get_timeout_for_node(node);
        let node_timeout = NodeTimeout::new(timeout_duration, self.config.on_timeout.clone());
        node_timeout.execute_with_retries(node, factory).await
    }

    /// Get a reference to the underlying config.
    pub fn config(&self) -> &TimeoutConfig {
        &self.config
    }
}

impl Default for TimeoutManager {
    fn default() -> Self {
        Self::new(TimeoutConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::Ordering;

    // ── TimeoutConfig tests ──

    #[test]
    fn test_timeout_config_defaults() {
        let config = TimeoutConfig::new();
        assert_eq!(config.default_timeout, Duration::from_secs(30));
        assert!(config.node_timeout.is_none());
        assert!(config.graph_timeout.is_none());
        assert!(matches!(config.on_timeout, TimeoutAction::Error));
    }

    #[test]
    fn test_timeout_config_builder() {
        let config = TimeoutConfig::new()
            .with_node_timeout(Duration::from_secs(10))
            .with_graph_timeout(Duration::from_secs(120))
            .with_default_timeout(Duration::from_secs(5))
            .with_timeout_action(TimeoutAction::Skip);

        assert_eq!(config.node_timeout, Some(Duration::from_secs(10)));
        assert_eq!(config.graph_timeout, Some(Duration::from_secs(120)));
        assert_eq!(config.default_timeout, Duration::from_secs(5));
        assert!(matches!(config.on_timeout, TimeoutAction::Skip));
    }

    #[test]
    fn test_timeout_config_effective_timeout() {
        let config = TimeoutConfig::new();
        assert_eq!(config.effective_timeout(), Duration::from_secs(30));

        let config = config.with_node_timeout(Duration::from_secs(10));
        assert_eq!(config.effective_timeout(), Duration::from_secs(10));
    }

    // ── TimeoutError tests ──

    #[test]
    fn test_timeout_error_display() {
        let err = TimeoutError {
            node: "my_node".to_string(),
            elapsed: Duration::from_millis(1050),
            limit: Duration::from_secs(1),
        };
        let msg = err.to_string();
        assert!(msg.contains("my_node"));
        assert!(msg.contains("timed out"));
    }

    // ── NodeTimeout tests ──

    #[tokio::test]
    async fn test_node_timeout_success() {
        let nt = NodeTimeout::new(Duration::from_secs(5), TimeoutAction::Error);
        let result = nt
            .execute("fast_node", || async { Ok(json!({"status": "ok"})) })
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), json!({"status": "ok"}));
    }

    #[tokio::test]
    async fn test_node_timeout_error_on_timeout() {
        let nt = NodeTimeout::new(Duration::from_millis(50), TimeoutAction::Error);
        let result = nt
            .execute("slow_node", || async {
                tokio::time::sleep(Duration::from_secs(5)).await;
                Ok(json!({"done": true}))
            })
            .await;
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("slow_node"));
        assert!(err_msg.contains("timed out"));
    }

    #[tokio::test]
    async fn test_node_timeout_return_partial() {
        let partial = json!({"partial": true});
        let nt = NodeTimeout::new(
            Duration::from_millis(50),
            TimeoutAction::ReturnPartial(partial.clone()),
        );
        let result = nt
            .execute("slow_node", || async {
                tokio::time::sleep(Duration::from_secs(5)).await;
                Ok(json!({"done": true}))
            })
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), partial);
    }

    #[tokio::test]
    async fn test_node_timeout_skip() {
        let nt = NodeTimeout::new(Duration::from_millis(50), TimeoutAction::Skip);
        let result = nt
            .execute("slow_node", || async {
                tokio::time::sleep(Duration::from_secs(5)).await;
                Ok(json!({"done": true}))
            })
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Null);
    }

    #[tokio::test]
    async fn test_node_timeout_retry_with_retries() {
        let nt = NodeTimeout::new(
            Duration::from_millis(50),
            TimeoutAction::Retry { max_retries: 2 },
        );
        // All attempts will time out
        let result = nt
            .execute_with_retries("slow_node", || async {
                tokio::time::sleep(Duration::from_secs(5)).await;
                Ok(json!({"done": true}))
            })
            .await;
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("slow_node"));
        assert!(err_msg.contains("2 retries"));
    }

    #[tokio::test]
    async fn test_node_timeout_retry_succeeds_on_second_attempt() {
        use std::sync::atomic::AtomicUsize;

        let attempt = Arc::new(AtomicUsize::new(0));
        let nt = NodeTimeout::new(
            Duration::from_millis(100),
            TimeoutAction::Retry { max_retries: 3 },
        );

        let attempt_clone = attempt.clone();
        let result = nt
            .execute_with_retries("flaky_node", move || {
                let attempt = attempt_clone.clone();
                async move {
                    let n = attempt.fetch_add(1, Ordering::SeqCst);
                    if n == 0 {
                        // First attempt: sleep too long
                        tokio::time::sleep(Duration::from_secs(5)).await;
                    }
                    Ok(json!({"attempt": n + 1}))
                }
            })
            .await;

        assert!(result.is_ok());
        let val = result.unwrap();
        // Second attempt should succeed
        assert_eq!(val, json!({"attempt": 2}));
    }

    #[tokio::test]
    async fn test_node_timeout_propagates_inner_error() {
        let nt = NodeTimeout::new(Duration::from_secs(5), TimeoutAction::Error);
        let result = nt
            .execute("err_node", || async {
                Err(LangGraphError::Other("inner failure".to_string()))
            })
            .await;
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("inner failure"));
    }

    // ── CancellationToken tests ──

    #[test]
    fn test_cancellation_token_initial_state() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn test_cancellation_token_cancel() {
        let token = CancellationToken::new();
        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn test_cancellation_token_reset() {
        let token = CancellationToken::new();
        token.cancel();
        assert!(token.is_cancelled());
        token.reset();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn test_cancellation_token_clone() {
        let token = CancellationToken::new();
        let token2 = token.clone();
        token.cancel();
        assert!(token2.is_cancelled());
    }

    #[tokio::test]
    async fn test_cancellation_token_cancelled_future() {
        let token = CancellationToken::new();
        let token2 = token.clone();

        // Spawn a task that cancels after a short delay
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            token2.cancel();
        });

        // This should resolve once cancel is called
        tokio::time::timeout(Duration::from_secs(2), token.cancelled())
            .await
            .expect("cancelled() should resolve when token is cancelled");
    }

    #[tokio::test]
    async fn test_cancellation_token_already_cancelled() {
        let token = CancellationToken::new();
        token.cancel();
        // Should return immediately
        tokio::time::timeout(Duration::from_millis(50), token.cancelled())
            .await
            .expect("cancelled() should resolve immediately when already cancelled");
    }

    #[tokio::test]
    async fn test_cancellation_with_timeout() {
        let token = CancellationToken::new();
        let token_clone = token.clone();

        let nt = NodeTimeout::new(Duration::from_secs(10), TimeoutAction::Error);

        // Spawn cancellation
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            token_clone.cancel();
        });

        // Execute with select between node work and cancellation
        let result = nt
            .execute("cancellable_node", || async {
                tokio::select! {
                    _ = token.cancelled() => {
                        Err(LangGraphError::Other("cancelled".to_string()))
                    }
                    _ = tokio::time::sleep(Duration::from_secs(60)) => {
                        Ok(json!({"done": true}))
                    }
                }
            })
            .await;

        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("cancelled"));
    }

    // ── ExecutionBudget tests ──

    #[tokio::test]
    async fn test_execution_budget_remaining() {
        let budget = ExecutionBudget::new(Duration::from_secs(10));
        // Should have close to 10s remaining
        assert!(budget.remaining() > Duration::from_secs(9));
        assert!(!budget.is_exhausted());
    }

    #[tokio::test]
    async fn test_execution_budget_exhausted() {
        let budget = ExecutionBudget::new(Duration::from_millis(50));
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert!(budget.is_exhausted());
        assert_eq!(budget.remaining(), Duration::ZERO);
    }

    #[test]
    fn test_execution_budget_allocate() {
        let budget = ExecutionBudget::new(Duration::from_secs(10));
        let allocated = budget.allocate("node_a");
        // Should be close to the total budget
        assert!(allocated > Duration::from_secs(9));
    }

    #[test]
    fn test_execution_budget_record_execution() {
        let budget = ExecutionBudget::new(Duration::from_secs(60));
        budget.record_execution("node_a", Duration::from_millis(500));
        budget.record_execution("node_b", Duration::from_millis(300));

        assert_eq!(
            budget.get_execution_time("node_a"),
            Some(Duration::from_millis(500))
        );
        assert_eq!(
            budget.get_execution_time("node_b"),
            Some(Duration::from_millis(300))
        );
        assert_eq!(budget.get_execution_time("node_c"), None);
    }

    #[tokio::test]
    async fn test_execution_budget_with_node_timeout() {
        let budget = ExecutionBudget::new(Duration::from_secs(10));
        let allocated = budget.allocate("node_x");

        let nt = NodeTimeout::new(allocated, TimeoutAction::Error);
        let start = Instant::now();
        let result = nt
            .execute("node_x", || async { Ok(json!({"result": 42})) })
            .await;
        let elapsed = start.elapsed();

        assert!(result.is_ok());
        budget.record_execution("node_x", elapsed);
        assert!(budget.get_execution_time("node_x").unwrap() < Duration::from_secs(1));
    }

    // ── TimeoutManager tests ──

    #[test]
    fn test_timeout_manager_default_timeout() {
        let manager = TimeoutManager::default();
        assert_eq!(
            manager.get_timeout_for_node("any_node"),
            Duration::from_secs(30)
        );
    }

    #[test]
    fn test_timeout_manager_config_node_timeout() {
        let config = TimeoutConfig::new().with_node_timeout(Duration::from_secs(10));
        let manager = TimeoutManager::new(config);
        assert_eq!(
            manager.get_timeout_for_node("any_node"),
            Duration::from_secs(10)
        );
    }

    #[test]
    fn test_timeout_manager_per_node_override() {
        let config = TimeoutConfig::new().with_node_timeout(Duration::from_secs(10));
        let mut manager = TimeoutManager::new(config);
        manager.set_node_timeout("special_node", Duration::from_secs(60));

        assert_eq!(
            manager.get_timeout_for_node("regular_node"),
            Duration::from_secs(10)
        );
        assert_eq!(
            manager.get_timeout_for_node("special_node"),
            Duration::from_secs(60)
        );
    }

    #[test]
    fn test_timeout_manager_remove_override() {
        let mut manager = TimeoutManager::default();
        manager.set_node_timeout("node_a", Duration::from_secs(5));
        assert_eq!(
            manager.get_timeout_for_node("node_a"),
            Duration::from_secs(5)
        );

        manager.remove_node_timeout("node_a");
        assert_eq!(
            manager.get_timeout_for_node("node_a"),
            Duration::from_secs(30)
        );
    }

    #[tokio::test]
    async fn test_timeout_manager_execute_success() {
        let manager = TimeoutManager::default();
        let result = manager
            .execute_with_timeout("node_a", || async { Ok(json!({"ok": true})) })
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), json!({"ok": true}));
    }

    #[tokio::test]
    async fn test_timeout_manager_execute_timeout() {
        let config = TimeoutConfig::new()
            .with_default_timeout(Duration::from_millis(50))
            .with_timeout_action(TimeoutAction::Error);
        let manager = TimeoutManager::new(config);

        let result = manager
            .execute_with_timeout("slow", || async {
                tokio::time::sleep(Duration::from_secs(5)).await;
                Ok(json!({"done": true}))
            })
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_timeout_manager_execute_with_retries() {
        use std::sync::atomic::AtomicUsize;

        let config = TimeoutConfig::new()
            .with_default_timeout(Duration::from_millis(100))
            .with_timeout_action(TimeoutAction::Retry { max_retries: 2 });
        let manager = TimeoutManager::new(config);

        let attempt = Arc::new(AtomicUsize::new(0));
        let attempt_clone = attempt.clone();

        let result = manager
            .execute_with_retries("retry_node", move || {
                let attempt = attempt_clone.clone();
                async move {
                    let n = attempt.fetch_add(1, Ordering::SeqCst);
                    if n < 1 {
                        tokio::time::sleep(Duration::from_secs(5)).await;
                    }
                    Ok(json!({"attempt": n + 1}))
                }
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), json!({"attempt": 2}));
    }

    #[tokio::test]
    async fn test_timeout_manager_per_node_timeout_execution() {
        let config = TimeoutConfig::new()
            .with_default_timeout(Duration::from_millis(50))
            .with_timeout_action(TimeoutAction::Error);
        let mut manager = TimeoutManager::new(config);
        // Give this node more time
        manager.set_node_timeout("generous_node", Duration::from_secs(5));

        // Should succeed with generous timeout
        let result = manager
            .execute_with_timeout("generous_node", || async {
                tokio::time::sleep(Duration::from_millis(100)).await;
                Ok(json!({"ok": true}))
            })
            .await;
        assert!(result.is_ok());

        // Should fail with default tight timeout
        let result = manager
            .execute_with_timeout("tight_node", || async {
                tokio::time::sleep(Duration::from_millis(100)).await;
                Ok(json!({"ok": true}))
            })
            .await;
        assert!(result.is_err());
    }

    #[test]
    fn test_timeout_action_variants() {
        // Verify all variants can be constructed
        let _err = TimeoutAction::Error;
        let _partial = TimeoutAction::ReturnPartial(json!(null));
        let _skip = TimeoutAction::Skip;
        let _retry = TimeoutAction::Retry { max_retries: 3 };
    }

    #[test]
    fn test_cancellation_token_default() {
        let token = CancellationToken::default();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn test_timeout_manager_config_accessor() {
        let config = TimeoutConfig::new().with_graph_timeout(Duration::from_secs(60));
        let manager = TimeoutManager::new(config);
        assert_eq!(
            manager.config().graph_timeout,
            Some(Duration::from_secs(60))
        );
    }
}
