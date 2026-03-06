use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use uuid::Uuid;

use crate::callbacks::CallbackHandler;
use crate::error::Result;

use super::base::Runnable;
use super::config::RunnableConfig;

/// Configuration for a scoped callback context.
///
/// Manages both local callbacks (specific to this scope) and inherited callbacks
/// (propagated from a parent scope). The `propagate` flag controls whether local
/// callbacks are passed down to child scopes.
#[derive(Clone)]
pub struct ScopedCallbackConfig {
    /// Callbacks only for this scope.
    pub local_callbacks: Vec<Arc<dyn CallbackHandler>>,
    /// Inherited from parent scope.
    pub inherited_callbacks: Vec<Arc<dyn CallbackHandler>>,
    /// Whether local callbacks propagate to children (default true).
    pub propagate: bool,
    /// Scope tags for filtering.
    pub tags: Vec<String>,
    /// Scope metadata.
    pub metadata: HashMap<String, Value>,
}

impl Default for ScopedCallbackConfig {
    fn default() -> Self {
        Self {
            local_callbacks: Vec::new(),
            inherited_callbacks: Vec::new(),
            propagate: true,
            tags: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

impl std::fmt::Debug for ScopedCallbackConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScopedCallbackConfig")
            .field(
                "local_callbacks",
                &format!("[{} handlers]", self.local_callbacks.len()),
            )
            .field(
                "inherited_callbacks",
                &format!("[{} handlers]", self.inherited_callbacks.len()),
            )
            .field("propagate", &self.propagate)
            .field("tags", &self.tags)
            .field("metadata", &self.metadata)
            .finish()
    }
}

/// RAII guard returned by `CallbackScope::enter()`.
///
/// While this guard exists, the scope is considered "active". Dropping the guard
/// exits the scope. This follows the RAII pattern for scope lifetime management.
pub struct ScopeGuard {
    _scope_id: Uuid,
}

impl ScopeGuard {
    fn new(scope_id: Uuid) -> Self {
        Self {
            _scope_id: scope_id,
        }
    }

    /// Returns the scope ID associated with this guard.
    pub fn scope_id(&self) -> Uuid {
        self._scope_id
    }
}

/// Manages the lifetime of a callback scope.
///
/// A `CallbackScope` holds a `ScopedCallbackConfig` and provides methods to
/// create child scopes that inherit callbacks, add or remove callbacks, and
/// enter the scope with RAII-based lifetime management.
pub struct CallbackScope {
    id: Uuid,
    config: ScopedCallbackConfig,
}

impl CallbackScope {
    /// Create a new scope with the given configuration.
    pub fn new(config: ScopedCallbackConfig) -> Self {
        Self {
            id: Uuid::new_v4(),
            config,
        }
    }

    /// Returns the unique ID of this scope.
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Returns a reference to this scope's configuration.
    pub fn config(&self) -> &ScopedCallbackConfig {
        &self.config
    }

    /// Enter this scope, returning an RAII guard.
    ///
    /// The scope is considered active while the guard exists.
    pub fn enter(&self) -> ScopeGuard {
        ScopeGuard::new(self.id)
    }

    /// Create a child scope that inherits callbacks from this scope.
    ///
    /// Inherited callbacks in the child are: this scope's inherited callbacks,
    /// plus this scope's local callbacks if `propagate` is true.
    /// The child starts with no local callbacks of its own.
    /// Tags and metadata are also inherited.
    pub fn child_scope(&self) -> CallbackScope {
        let child_config = merge_callback_configs(&self.config, &ScopedCallbackConfig::default());
        CallbackScope::new(child_config)
    }

    /// Returns all callbacks: inherited first, then local.
    pub fn all_callbacks(&self) -> Vec<Arc<dyn CallbackHandler>> {
        let mut all = self.config.inherited_callbacks.clone();
        all.extend(self.config.local_callbacks.iter().cloned());
        all
    }

    /// Add a callback to this scope's local callbacks.
    pub fn with_callback(&mut self, handler: Arc<dyn CallbackHandler>) {
        self.config.local_callbacks.push(handler);
    }

    /// Filter out callbacks whose `name()` matches the type name of `T`.
    ///
    /// This removes matching callbacks from both local and inherited lists.
    pub fn without_callback_type<T: CallbackHandler + 'static>(&mut self) {
        let type_name = std::any::type_name::<T>();
        self.config
            .local_callbacks
            .retain(|h| !handler_is_type::<T>(h.as_ref(), type_name));
        self.config
            .inherited_callbacks
            .retain(|h| !handler_is_type::<T>(h.as_ref(), type_name));
    }
}

/// Check if a handler matches the given type name.
///
/// Uses the `name()` method on `CallbackHandler` and also checks `type_name`.
fn handler_is_type<T: 'static>(handler: &dyn CallbackHandler, type_name: &str) -> bool {
    // Check if the handler's name contains the type name (for custom names)
    // or if the full type name matches
    let handler_name = handler.name();
    // Try exact match on the short type name
    let short_name = type_name.rsplit("::").next().unwrap_or(type_name);
    handler_name == short_name || handler_name == type_name
}

/// A wrapper around a `Runnable` that injects scoped callbacks during execution.
///
/// When `invoke()` is called, the scoped callbacks are merged into the
/// `RunnableConfig` before delegating to the inner runnable. This allows
/// callbacks to be attached to specific runnables without affecting siblings.
pub struct RunnableWithCallbacks {
    inner: Arc<dyn Runnable>,
    scoped_config: ScopedCallbackConfig,
}

impl RunnableWithCallbacks {
    /// Create a new wrapper with scoped callbacks around the given runnable.
    pub fn new(inner: Arc<dyn Runnable>, scoped_config: ScopedCallbackConfig) -> Self {
        Self {
            inner,
            scoped_config,
        }
    }

    /// Returns a reference to the inner runnable.
    pub fn inner(&self) -> &Arc<dyn Runnable> {
        &self.inner
    }

    /// Returns a reference to the scoped callback configuration.
    pub fn scoped_config(&self) -> &ScopedCallbackConfig {
        &self.scoped_config
    }
}

#[async_trait]
impl Runnable for RunnableWithCallbacks {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        let base_config = config.cloned().unwrap_or_default();

        // Merge scoped callbacks into the config
        let mut merged_callbacks = base_config.callbacks.clone();
        merged_callbacks.extend(self.scoped_config.inherited_callbacks.iter().cloned());
        merged_callbacks.extend(self.scoped_config.local_callbacks.iter().cloned());

        let mut merged_tags = base_config.tags.clone();
        merged_tags.extend(self.scoped_config.tags.iter().cloned());

        let mut merged_metadata = base_config.metadata.clone();
        merged_metadata.extend(
            self.scoped_config
                .metadata
                .iter()
                .map(|(k, v)| (k.clone(), v.clone())),
        );

        let merged_config = RunnableConfig {
            callbacks: merged_callbacks,
            tags: merged_tags,
            metadata: merged_metadata,
            ..base_config
        };

        self.inner.invoke(input, Some(&merged_config)).await
    }
}

/// Dispatches callback events to multiple handlers.
///
/// Handles errors gracefully: if one handler fails, the error is logged
/// and dispatch continues to remaining handlers. Supports tag-based filtering.
pub struct CallbackDispatcher;

/// A callback event that can be dispatched to handlers.
#[derive(Debug, Clone)]
pub struct CallbackEvent {
    /// Event name (e.g., "on_chain_start", "on_llm_end").
    pub name: String,
    /// Event data payload.
    pub data: Value,
    /// Run ID for this event.
    pub run_id: Uuid,
    /// Optional parent run ID.
    pub parent_run_id: Option<Uuid>,
    /// Tags associated with this event for filtering.
    pub tags: Vec<String>,
}

impl CallbackEvent {
    /// Create a new callback event.
    pub fn new(name: impl Into<String>, data: Value, run_id: Uuid) -> Self {
        Self {
            name: name.into(),
            data,
            run_id,
            parent_run_id: None,
            tags: Vec::new(),
        }
    }

    /// Set tags on this event.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Set parent run ID on this event.
    pub fn with_parent_run_id(mut self, parent_run_id: Uuid) -> Self {
        self.parent_run_id = Some(parent_run_id);
        self
    }
}

impl CallbackDispatcher {
    /// Dispatch an event to all callbacks, handling errors gracefully.
    ///
    /// If a handler returns an error and `raise_error()` is false (the default),
    /// the error is logged via `eprintln!` and dispatch continues.
    /// If `raise_error()` is true, the error is returned immediately.
    pub async fn dispatch_to_all(
        callbacks: &[Arc<dyn CallbackHandler>],
        event: &CallbackEvent,
    ) -> Result<()> {
        for handler in callbacks {
            let result = handler
                .on_custom_event(&event.name, &event.data, event.run_id, event.parent_run_id)
                .await;

            if let Err(e) = result {
                if handler.raise_error() {
                    return Err(e);
                }
                eprintln!(
                    "Callback handler '{}' error (ignored): {}",
                    handler.name(),
                    e
                );
            }
        }
        Ok(())
    }

    /// Dispatch an event to callbacks, filtering by tags.
    ///
    /// Only dispatches to handlers whose `name()` is contained in the event's
    /// tags, or dispatches to all if the event has no tags.
    pub async fn dispatch_with_tag_filter(
        callbacks: &[Arc<dyn CallbackHandler>],
        event: &CallbackEvent,
        filter_tags: &[String],
    ) -> Result<()> {
        if filter_tags.is_empty() {
            return Self::dispatch_to_all(callbacks, event).await;
        }

        for handler in callbacks {
            // If filter tags are specified, only dispatch to handlers whose name
            // appears in the filter tags list.
            if !filter_tags.contains(&handler.name().to_string()) {
                continue;
            }

            let result = handler
                .on_custom_event(&event.name, &event.data, event.run_id, event.parent_run_id)
                .await;

            if let Err(e) = result {
                if handler.raise_error() {
                    return Err(e);
                }
                eprintln!(
                    "Callback handler '{}' error (ignored): {}",
                    handler.name(),
                    e
                );
            }
        }
        Ok(())
    }
}

/// Wrap a runnable with scoped callbacks.
///
/// This is a convenience function for creating a `RunnableWithCallbacks`.
pub fn with_callbacks(
    runnable: Arc<dyn Runnable>,
    callbacks: Vec<Arc<dyn CallbackHandler>>,
) -> RunnableWithCallbacks {
    let config = ScopedCallbackConfig {
        local_callbacks: callbacks,
        ..ScopedCallbackConfig::default()
    };
    RunnableWithCallbacks::new(runnable, config)
}

/// Merge a parent scope config with a child scope config.
///
/// The result's inherited callbacks are: parent's inherited + parent's local
/// (if parent propagates). The result's local callbacks are the child's local
/// callbacks. Tags and metadata are merged (parent + child).
pub fn merge_callback_configs(
    parent: &ScopedCallbackConfig,
    child: &ScopedCallbackConfig,
) -> ScopedCallbackConfig {
    let mut inherited = parent.inherited_callbacks.clone();
    if parent.propagate {
        inherited.extend(parent.local_callbacks.iter().cloned());
    }

    let mut tags = parent.tags.clone();
    tags.extend(child.tags.iter().cloned());

    let mut metadata = parent.metadata.clone();
    metadata.extend(child.metadata.iter().map(|(k, v)| (k.clone(), v.clone())));

    ScopedCallbackConfig {
        local_callbacks: child.local_callbacks.clone(),
        inherited_callbacks: inherited,
        propagate: child.propagate,
        tags,
        metadata,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::callbacks::handlers::{LogLevel, LoggingCallbackHandler, MetricsCallbackHandler};
    use serde_json::json;

    // ---------------------------------------------------------------
    // Test helpers
    // ---------------------------------------------------------------

    /// A test callback handler that records calls.
    struct TestCallbackHandler {
        handler_name: String,
        call_count: std::sync::atomic::AtomicUsize,
    }

    impl TestCallbackHandler {
        fn new(name: &str) -> Self {
            Self {
                handler_name: name.to_string(),
                call_count: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.call_count
                .load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl CallbackHandler for TestCallbackHandler {
        fn name(&self) -> &str {
            &self.handler_name
        }

        async fn on_custom_event(
            &self,
            _name: &str,
            _data: &Value,
            _run_id: Uuid,
            _parent_run_id: Option<Uuid>,
        ) -> Result<()> {
            self.call_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        }
    }

    /// A callback handler that always fails.
    struct FailingCallbackHandler {
        should_raise: bool,
    }

    impl FailingCallbackHandler {
        fn new(should_raise: bool) -> Self {
            Self { should_raise }
        }
    }

    #[async_trait]
    impl CallbackHandler for FailingCallbackHandler {
        fn name(&self) -> &str {
            "FailingCallbackHandler"
        }

        fn raise_error(&self) -> bool {
            self.should_raise
        }

        async fn on_custom_event(
            &self,
            _name: &str,
            _data: &Value,
            _run_id: Uuid,
            _parent_run_id: Option<Uuid>,
        ) -> Result<()> {
            Err(crate::error::RustChainError::Other(
                "handler failure".to_string(),
            ))
        }
    }

    /// Simple test runnable that returns its input.
    struct EchoRunnable;

    #[async_trait]
    impl Runnable for EchoRunnable {
        fn name(&self) -> &str {
            "EchoRunnable"
        }

        async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
            // Return the number of callbacks in the config as proof they were merged
            let cb_count = config.map(|c| c.callbacks.len()).unwrap_or(0);
            Ok(json!({
                "input": input,
                "callback_count": cb_count,
            }))
        }
    }

    // ---------------------------------------------------------------
    // Tests
    // ---------------------------------------------------------------

    #[test]
    fn test_scoped_callback_config_defaults() {
        let config = ScopedCallbackConfig::default();
        assert!(config.local_callbacks.is_empty());
        assert!(config.inherited_callbacks.is_empty());
        assert!(config.propagate);
        assert!(config.tags.is_empty());
        assert!(config.metadata.is_empty());
    }

    #[test]
    fn test_scoped_callback_config_creation_with_values() {
        let handler = Arc::new(TestCallbackHandler::new("test"));
        let config = ScopedCallbackConfig {
            local_callbacks: vec![handler.clone()],
            inherited_callbacks: vec![],
            propagate: false,
            tags: vec!["tag1".into()],
            metadata: {
                let mut m = HashMap::new();
                m.insert("key".into(), json!("value"));
                m
            },
        };
        assert_eq!(config.local_callbacks.len(), 1);
        assert!(!config.propagate);
        assert_eq!(config.tags, vec!["tag1".to_string()]);
        assert_eq!(config.metadata.get("key"), Some(&json!("value")));
    }

    #[test]
    fn test_callback_inheritance_parent_to_child() {
        let parent_handler = Arc::new(TestCallbackHandler::new("parent"));
        let local_handler = Arc::new(TestCallbackHandler::new("local"));

        let parent_config = ScopedCallbackConfig {
            local_callbacks: vec![local_handler.clone()],
            inherited_callbacks: vec![parent_handler.clone()],
            propagate: true,
            tags: vec!["parent_tag".into()],
            metadata: HashMap::new(),
        };

        let scope = CallbackScope::new(parent_config);
        let child = scope.child_scope();

        // Child should inherit: parent's inherited + parent's local (propagate=true)
        assert_eq!(child.config().inherited_callbacks.len(), 2);
        // Child has no local callbacks of its own
        assert!(child.config().local_callbacks.is_empty());
        // Child inherits tags
        assert!(child.config().tags.contains(&"parent_tag".to_string()));
    }

    #[test]
    fn test_propagation_flag_false_blocks_local() {
        let parent_inherited = Arc::new(TestCallbackHandler::new("inherited"));
        let parent_local = Arc::new(TestCallbackHandler::new("local_no_propagate"));

        let parent_config = ScopedCallbackConfig {
            local_callbacks: vec![parent_local.clone()],
            inherited_callbacks: vec![parent_inherited.clone()],
            propagate: false,
            tags: vec![],
            metadata: HashMap::new(),
        };

        let scope = CallbackScope::new(parent_config);
        let child = scope.child_scope();

        // With propagate=false, child only gets parent's inherited, not local
        assert_eq!(child.config().inherited_callbacks.len(), 1);
        assert_eq!(
            child.config().inherited_callbacks[0].name(),
            "inherited"
        );
    }

    #[tokio::test]
    async fn test_runnable_with_callbacks_invoke() {
        let echo = Arc::new(EchoRunnable) as Arc<dyn Runnable>;
        let handler1 = Arc::new(TestCallbackHandler::new("h1"));
        let handler2 = Arc::new(TestCallbackHandler::new("h2"));

        let scoped_config = ScopedCallbackConfig {
            local_callbacks: vec![handler1, handler2],
            inherited_callbacks: vec![],
            propagate: true,
            tags: vec!["scope_tag".into()],
            metadata: HashMap::new(),
        };

        let wrapped = RunnableWithCallbacks::new(echo, scoped_config);
        let result = wrapped.invoke(json!("hello"), None).await.unwrap();

        // The EchoRunnable reports callback count from merged config
        assert_eq!(result["callback_count"], 2);
        assert_eq!(result["input"], "hello");
    }

    #[test]
    fn test_merge_callback_configs_parent_child() {
        let p_handler = Arc::new(TestCallbackHandler::new("parent_h"));
        let p_local = Arc::new(TestCallbackHandler::new("parent_local"));
        let c_handler = Arc::new(TestCallbackHandler::new("child_h"));

        let parent = ScopedCallbackConfig {
            local_callbacks: vec![p_local.clone()],
            inherited_callbacks: vec![p_handler.clone()],
            propagate: true,
            tags: vec!["ptag".into()],
            metadata: {
                let mut m = HashMap::new();
                m.insert("pk".into(), json!("pv"));
                m
            },
        };

        let child = ScopedCallbackConfig {
            local_callbacks: vec![c_handler.clone()],
            inherited_callbacks: vec![],
            propagate: true,
            tags: vec!["ctag".into()],
            metadata: {
                let mut m = HashMap::new();
                m.insert("ck".into(), json!("cv"));
                m
            },
        };

        let merged = merge_callback_configs(&parent, &child);

        // Inherited: parent's inherited + parent's local (propagate=true)
        assert_eq!(merged.inherited_callbacks.len(), 2);
        // Local: child's local
        assert_eq!(merged.local_callbacks.len(), 1);
        assert_eq!(merged.local_callbacks[0].name(), "child_h");
        // Tags merged
        assert!(merged.tags.contains(&"ptag".to_string()));
        assert!(merged.tags.contains(&"ctag".to_string()));
        // Metadata merged
        assert_eq!(merged.metadata.get("pk"), Some(&json!("pv")));
        assert_eq!(merged.metadata.get("ck"), Some(&json!("cv")));
    }

    #[tokio::test]
    async fn test_tag_based_filtering() {
        let h1 = Arc::new(TestCallbackHandler::new("handler_a"));
        let h2 = Arc::new(TestCallbackHandler::new("handler_b"));
        let callbacks: Vec<Arc<dyn CallbackHandler>> = vec![h1.clone(), h2.clone()];

        let event = CallbackEvent::new("test_event", json!({}), Uuid::new_v4());

        // Filter to only handler_a
        CallbackDispatcher::dispatch_with_tag_filter(
            &callbacks,
            &event,
            &["handler_a".to_string()],
        )
        .await
        .unwrap();

        assert_eq!(h1.calls(), 1);
        assert_eq!(h2.calls(), 0);
    }

    #[test]
    fn test_metadata_propagation_to_child() {
        let parent = ScopedCallbackConfig {
            local_callbacks: vec![],
            inherited_callbacks: vec![],
            propagate: true,
            tags: vec!["parent_t".into()],
            metadata: {
                let mut m = HashMap::new();
                m.insert("env".into(), json!("prod"));
                m.insert("shared".into(), json!("parent_val"));
                m
            },
        };

        let child = ScopedCallbackConfig {
            local_callbacks: vec![],
            inherited_callbacks: vec![],
            propagate: true,
            tags: vec!["child_t".into()],
            metadata: {
                let mut m = HashMap::new();
                m.insert("shared".into(), json!("child_val"));
                m.insert("child_only".into(), json!(true));
                m
            },
        };

        let merged = merge_callback_configs(&parent, &child);

        // Parent metadata is present
        assert_eq!(merged.metadata.get("env"), Some(&json!("prod")));
        // Child overrides shared key
        assert_eq!(merged.metadata.get("shared"), Some(&json!("child_val")));
        // Child-only key is present
        assert_eq!(merged.metadata.get("child_only"), Some(&json!(true)));
        // Both tag sets merged
        assert_eq!(merged.tags.len(), 2);
    }

    #[tokio::test]
    async fn test_multiple_nested_scopes() {
        let h_root = Arc::new(TestCallbackHandler::new("root"));
        let h_mid = Arc::new(TestCallbackHandler::new("mid"));
        let h_leaf = Arc::new(TestCallbackHandler::new("leaf"));

        // Root scope
        let root_config = ScopedCallbackConfig {
            local_callbacks: vec![h_root.clone()],
            inherited_callbacks: vec![],
            propagate: true,
            tags: vec!["root".into()],
            metadata: HashMap::new(),
        };
        let root_scope = CallbackScope::new(root_config);

        // Middle scope (child of root)
        let mut mid_scope = root_scope.child_scope();
        mid_scope.with_callback(h_mid.clone());

        // Leaf scope (child of middle)
        let mut leaf_scope = mid_scope.child_scope();
        leaf_scope.with_callback(h_leaf.clone());

        let all = leaf_scope.all_callbacks();
        // inherited from root (1) + mid's local propagated (1) + leaf's local (1)
        assert_eq!(all.len(), 3);
        // Order: inherited first, then local
        assert_eq!(all[0].name(), "root");
        assert_eq!(all[1].name(), "mid");
        assert_eq!(all[2].name(), "leaf");
    }

    #[tokio::test]
    async fn test_dispatcher_error_handling_continues() {
        let failing = Arc::new(FailingCallbackHandler::new(false));
        let good = Arc::new(TestCallbackHandler::new("good"));

        let callbacks: Vec<Arc<dyn CallbackHandler>> =
            vec![failing as Arc<dyn CallbackHandler>, good.clone()];

        let event = CallbackEvent::new("test", json!({}), Uuid::new_v4());

        // Should not fail even though first handler errors
        let result = CallbackDispatcher::dispatch_to_all(&callbacks, &event).await;
        assert!(result.is_ok());

        // Good handler should still have been called
        assert_eq!(good.calls(), 1);
    }

    #[tokio::test]
    async fn test_dispatcher_raise_error_stops_dispatch() {
        let failing = Arc::new(FailingCallbackHandler::new(true));
        let good = Arc::new(TestCallbackHandler::new("good"));

        let callbacks: Vec<Arc<dyn CallbackHandler>> =
            vec![failing as Arc<dyn CallbackHandler>, good.clone()];

        let event = CallbackEvent::new("test", json!({}), Uuid::new_v4());

        // Should fail because first handler has raise_error=true
        let result = CallbackDispatcher::dispatch_to_all(&callbacks, &event).await;
        assert!(result.is_err());

        // Good handler should NOT have been called
        assert_eq!(good.calls(), 0);
    }

    #[test]
    fn test_without_callback_type_filtering() {
        let logging = Arc::new(LoggingCallbackHandler::new(LogLevel::Info));
        let metrics = Arc::new(MetricsCallbackHandler::new());

        let config = ScopedCallbackConfig {
            local_callbacks: vec![
                logging as Arc<dyn CallbackHandler>,
                metrics as Arc<dyn CallbackHandler>,
            ],
            inherited_callbacks: vec![],
            propagate: true,
            tags: vec![],
            metadata: HashMap::new(),
        };

        let mut scope = CallbackScope::new(config);
        assert_eq!(scope.all_callbacks().len(), 2);

        scope.without_callback_type::<LoggingCallbackHandler>();
        let remaining = scope.all_callbacks();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].name(), "MetricsCallbackHandler");
    }

    #[test]
    fn test_empty_scope_behavior() {
        let config = ScopedCallbackConfig::default();
        let scope = CallbackScope::new(config);

        assert!(scope.all_callbacks().is_empty());

        let child = scope.child_scope();
        assert!(child.all_callbacks().is_empty());
        assert!(child.config().tags.is_empty());
        assert!(child.config().metadata.is_empty());
    }

    #[test]
    fn test_all_callbacks_ordering_inherited_first() {
        let inherited_h = Arc::new(TestCallbackHandler::new("inherited"));
        let local_h = Arc::new(TestCallbackHandler::new("local"));

        let config = ScopedCallbackConfig {
            local_callbacks: vec![local_h],
            inherited_callbacks: vec![inherited_h],
            propagate: true,
            tags: vec![],
            metadata: HashMap::new(),
        };

        let scope = CallbackScope::new(config);
        let all = scope.all_callbacks();

        assert_eq!(all.len(), 2);
        assert_eq!(all[0].name(), "inherited");
        assert_eq!(all[1].name(), "local");
    }

    #[test]
    fn test_with_callbacks_helper() {
        let runnable = Arc::new(EchoRunnable) as Arc<dyn Runnable>;
        let handler = Arc::new(TestCallbackHandler::new("helper_test"));

        let wrapped = with_callbacks(runnable, vec![handler as Arc<dyn CallbackHandler>]);
        assert_eq!(wrapped.scoped_config().local_callbacks.len(), 1);
        assert_eq!(wrapped.name(), "EchoRunnable");
    }

    #[test]
    fn test_scope_guard_returns_scope_id() {
        let config = ScopedCallbackConfig::default();
        let scope = CallbackScope::new(config);
        let guard = scope.enter();
        assert_eq!(guard.scope_id(), scope.id());
    }
}
