//! Event bus system for agent lifecycle observability.
//!
//! Provides an [`EventBus`] that dispatches [`AgentEvent`]s to registered
//! [`EventHandler`] implementations. Consumers can subscribe handlers to
//! observe agent starts, completions, model calls, tool invocations,
//! middleware execution, and custom application events.
//!
//! # Built-in handlers
//!
//! - [`LoggingEventHandler`] -- logs events to stderr
//! - [`MetricsEventHandler`] -- collects call counts and average durations
//! - [`BufferEventHandler`] -- buffers the last N events for inspection
//!
//! # Example
//!
//! ```rust,ignore
//! use cognisagent::events::{EventBus, EventBusBuilder, AgentEventType, LoggingEventHandler};
//! use std::sync::Arc;
//!
//! let bus = EventBusBuilder::new()
//!     .handler(Arc::new(LoggingEventHandler))
//!     .build();
//!
//! bus.emit(AgentEventType::AgentStarted {
//!     agent_id: "a1".into(),
//!     config: serde_json::json!({}),
//! }).await.unwrap();
//! ```

use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};

/// Describes what happened during an agent lifecycle event.
#[derive(Debug, Clone)]
pub enum AgentEventType {
    /// An agent has started execution.
    AgentStarted { agent_id: String, config: Value },
    /// An agent completed successfully.
    AgentCompleted {
        agent_id: String,
        result: Value,
        duration: Duration,
    },
    /// An agent encountered an error.
    AgentError { agent_id: String, error: String },
    /// A model call is about to begin.
    ModelCallStart {
        model: String,
        messages_count: usize,
    },
    /// A model call completed.
    ModelCallEnd {
        model: String,
        tokens_used: usize,
        duration: Duration,
    },
    /// A tool call is about to begin.
    ToolCallStart { tool: String, input: String },
    /// A tool call completed successfully.
    ToolCallEnd {
        tool: String,
        output: String,
        duration: Duration,
    },
    /// A tool call failed.
    ToolCallError { tool: String, error: String },
    /// A middleware hook executed.
    MiddlewareExecuted {
        middleware: String,
        phase: String,
        duration: Duration,
    },
    /// A plan step was updated.
    PlanUpdated {
        plan_id: String,
        step: usize,
        status: String,
    },
    /// A tool invocation is paused awaiting explicit human approval. The
    /// `token` uniquely identifies this pending decision and must be passed
    /// back to the resolver to approve or reject.
    PendingApproval {
        token: String,
        tool: String,
        input: Value,
    },
    /// A previously-emitted pending approval has been resolved.
    ApprovalResolved {
        token: String,
        tool: String,
        approved: bool,
        reason: Option<String>,
    },
    /// A custom application-defined event.
    Custom { name: String, data: Value },
}

impl AgentEventType {
    /// Returns a short string name for the event type, suitable for filtering.
    pub fn name(&self) -> &str {
        match self {
            Self::AgentStarted { .. } => "AgentStarted",
            Self::AgentCompleted { .. } => "AgentCompleted",
            Self::AgentError { .. } => "AgentError",
            Self::ModelCallStart { .. } => "ModelCallStart",
            Self::ModelCallEnd { .. } => "ModelCallEnd",
            Self::ToolCallStart { .. } => "ToolCallStart",
            Self::ToolCallEnd { .. } => "ToolCallEnd",
            Self::ToolCallError { .. } => "ToolCallError",
            Self::MiddlewareExecuted { .. } => "MiddlewareExecuted",
            Self::PlanUpdated { .. } => "PlanUpdated",
            Self::PendingApproval { .. } => "PendingApproval",
            Self::ApprovalResolved { .. } => "ApprovalResolved",
            Self::Custom { name, .. } => name.as_str(),
        }
    }

    /// Returns the duration carried by this event, if any.
    pub fn duration(&self) -> Option<Duration> {
        match self {
            Self::AgentCompleted { duration, .. }
            | Self::ModelCallEnd { duration, .. }
            | Self::ToolCallEnd { duration, .. }
            | Self::MiddlewareExecuted { duration, .. } => Some(*duration),
            _ => None,
        }
    }
}

impl fmt::Display for AgentEventType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AgentStarted { agent_id, .. } => write!(f, "AgentStarted({})", agent_id),
            Self::AgentCompleted {
                agent_id, duration, ..
            } => write!(f, "AgentCompleted({}, {:?})", agent_id, duration),
            Self::AgentError { agent_id, error } => {
                write!(f, "AgentError({}: {})", agent_id, error)
            }
            Self::ModelCallStart {
                model,
                messages_count,
            } => write!(f, "ModelCallStart({}, {} msgs)", model, messages_count),
            Self::ModelCallEnd {
                model,
                tokens_used,
                duration,
            } => write!(
                f,
                "ModelCallEnd({}, {} tokens, {:?})",
                model, tokens_used, duration
            ),
            Self::ToolCallStart { tool, .. } => write!(f, "ToolCallStart({})", tool),
            Self::ToolCallEnd { tool, duration, .. } => {
                write!(f, "ToolCallEnd({}, {:?})", tool, duration)
            }
            Self::ToolCallError { tool, error } => {
                write!(f, "ToolCallError({}: {})", tool, error)
            }
            Self::MiddlewareExecuted {
                middleware,
                phase,
                duration,
            } => write!(
                f,
                "MiddlewareExecuted({}, {}, {:?})",
                middleware, phase, duration
            ),
            Self::PlanUpdated {
                plan_id,
                step,
                status,
            } => write!(f, "PlanUpdated({}, step {}, {})", plan_id, step, status),
            // Note: `token` is intentionally omitted from Display output.
            // Approval tokens are actionable credentials (anyone holding one
            // can approve/reject the invocation), so they must not leak into
            // logs. Consumers that need the token should read the struct field.
            Self::PendingApproval { tool, .. } => {
                write!(f, "PendingApproval(tool={}, token=<redacted>)", tool)
            }
            Self::ApprovalResolved { tool, approved, .. } => write!(
                f,
                "ApprovalResolved(tool={}, token=<redacted>, approved={})",
                tool, approved
            ),
            Self::Custom { name, .. } => write!(f, "Custom({})", name),
        }
    }
}

/// A timestamped, sequenced event emitted by the event bus.
#[derive(Debug, Clone)]
pub struct AgentEvent {
    /// The event payload.
    pub event_type: AgentEventType,
    /// When the event was created.
    pub timestamp: SystemTime,
    /// The run that produced this event.
    pub run_id: String,
    /// Monotonically increasing sequence number within this bus.
    pub sequence: u64,
}

/// Trait for receiving agent events.
///
/// Implementations can filter which events they receive by overriding
/// [`event_filter`](EventHandler::event_filter).
#[async_trait]
pub trait EventHandler: Send + Sync {
    /// Process an event. Errors are logged but do not stop dispatch.
    async fn handle(
        &self,
        event: &AgentEvent,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// If `Some`, only events whose [`AgentEventType::name`] is in the
    /// returned list will be dispatched to this handler.
    fn event_filter(&self) -> Option<Vec<String>> {
        None
    }
}

// ---------------------------------------------------------------------------
// Built-in handlers
// ---------------------------------------------------------------------------

/// Logs every received event to stderr.
#[derive(Debug, Clone)]
pub struct LoggingEventHandler;

#[async_trait]
impl EventHandler for LoggingEventHandler {
    async fn handle(
        &self,
        event: &AgentEvent,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        eprintln!(
            "[event seq={} run={}] {}",
            event.sequence, event.run_id, event.event_type
        );
        Ok(())
    }
}

/// Collects per-event-type call counts and cumulative durations.
#[derive(Debug)]
pub struct MetricsEventHandler {
    metrics: RwLock<HashMap<String, MetricEntry>>,
}

/// A single metric entry.
#[derive(Debug, Clone, Default)]
pub struct MetricEntry {
    /// Total number of times this event type was seen.
    pub count: u64,
    /// Sum of durations for events that carry one.
    pub total_duration: Duration,
}

impl MetricEntry {
    /// Average duration per event (returns zero if count is 0).
    pub fn avg_duration(&self) -> Duration {
        if self.count == 0 {
            Duration::ZERO
        } else {
            self.total_duration / self.count as u32
        }
    }
}

impl MetricsEventHandler {
    /// Create a new, empty metrics handler.
    pub fn new() -> Self {
        Self {
            metrics: RwLock::new(HashMap::new()),
        }
    }

    /// Return a snapshot of the current metrics.
    pub fn snapshot(&self) -> HashMap<String, MetricEntry> {
        self.metrics.read().unwrap().clone()
    }
}

impl Default for MetricsEventHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl EventHandler for MetricsEventHandler {
    async fn handle(
        &self,
        event: &AgentEvent,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let name = event.event_type.name().to_string();
        let duration = event.event_type.duration().unwrap_or(Duration::ZERO);
        let mut metrics = self.metrics.write().unwrap();
        let entry = metrics.entry(name).or_default();
        entry.count += 1;
        entry.total_duration += duration;
        Ok(())
    }
}

/// Buffers the last N events in a ring buffer for later inspection.
#[derive(Debug)]
pub struct BufferEventHandler {
    capacity: usize,
    events: RwLock<Vec<AgentEvent>>,
}

impl BufferEventHandler {
    /// Create a buffer that retains at most `capacity` events.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            events: RwLock::new(Vec::with_capacity(capacity)),
        }
    }

    /// Return a snapshot of the buffered events (oldest first).
    pub fn snapshot(&self) -> Vec<AgentEvent> {
        self.events.read().unwrap().clone()
    }

    /// Number of events currently buffered.
    pub fn len(&self) -> usize {
        self.events.read().unwrap().len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[async_trait]
impl EventHandler for BufferEventHandler {
    async fn handle(
        &self,
        event: &AgentEvent,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut buf = self.events.write().unwrap();
        if buf.len() >= self.capacity {
            buf.remove(0);
        }
        buf.push(event.clone());
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// EventBus
// ---------------------------------------------------------------------------

/// Central dispatcher that fans out events to registered handlers.
///
/// Thread-safe: cloning an `EventBus` shares the same internal state.
#[derive(Clone)]
pub struct EventBus {
    inner: Arc<EventBusInner>,
}

struct EventBusInner {
    handlers: RwLock<Vec<Arc<dyn EventHandler>>>,
    run_id: String,
    sequence: AtomicU64,
}

impl fmt::Debug for EventBus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EventBus")
            .field("run_id", &self.inner.run_id)
            .field("sequence", &self.inner.sequence.load(Ordering::Relaxed))
            .field("handler_count", &self.inner.handlers.read().unwrap().len())
            .finish()
    }
}

impl EventBus {
    /// Create a new event bus with no handlers and a generated run id.
    pub fn new() -> Self {
        Self::with_run_id(uuid::Uuid::new_v4().to_string())
    }

    /// Create a new event bus with a specific run id.
    pub fn with_run_id(run_id: String) -> Self {
        Self {
            inner: Arc::new(EventBusInner {
                handlers: RwLock::new(Vec::new()),
                run_id,
                sequence: AtomicU64::new(0),
            }),
        }
    }

    /// Register a handler. Returns its index (usable with [`unsubscribe`](Self::unsubscribe)).
    pub fn subscribe(&self, handler: Arc<dyn EventHandler>) -> usize {
        let mut handlers = self.inner.handlers.write().unwrap();
        let idx = handlers.len();
        handlers.push(handler);
        idx
    }

    /// Remove a handler by index. No-op if the index is out of range.
    pub fn unsubscribe(&self, handler_id: usize) {
        let mut handlers = self.inner.handlers.write().unwrap();
        if handler_id < handlers.len() {
            handlers.remove(handler_id);
        }
    }

    /// Number of registered handlers.
    pub fn handler_count(&self) -> usize {
        self.inner.handlers.read().unwrap().len()
    }

    /// The run id associated with this bus.
    pub fn run_id(&self) -> &str {
        &self.inner.run_id
    }

    /// Current sequence counter value (next event will get this number).
    pub fn current_sequence(&self) -> u64 {
        self.inner.sequence.load(Ordering::Relaxed)
    }

    /// Create an [`AgentEvent`], dispatch it to all matching handlers, and return it.
    pub async fn emit(
        &self,
        event_type: AgentEventType,
    ) -> Result<AgentEvent, Box<dyn std::error::Error + Send + Sync>> {
        let event = AgentEvent {
            timestamp: SystemTime::now(),
            run_id: self.inner.run_id.clone(),
            sequence: self.inner.sequence.fetch_add(1, Ordering::Relaxed),
            event_type,
        };

        let handlers = self.inner.handlers.read().unwrap().clone();
        let event_name = event.event_type.name().to_string();

        for handler in &handlers {
            if let Some(filter) = handler.event_filter() {
                if !filter.iter().any(|f| f == &event_name) {
                    continue;
                }
            }
            // Errors from individual handlers are logged, not propagated.
            if let Err(e) = handler.handle(&event).await {
                eprintln!("[EventBus] handler error: {}", e);
            }
        }

        Ok(event)
    }

    /// Fire-and-forget: spawns a tokio task to emit the event.
    pub fn emit_sync(&self, event_type: AgentEventType) {
        let bus = self.clone();
        tokio::spawn(async move {
            let _ = bus.emit(event_type).await;
        });
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for constructing an [`EventBus`] with handlers pre-registered.
#[derive(Default)]
pub struct EventBusBuilder {
    handlers: Vec<Arc<dyn EventHandler>>,
    run_id: Option<String>,
    buffer_size: Option<usize>,
}

impl EventBusBuilder {
    /// Start building.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a handler.
    pub fn handler(mut self, h: Arc<dyn EventHandler>) -> Self {
        self.handlers.push(h);
        self
    }

    /// Set the run id (defaults to a generated UUID).
    pub fn run_id(mut self, id: impl Into<String>) -> Self {
        self.run_id = Some(id.into());
        self
    }

    /// Automatically add a [`BufferEventHandler`] with the given capacity.
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = Some(size);
        self
    }

    /// Build the event bus.
    pub fn build(self) -> EventBus {
        let bus = match self.run_id {
            Some(id) => EventBus::with_run_id(id),
            None => EventBus::new(),
        };
        for h in self.handlers {
            bus.subscribe(h);
        }
        if let Some(size) = self.buffer_size {
            bus.subscribe(Arc::new(BufferEventHandler::new(size)));
        }
        bus
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// Simple counter handler for testing.
    #[derive(Debug)]
    struct CountingHandler {
        count: AtomicUsize,
        filter: Option<Vec<String>>,
    }

    impl CountingHandler {
        fn new() -> Self {
            Self {
                count: AtomicUsize::new(0),
                filter: None,
            }
        }

        fn with_filter(filter: Vec<&str>) -> Self {
            Self {
                count: AtomicUsize::new(0),
                filter: Some(filter.into_iter().map(String::from).collect()),
            }
        }

        fn count(&self) -> usize {
            self.count.load(Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl EventHandler for CountingHandler {
        async fn handle(
            &self,
            _event: &AgentEvent,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            self.count.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn event_filter(&self) -> Option<Vec<String>> {
            self.filter.clone()
        }
    }

    /// Handler that always returns an error.
    struct FailingHandler;

    #[async_trait]
    impl EventHandler for FailingHandler {
        async fn handle(
            &self,
            _event: &AgentEvent,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Err("intentional failure".into())
        }
    }

    fn agent_started() -> AgentEventType {
        AgentEventType::AgentStarted {
            agent_id: "a1".into(),
            config: serde_json::json!({}),
        }
    }

    fn tool_call_end() -> AgentEventType {
        AgentEventType::ToolCallEnd {
            tool: "search".into(),
            output: "ok".into(),
            duration: Duration::from_millis(42),
        }
    }

    // 1. Basic emit produces an event
    #[tokio::test]
    async fn test_emit_returns_event() {
        let bus = EventBus::with_run_id("run-1".into());
        let event = bus.emit(agent_started()).await.unwrap();
        assert_eq!(event.run_id, "run-1");
        assert_eq!(event.sequence, 0);
    }

    // 2. Sequence increments
    #[tokio::test]
    async fn test_sequence_increments() {
        let bus = EventBus::new();
        let e1 = bus.emit(agent_started()).await.unwrap();
        let e2 = bus.emit(agent_started()).await.unwrap();
        assert_eq!(e1.sequence, 0);
        assert_eq!(e2.sequence, 1);
    }

    // 3. Handler receives events
    #[tokio::test]
    async fn test_handler_receives_events() {
        let bus = EventBus::new();
        let h = Arc::new(CountingHandler::new());
        bus.subscribe(h.clone());
        bus.emit(agent_started()).await.unwrap();
        bus.emit(agent_started()).await.unwrap();
        assert_eq!(h.count(), 2);
    }

    // 4. Multiple handlers
    #[tokio::test]
    async fn test_multiple_handlers() {
        let bus = EventBus::new();
        let h1 = Arc::new(CountingHandler::new());
        let h2 = Arc::new(CountingHandler::new());
        bus.subscribe(h1.clone());
        bus.subscribe(h2.clone());
        bus.emit(agent_started()).await.unwrap();
        assert_eq!(h1.count(), 1);
        assert_eq!(h2.count(), 1);
    }

    // 5. Unsubscribe removes handler
    #[tokio::test]
    async fn test_unsubscribe() {
        let bus = EventBus::new();
        let h = Arc::new(CountingHandler::new());
        let id = bus.subscribe(h.clone());
        bus.emit(agent_started()).await.unwrap();
        assert_eq!(h.count(), 1);
        bus.unsubscribe(id);
        bus.emit(agent_started()).await.unwrap();
        assert_eq!(h.count(), 1);
    }

    // 6. Event filter — handler only gets matching events
    #[tokio::test]
    async fn test_event_filter() {
        let bus = EventBus::new();
        let h = Arc::new(CountingHandler::with_filter(vec!["ToolCallEnd"]));
        bus.subscribe(h.clone());
        bus.emit(agent_started()).await.unwrap();
        assert_eq!(h.count(), 0);
        bus.emit(tool_call_end()).await.unwrap();
        assert_eq!(h.count(), 1);
    }

    // 7. No filter means all events
    #[tokio::test]
    async fn test_no_filter_receives_all() {
        let bus = EventBus::new();
        let h = Arc::new(CountingHandler::new());
        bus.subscribe(h.clone());
        bus.emit(agent_started()).await.unwrap();
        bus.emit(tool_call_end()).await.unwrap();
        bus.emit(AgentEventType::AgentError {
            agent_id: "a1".into(),
            error: "oops".into(),
        })
        .await
        .unwrap();
        assert_eq!(h.count(), 3);
    }

    // 8. Failing handler does not break dispatch
    #[tokio::test]
    async fn test_failing_handler_does_not_break_dispatch() {
        let bus = EventBus::new();
        let good = Arc::new(CountingHandler::new());
        bus.subscribe(Arc::new(FailingHandler));
        bus.subscribe(good.clone());
        bus.emit(agent_started()).await.unwrap();
        assert_eq!(good.count(), 1);
    }

    // 9. BufferEventHandler stores events
    #[tokio::test]
    async fn test_buffer_handler_stores_events() {
        let buf = Arc::new(BufferEventHandler::new(10));
        let bus = EventBus::new();
        bus.subscribe(buf.clone());
        bus.emit(agent_started()).await.unwrap();
        bus.emit(tool_call_end()).await.unwrap();
        assert_eq!(buf.len(), 2);
        assert!(!buf.is_empty());
    }

    // 10. BufferEventHandler respects capacity
    #[tokio::test]
    async fn test_buffer_handler_capacity() {
        let buf = Arc::new(BufferEventHandler::new(2));
        let bus = EventBus::new();
        bus.subscribe(buf.clone());
        for _ in 0..5 {
            bus.emit(agent_started()).await.unwrap();
        }
        assert_eq!(buf.len(), 2);
        let snap = buf.snapshot();
        assert_eq!(snap[0].sequence, 3);
        assert_eq!(snap[1].sequence, 4);
    }

    // 11. MetricsEventHandler counts
    #[tokio::test]
    async fn test_metrics_handler_counts() {
        let m = Arc::new(MetricsEventHandler::new());
        let bus = EventBus::new();
        bus.subscribe(m.clone());
        bus.emit(agent_started()).await.unwrap();
        bus.emit(agent_started()).await.unwrap();
        let snap = m.snapshot();
        assert_eq!(snap["AgentStarted"].count, 2);
    }

    // 12. MetricsEventHandler tracks duration
    #[tokio::test]
    async fn test_metrics_handler_duration() {
        let m = Arc::new(MetricsEventHandler::new());
        let bus = EventBus::new();
        bus.subscribe(m.clone());
        bus.emit(tool_call_end()).await.unwrap();
        let snap = m.snapshot();
        assert_eq!(
            snap["ToolCallEnd"].total_duration,
            Duration::from_millis(42)
        );
        assert_eq!(
            snap["ToolCallEnd"].avg_duration(),
            Duration::from_millis(42)
        );
    }

    // 13. MetricsEventHandler average duration
    #[tokio::test]
    async fn test_metrics_handler_avg_duration() {
        let m = Arc::new(MetricsEventHandler::new());
        let bus = EventBus::new();
        bus.subscribe(m.clone());
        bus.emit(AgentEventType::ToolCallEnd {
            tool: "t".into(),
            output: "o".into(),
            duration: Duration::from_millis(10),
        })
        .await
        .unwrap();
        bus.emit(AgentEventType::ToolCallEnd {
            tool: "t".into(),
            output: "o".into(),
            duration: Duration::from_millis(30),
        })
        .await
        .unwrap();
        let snap = m.snapshot();
        assert_eq!(
            snap["ToolCallEnd"].avg_duration(),
            Duration::from_millis(20)
        );
    }

    // 14. EventBusBuilder basic build
    #[tokio::test]
    async fn test_builder_basic() {
        let h = Arc::new(CountingHandler::new());
        let bus = EventBusBuilder::new()
            .run_id("builder-run")
            .handler(h.clone())
            .build();
        assert_eq!(bus.run_id(), "builder-run");
        bus.emit(agent_started()).await.unwrap();
        assert_eq!(h.count(), 1);
    }

    // 15. EventBusBuilder with buffer_size
    #[tokio::test]
    async fn test_builder_buffer_size() {
        let bus = EventBusBuilder::new().buffer_size(5).build();
        // buffer handler is auto-added
        assert_eq!(bus.handler_count(), 1);
    }

    // 16. AgentEventType::name
    #[test]
    fn test_event_type_names() {
        assert_eq!(agent_started().name(), "AgentStarted");
        assert_eq!(tool_call_end().name(), "ToolCallEnd");
        assert_eq!(
            AgentEventType::Custom {
                name: "MyEvent".into(),
                data: Value::Null
            }
            .name(),
            "MyEvent"
        );
    }

    // 17. AgentEventType::duration
    #[test]
    fn test_event_type_duration() {
        assert!(agent_started().duration().is_none());
        assert_eq!(tool_call_end().duration(), Some(Duration::from_millis(42)));
    }

    // 18. Display impl
    #[test]
    fn test_event_type_display() {
        let s = format!("{}", agent_started());
        assert!(s.contains("AgentStarted"));
        assert!(s.contains("a1"));
    }

    // 19. handler_count
    #[tokio::test]
    async fn test_handler_count() {
        let bus = EventBus::new();
        assert_eq!(bus.handler_count(), 0);
        let h = Arc::new(CountingHandler::new());
        bus.subscribe(h);
        assert_eq!(bus.handler_count(), 1);
    }

    // 20. Event bus clone shares state
    #[tokio::test]
    async fn test_clone_shares_state() {
        let bus = EventBus::new();
        let h = Arc::new(CountingHandler::new());
        bus.subscribe(h.clone());
        let bus2 = bus.clone();
        bus2.emit(agent_started()).await.unwrap();
        assert_eq!(h.count(), 1);
        assert_eq!(bus.current_sequence(), 1);
    }

    // 21. All event types can be emitted
    #[tokio::test]
    async fn test_all_event_types_emit() {
        let h = Arc::new(CountingHandler::new());
        let bus = EventBus::new();
        bus.subscribe(h.clone());

        let types = vec![
            AgentEventType::AgentStarted {
                agent_id: "a".into(),
                config: Value::Null,
            },
            AgentEventType::AgentCompleted {
                agent_id: "a".into(),
                result: Value::Null,
                duration: Duration::ZERO,
            },
            AgentEventType::AgentError {
                agent_id: "a".into(),
                error: "e".into(),
            },
            AgentEventType::ModelCallStart {
                model: "m".into(),
                messages_count: 1,
            },
            AgentEventType::ModelCallEnd {
                model: "m".into(),
                tokens_used: 100,
                duration: Duration::ZERO,
            },
            AgentEventType::ToolCallStart {
                tool: "t".into(),
                input: "i".into(),
            },
            AgentEventType::ToolCallEnd {
                tool: "t".into(),
                output: "o".into(),
                duration: Duration::ZERO,
            },
            AgentEventType::ToolCallError {
                tool: "t".into(),
                error: "e".into(),
            },
            AgentEventType::MiddlewareExecuted {
                middleware: "mw".into(),
                phase: "before".into(),
                duration: Duration::ZERO,
            },
            AgentEventType::PlanUpdated {
                plan_id: "p".into(),
                step: 0,
                status: "done".into(),
            },
            AgentEventType::PendingApproval {
                token: "tok-1".into(),
                tool: "shell".into(),
                input: Value::Null,
            },
            AgentEventType::ApprovalResolved {
                token: "tok-1".into(),
                tool: "shell".into(),
                approved: true,
                reason: None,
            },
            AgentEventType::Custom {
                name: "c".into(),
                data: Value::Null,
            },
        ];

        for t in types {
            bus.emit(t).await.unwrap();
        }
        assert_eq!(h.count(), 13);
    }

    // 22. LoggingEventHandler does not error
    #[tokio::test]
    async fn test_logging_handler_no_error() {
        let bus = EventBus::new();
        bus.subscribe(Arc::new(LoggingEventHandler));
        let result = bus.emit(agent_started()).await;
        assert!(result.is_ok());
    }

    // 23. Multiple filters on different handlers
    #[tokio::test]
    async fn test_multiple_filters() {
        let bus = EventBus::new();
        let h_agent = Arc::new(CountingHandler::with_filter(vec![
            "AgentStarted",
            "AgentError",
        ]));
        let h_tool = Arc::new(CountingHandler::with_filter(vec!["ToolCallEnd"]));
        bus.subscribe(h_agent.clone());
        bus.subscribe(h_tool.clone());

        bus.emit(agent_started()).await.unwrap();
        bus.emit(tool_call_end()).await.unwrap();
        bus.emit(AgentEventType::ModelCallStart {
            model: "m".into(),
            messages_count: 1,
        })
        .await
        .unwrap();

        assert_eq!(h_agent.count(), 1);
        assert_eq!(h_tool.count(), 1);
    }

    // 24. Unsubscribe out of bounds is a no-op
    #[tokio::test]
    async fn test_unsubscribe_out_of_bounds() {
        let bus = EventBus::new();
        bus.unsubscribe(999); // should not panic
        assert_eq!(bus.handler_count(), 0);
    }

    // 25. Timestamp is reasonable
    #[tokio::test]
    async fn test_event_timestamp() {
        let before = SystemTime::now();
        let bus = EventBus::new();
        let event = bus.emit(agent_started()).await.unwrap();
        let after = SystemTime::now();
        assert!(event.timestamp >= before);
        assert!(event.timestamp <= after);
    }
}
