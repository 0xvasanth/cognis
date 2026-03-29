pub mod assign;
pub mod base;
pub mod batch;
pub mod batch_processor;
pub mod binding;
pub mod branch;
pub mod cache;
pub mod config;
pub mod configurable;
pub mod each;
pub mod error_handling;
pub mod events;
pub mod ext;
pub mod fallbacks;
pub mod graph;
pub mod history;
pub mod lambda;
pub mod middleware;
pub mod parallel;
pub mod passthrough;
pub mod pipe;
pub mod pipeline;
pub mod rate_limit;
pub mod retry;
pub mod router;
pub mod router_ext;
pub mod schema;
pub mod scoped_callbacks;
pub mod sequence;
pub mod serialization;
pub mod timeout;
pub mod tracing;

use std::pin::Pin;

use futures::Stream;
use serde_json::Value;

use crate::error::Result;

/// A pinned, boxed stream of runnable results.
pub type RunnableStream = Pin<Box<dyn Stream<Item = Result<Value>> + Send>>;

// Re-exports
pub use assign::{RunnableAssign, RunnablePick};
pub use base::Runnable;
pub use batch::{batch_invoke, RunnableBatch};
pub use batch_processor::{
    batch_process, BatchConfig, BatchItemResult, BatchProgress, BatchResult, ChunkIterator,
    RunnableBatchProcessor,
};
pub use binding::RunnableBinding;
pub use branch::RunnableBranch;
pub use cache::{
    CacheConfig, CacheEntry, CacheInvalidator, CacheKey, CacheStats, CachedRunnable,
    EvictionPolicy, RunnableCache,
};
pub use config::{
    ensure_config, get_config_list, merge_configs, patch_config, ConfigPatch, RunnableConfig,
};
pub use configurable::{
    ConfigurableAlternatives, ConfigurableField, ConfigurableFieldType, ConfigurableSpec,
    RunnableConfigurableFields,
};
pub use each::RunnableEach;
pub use error_handling::{
    CircuitBreaker, CircuitState, ClassifiedError, ErrorAction, ErrorChain, ErrorClassifier,
    ErrorHandler, ErrorKind, MapErrorHandler, PatternErrorClassifier, RecoveryHandler,
};
pub use events::{EventEmitter, EventFilter, EventTrace, RunEvent, RunEventType};
pub use ext::RunnableExt;
pub use fallbacks::{with_fallbacks, RunnableWithFallbacks};
pub use graph::{
    AsciiExporter, Branch, DotExporter, DotOptions, Edge, Graph, GraphEdge, GraphNode,
    MermaidExporter, Node, NodeData, NodeType, RunnableGraph,
};
pub use history::{ConfigurableFieldSpec, RunnableWithMessageHistory};
pub use lambda::RunnableLambda;
pub use middleware::{
    LoggingMiddleware, MiddlewareAction, MiddlewareChain, RetryMiddleware, RunnableMiddleware,
    RunnableWithMiddleware, TimingMiddleware, TransformMiddleware, ValidationMiddleware,
};
pub use parallel::RunnableParallel;
pub use passthrough::RunnablePassthrough;
pub use pipe::{pipe as pipe_fn, PipeBuilder, RunnablePipe, RunnableRef};
pub use pipeline::{
    ParallelPipeline, Pipeline, PipelineBuilder, PipelineMetrics, PipelineResult, PipelineStage,
};
pub use rate_limit::{
    ConcurrencyGuard, ConcurrencyLimiter, RateLimitConfig, RateLimiter, RunnableRateLimit,
    RunnableThrottle, SlidingWindowCounter,
};
pub use retry::RunnableRetry;
pub use router::{RouterRunnable, RunnableFnBranch, RunnableRouter};
pub use router_ext::{
    ConditionalBranch, ContentRouter, FallbackRouter, KeyRouter, RegexRouter, RoutingTable,
};
pub use schema::{
    EventData, RunnableSchema, SchemaContract, SchemaError, SchemaField, SchemaInference,
    SchemaType, SchemaValidationResult, StreamEvent,
};
pub use scoped_callbacks::{
    merge_callback_configs, with_callbacks, CallbackDispatcher, CallbackEvent, CallbackScope,
    RunnableWithCallbacks, ScopeGuard, ScopedCallbackConfig,
};
pub use sequence::RunnableSequence;
pub use serialization::{
    ConfigDiff, ConfigMigration, ConfigRegistry, ConfigSchema, ParamSpec, ParamType,
    SerializableConfig, SerializableConfigBuilder,
};
pub use timeout::{RunnableDeadline, RunnableTimeout, TimeoutBehavior, TimeoutConfig};
pub use tracing::{
    CompactTraceFormatter, JsonTraceFormatter, SpanResult, TextTraceFormatter, TraceCollector,
    TraceEntry, TraceExporter, TraceFormatter, TraceLevel, TraceSpan,
};

/// Creates a `RunnableSequence` from a list of runnables.
///
/// # Example
/// ```ignore
/// let seq = chain!(lambda1, lambda2, lambda3);
/// ```
#[macro_export]
macro_rules! chain {
    ($($step:expr),+ $(,)?) => {
        $crate::runnables::RunnableSequence::new(vec![
            $(std::sync::Arc::new($step) as std::sync::Arc<dyn $crate::runnables::Runnable>,)+
        ])
    };
}
