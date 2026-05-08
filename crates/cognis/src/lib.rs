//! # cognis
//!
//! v2-beta umbrella crate. Re-exports `cognis-core`, `cognis-graph`,
//! `cognis-llm`, `cognis-rag` and adds:
//! - [`agent`] — the standard ReAct agent loop.
//! - [`backend`] — agent workspace (in-memory or sandboxed real FS).
//! - [`middleware`] — composable LLM-call hooks (retry, fallback, caching,
//!   redaction, summarization, …).
//! - [`retrievers`] — LLM-driven retrievers (multi-query, contextual
//!   compression).

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

// Sub-crate re-exports.
pub use cognis_core;
pub use cognis_graph;
pub use cognis_llm;
pub use cognis_rag;

pub use cognis_core::{
    CharTokenizer, CognisError, Event, EventStream, Extensions, FnTokenizer, JsonSchema, Loader,
    Message, Observer, Result, Runnable, RunnableConfig, RunnableDefinition, RunnableStream,
    Serializable, Tokenizer, ToolCall,
};
pub use cognis_graph::{
    node_fn, ActiveSnapshot, AuditEntry, AuditKind, AuditLog, AuditLogObserver, Checkpointer,
    CompiledGraph, Goto, Graph, GraphMetrics, GraphSnapshot, GraphState, InMemoryAuditLog,
    InMemoryCheckpointer, MetricsObserver, Node, NodeCtx, NodeOut, NodeRetryPolicy, NodeTiming,
    ProfilingObserver, Subgraph,
};
pub use cognis_llm::{
    Aggregated, BaseTool, ChatOptions, ChatResponse, Client, ClientBuilder, LLMProvider, Provider,
    SchemaBasedTool, StreamAggregator, StreamChunk, Tool, ToolDefinition, ToolInput, ToolOutput,
    ToolRegistry, Usage, UsageTracker,
};
#[cfg(feature = "ollama")]
pub use cognis_rag::OllamaEmbeddings;
#[cfg(feature = "openai")]
pub use cognis_rag::OpenAIEmbeddings;
pub use cognis_rag::{
    CachingRetriever, CompressorPipeline, CrossEncoder, CrossEncoderReranker, Distance, Docstore,
    Document, Embeddings, FakeEmbeddings, Filter, FnCrossEncoder, InMemoryDocstore,
    InMemoryRecordManager, InMemoryVectorStore, IncrementalReport, LongContextReorder,
    MultiVectorIndexer, MultiVectorRetriever, ParentDocumentRetriever, QueryTranslatorRetriever,
    RecordManager, SearchResult, VectorStore,
};

// New stage-5 modules.
pub mod agent;
pub mod agent_bus;
pub mod agent_events;
pub mod backend;
#[cfg(feature = "cache-sqlite")]
pub mod cache_sqlite;
pub mod eval;
pub mod history;
pub mod middleware;
pub mod multi_agent;
pub mod observers;
pub mod presets;
pub mod retrievers;
pub mod session;
pub mod skills;
pub mod telemetry;
pub mod tools;

pub use agent::{
    default_react_graph, default_react_graph_with_limits, Agent, AgentBuilder, AgentHealth,
    AgentLifecycle, AgentPlugin, AgentResponse, AgentState, AgentStateUpdate, Buffer,
    ClosurePlugin, ConversationMode, EntityExtractor, EntityFact, EntityMemory, FnPlugin,
    HybridMemory, KnowledgeGraphMemory, LifecyclePlugin, Memory, OnStart, PluginRegistry,
    SummaryBufferMemory, SummaryMemory, ThinkNode, TokenBufferMemory, ToolDispatchNode, Triple,
    TripleExtractor, VectorMemory, Window, Workflow, WorkflowState, WorkflowStateUpdate,
};
pub use agent_bus::{AgentBus, SubscribeError, Subscription};
pub use agent_events::{AgentEvent, AgentEventBus, EventSubscription, DEFAULT_EVENTS_TOPIC};
pub use backend::{
    Backend, Blob, GrepHit, InMemoryStateBackend, InMemoryStorageBackend, LocalFsStorageBackend,
    MemoryBackend, SandboxedFsBackend, StateBackend, StorageBackend,
};
pub use eval::{
    Contains, EvalCase, EvalReport, EvalRow, EvalRunner, Evaluator, ExactMatch, LlmJudge,
};
pub use history::{
    HistoryStore, HistoryTrimmer, InMemoryHistory, RunnableWithMessageHistory, SessionKey,
    SessionResolver,
};
pub use middleware::{
    AlwaysSkip, ApprovalGate, AutoApproveAll, AutoRejectAll, CapMessageLength, ChatApproval,
    ChatApprover, ContextEditing, ContextInjection, ContextProvider, DropMatching, EditPolicy,
    EmulatorSource, FilesystemMiddleware, FixedRecovery, FnContextProvider, FnRecovery,
    FnToolCallPatcher, HumanDecision, HumanInTheLoop, HumanResponder, LimitTools, MapEmulator,
    Middleware, MiddlewareCtx, MiddlewarePipeline, ModelCallLimit, ModelFallback, ModelRetry, Next,
    PatchToolCalls, PiiRedactor, PipelinedClient, Planning, PromptCaching, RateLimit, RateLimiter,
    Recovery, RecoveryStrategy, RegexRedactor, SubagentMiddleware, SubagentRouter, Summarization,
    TodoMiddleware, TokenBucket, TokenCounter, ToolAllowList, ToolCallLimit, ToolCallPatcher,
    ToolDenyList, ToolEmulator, ToolFilter, ToolRetry, ToolRetryClassifier, ToolSelection,
    WorkspaceLister,
};
pub use multi_agent::{
    AgentMessage, Consensus, HandoffStrategy, Hierarchical, InMemoryMessageBus, MessageBus,
    MultiAgentOrchestrator, ParallelVote, RoundRobin, Sequential, Supervisor,
};
pub use observers::TracingObserver;
pub use retrievers::{
    ContextualCompressionRetriever, MultiQueryRetriever, RerankingRetriever, SearchSpec,
    SelfQueryRetriever, TimeWeightedRetriever,
};
pub use session::{InMemorySessionStore, Session, SessionStore, SessionStoreHandle};
pub use skills::{
    AllSkills, BuiltSkill, KeywordSelector, Skill, SkillBuilder, SkillRegistry, SkillSelector,
};
pub use telemetry::{
    InMemoryTelemetry, TelemetryEvent, TelemetryHandle, TelemetrySink, TelemetrySnapshot,
};
pub use tools::{
    register_filesystem_tools, AllowList, ApprovalGatedTool, Approver, AutoApprove, CachedTool,
    Calculator, CodeSanitizer, Decision, DotPathEngine, ExecutionPlan, FileEditTool,
    FileExistsTool, FileGlobTool, FileGrepTool, FileListTool, FileReadTool, FileWriteTool,
    HumanTool, JsonQueryTool, OrchestratorResult, PythonReplConfig, PythonReplTool, QueryEngine,
    RejectAll, RetrieverTool, SanitizationError, ShellTool, StaticResponder, SubAgentTool,
    ToolHumanResponder, ToolOrchestrator, ToolStep,
};
#[cfg(feature = "tools-http")]
pub use tools::{
    AuthScheme, BearerAuth, HeaderAuth, HttpMethod, HttpRequest, OpenApiToolset, TavilyProvider,
    TavilyProviderBuilder, WebSearchInput, WebSearchProvider, WebSearchResult, WebSearchTool,
    WikipediaAction, WikipediaTool, WikipediaToolBuilder,
};

/// Common imports for v2 user code building agents.
pub mod prelude {
    pub use crate::*;
    pub use crate::{Distance, Embeddings, InMemoryVectorStore, SearchResult, VectorStore};
    pub use async_trait::async_trait;
}
