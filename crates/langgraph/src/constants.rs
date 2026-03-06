//! LangGraph constants.

/// Tag to disable streaming for a chat model.
pub const TAG_NOSTREAM: &str = "nostream";
/// Tag to hide a node/edge from tracing/streaming.
pub const TAG_HIDDEN: &str = "langsmith:hidden";
/// The last (maybe virtual) node in graph-style Pregel.
pub const END: &str = "__end__";
/// The first (maybe virtual) node in graph-style Pregel.
pub const START: &str = "__start__";

// Internal constants (used by pregel execution engine)
#[allow(dead_code)]
pub(crate) const INTERRUPT: &str = "__interrupt__";
#[allow(dead_code)]
pub(crate) const TASKS: &str = "__pregel_tasks";
#[allow(dead_code)]
pub(crate) const NS_SEP: &str = "|";
#[allow(dead_code)]
pub(crate) const NS_END: &str = ":";
#[allow(dead_code)]
pub(crate) const RESUME: &str = "__pregel_resume";

// Config key constants
#[allow(dead_code)]
pub const CONFIG_KEY_CHECKPOINTER: &str = "checkpointer";
#[allow(dead_code)]
pub const CONFIG_KEY_STREAM: &str = "__pregel_stream";
#[allow(dead_code)]
pub const CONFIG_KEY_SEND: &str = "__pregel_send";
#[allow(dead_code)]
pub const CONFIG_KEY_READ: &str = "__pregel_read";
#[allow(dead_code)]
pub const CONFIG_KEY_CALL: &str = "__pregel_call";
#[allow(dead_code)]
pub const CONFIG_KEY_CACHE: &str = "__pregel_cache";
#[allow(dead_code)]
pub const CONFIG_KEY_RESUMING: &str = "__pregel_resuming";
#[allow(dead_code)]
pub const CONFIG_KEY_TASK_ID: &str = "task_id";
#[allow(dead_code)]
pub const CONFIG_KEY_THREAD_ID: &str = "thread_id";
#[allow(dead_code)]
pub const CONFIG_KEY_CHECKPOINT_MAP: &str = "checkpoint_map";
#[allow(dead_code)]
pub const CONFIG_KEY_CHECKPOINT_ID: &str = "checkpoint_id";
#[allow(dead_code)]
pub const CONFIG_KEY_CHECKPOINT_NS: &str = "checkpoint_ns";
#[allow(dead_code)]
pub const CONFIG_KEY_NODE_FINISHED: &str = "__pregel_node_finished";
#[allow(dead_code)]
pub const CONFIG_KEY_SCRATCHPAD: &str = "__pregel_scratchpad";
#[allow(dead_code)]
pub const CONFIG_KEY_DURABILITY: &str = "__pregel_durability";
#[allow(dead_code)]
pub const CONFIG_KEY_RUNTIME: &str = "__pregel_runtime";

// Internal marker values
#[allow(dead_code)]
pub const INPUT: &str = "__input__";
#[allow(dead_code)]
pub const ERROR: &str = "__error__";
#[allow(dead_code)]
pub const NO_WRITES: &str = "__no_writes__";
#[allow(dead_code)]
pub const PUSH: &str = "__push__";
#[allow(dead_code)]
pub const PULL: &str = "__pull__";
