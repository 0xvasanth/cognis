pub mod base;
pub mod batch;
pub mod binding;
pub mod branch;
pub mod config;
pub mod configurable;
pub mod each;
pub mod events;
pub mod ext;
pub mod fallbacks;
pub mod graph;
pub mod history;
pub mod lambda;
pub mod parallel;
pub mod passthrough;
pub mod pipe;
pub mod retry;
pub mod router;
pub mod schema;
pub mod sequence;

use std::pin::Pin;

use futures::Stream;
use serde_json::Value;

use crate::error::Result;

/// A pinned, boxed stream of runnable results.
pub type RunnableStream = Pin<Box<dyn Stream<Item = Result<Value>> + Send>>;

// Re-exports
pub use base::Runnable;
pub use binding::RunnableBinding;
pub use branch::RunnableBranch;
pub use config::{ensure_config, get_config_list, merge_configs, patch_config, ConfigPatch, RunnableConfig};
pub use each::RunnableEach;
pub use fallbacks::{RunnableWithFallbacks, with_fallbacks};
pub use lambda::RunnableLambda;
pub use parallel::RunnableParallel;
pub use passthrough::{RunnableAssign, RunnablePassthrough, RunnablePick};
pub use router::{RouterRunnable, RunnableRouter, RunnableFnBranch};
pub use configurable::{ConfigurableField, RunnableConfigurableFields};
pub use retry::RunnableRetry;
pub use sequence::RunnableSequence;
pub use pipe::{PipeBuilder, RunnablePipe, pipe as pipe_fn};
pub use ext::RunnableExt;
pub use graph::{Branch, Edge, Graph, Node, NodeData};
pub use history::{ConfigurableFieldSpec, RunnableWithMessageHistory};
pub use batch::{batch_invoke, RunnableBatch};

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
