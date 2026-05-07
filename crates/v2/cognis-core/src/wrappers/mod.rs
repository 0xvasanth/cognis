//! Resilience and policy wrappers for any `Runnable`.
//!
//! - [`Retry`] — exponential backoff on retryable errors.
//! - [`Fallback`] — switch to a backup runnable on error.
//! - [`Timeout`] — bound per-invocation duration.
//! - [`Cache`] — memoize by an input-derived key.
//!
//! All wrappers are themselves `Runnable`s, so they compose with `Pipe`
//! (and `|`). The fluent shorthand lives on [`crate::RunnableExt`]:
//!
//! ```ignore
//! let resilient = model
//!     .with_max_retries(3)
//!     .with_timeout(Duration::from_secs(30))
//!     .with_fallback(backup_model);
//! ```

pub mod assign;
pub mod bind;
pub mod cache;
pub mod configurable;
pub mod fallback;
pub mod retry;
pub mod timeout;

pub use assign::{Assign, AssignOutput};
pub use bind::Bind;
pub use cache::{Cache, CacheBackend, MemoryCache};
pub use configurable::{ConfigKey, Configurable};
pub use fallback::Fallback;
pub use retry::{Retry, RetryPolicy};
pub use timeout::Timeout;
