//! Composition primitives for `Runnable`.
//!
//! - [`Pipe`] — sequential `A | B`. The signature DX of cognis.
//! - [`Parallel`] — fan out one input to many runnables.
//! - [`Branch`] — predicate-based dispatch.
//! - [`Lambda`] — wrap an async or sync closure as a `Runnable`.
//! - [`Passthrough`] — identity runnable.
//! - [`Each`] — apply a runnable to every element of a `Vec<I>`.
//!
//! For any `Runnable`, the [`crate::RunnableExt`] trait adds fluent
//! `.pipe(next)`, `.each()`, etc.
//!
//! ```ignore
//! use cognis_core::RunnableExt;
//! let chain = prompt.pipe(model).pipe(parser);
//! ```

pub mod branch;
pub mod each;
pub mod lambda;
pub mod parallel;
pub mod passthrough;
pub mod pipe;

pub use branch::{Branch, BranchCase};
pub use each::Each;
pub use lambda::{lambda, Lambda};
pub use parallel::Parallel;
pub use passthrough::Passthrough;
pub use pipe::{pipe, Pipe};
