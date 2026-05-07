//! Compile-time setting controlling **when** checkpoints are persisted
//! relative to a superstep's execution.
//!
//! Mirrors V1's `Durability` modes:
//!
//! - [`Durability::Sync`] (default) — `cp.save(...)` is awaited inline at
//!   the end of every superstep. Strongest guarantee; the next step never
//!   starts until the previous step's state is durably persisted. Used in
//!   tests and production deployments where loss of a single step's work
//!   is unacceptable.
//!
//! - [`Durability::Async`] — `cp.save(...)` is spawned and the engine
//!   advances without awaiting. The save races with the next superstep;
//!   on a crash you may lose up to one step. Use when checkpoint backends
//!   are slow (network/postgres) and step boundaries are frequent.
//!
//! - [`Durability::Exit`] — only the final state is persisted (one
//!   `cp.save` call when the graph reaches `End` or runs out of work).
//!   Cheapest; suitable for short, non-resumable workflows where
//!   intermediate snapshots aren't useful.
//!
//! - [`Durability::Every`] (extension) — save every N steps. Takes a
//!   stride and falls back to one of the above modes for every Nth step.
//!
//! Custom strategies plug in via [`DurabilityHook`] — see the trait for
//! how to pass a fully custom decision function.

use std::sync::Arc;

/// Decision for one superstep's checkpoint timing. Returned by a
/// [`DurabilityHook`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurabilityDecision {
    /// Await `cp.save` inline.
    Sync,
    /// Spawn `cp.save`.
    Async,
    /// Skip persisting this step.
    Skip,
}

/// Object-safe hook called per step to decide when to checkpoint. Plugs
/// into [`Durability::Custom`] for fully bespoke policies (e.g. "save
/// every 10 steps to S3", "sync only on tool-result steps").
pub trait DurabilityHook: Send + Sync {
    /// Decide whether/how to persist `step` for `run_id`.
    /// `is_terminal` is true on the final emitted decision (graph End).
    fn decide(&self, step: u64, is_terminal: bool) -> DurabilityDecision;
}

/// Convenience: any `Fn(u64, bool) -> DurabilityDecision + Send + Sync`
/// is a `DurabilityHook`.
impl<F> DurabilityHook for F
where
    F: Fn(u64, bool) -> DurabilityDecision + Send + Sync,
{
    fn decide(&self, step: u64, is_terminal: bool) -> DurabilityDecision {
        (self)(step, is_terminal)
    }
}

/// Checkpoint timing relative to step execution.
#[derive(Clone, Default)]
pub enum Durability {
    /// Await `cp.save` inline after each step (default).
    #[default]
    Sync,
    /// Spawn `cp.save` without awaiting. Crash before the spawn lands
    /// loses the most recent step.
    Async,
    /// Save only at graph completion. No intermediate checkpoints.
    Exit,
    /// Save every Nth step (n>=1) using the wrapped sub-mode for that
    /// step. Other steps are skipped. The graph-completion save is
    /// always emitted regardless of stride.
    Every {
        /// Stride. 1 means "every step" (equivalent to the wrapped mode).
        n: u64,
        /// Mode used on the Nth step.
        mode: Box<Durability>,
    },
    /// Fully user-defined policy.
    Custom(Arc<dyn DurabilityHook>),
}

impl std::fmt::Debug for Durability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sync => f.write_str("Sync"),
            Self::Async => f.write_str("Async"),
            Self::Exit => f.write_str("Exit"),
            Self::Every { n, mode } => f
                .debug_struct("Every")
                .field("n", n)
                .field("mode", mode)
                .finish(),
            Self::Custom(_) => f.write_str("Custom(<hook>)"),
        }
    }
}

impl PartialEq for Durability {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Sync, Self::Sync) | (Self::Async, Self::Async) | (Self::Exit, Self::Exit) => {
                true
            }
            (Self::Every { n: a, mode: ma }, Self::Every { n: b, mode: mb }) => a == b && ma == mb,
            // Two `Custom` impls are never equal (no way to compare hooks).
            _ => false,
        }
    }
}

impl Durability {
    /// Decide what action to take at the end of `step`. `is_terminal`
    /// is true on the graph-completion save (always honored except for
    /// `Skip`-returning custom hooks that opt out).
    pub fn decide(&self, step: u64, is_terminal: bool) -> DurabilityDecision {
        match self {
            Self::Sync => DurabilityDecision::Sync,
            Self::Async => DurabilityDecision::Async,
            Self::Exit => {
                if is_terminal {
                    DurabilityDecision::Sync
                } else {
                    DurabilityDecision::Skip
                }
            }
            Self::Every { n, mode } => {
                if is_terminal {
                    return DurabilityDecision::Sync;
                }
                let stride = (*n).max(1);
                if step.is_multiple_of(stride) {
                    mode.decide(step, false)
                } else {
                    DurabilityDecision::Skip
                }
            }
            Self::Custom(h) => h.decide(step, is_terminal),
        }
    }

    /// True if the engine should save inline after each step.
    pub fn save_per_step_sync(&self) -> bool {
        matches!(self, Self::Sync)
    }

    /// True if the engine should spawn an async save after each step.
    pub fn save_per_step_async(&self) -> bool {
        matches!(self, Self::Async)
    }

    /// True if a final-only save should be emitted on graph completion.
    pub fn save_on_exit(&self) -> bool {
        matches!(self, Self::Exit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_sync() {
        assert_eq!(Durability::default(), Durability::Sync);
    }

    #[test]
    fn predicates_match_variants() {
        assert!(Durability::Sync.save_per_step_sync());
        assert!(!Durability::Sync.save_per_step_async());
        assert!(!Durability::Sync.save_on_exit());

        assert!(!Durability::Async.save_per_step_sync());
        assert!(Durability::Async.save_per_step_async());
        assert!(!Durability::Async.save_on_exit());

        assert!(!Durability::Exit.save_per_step_sync());
        assert!(!Durability::Exit.save_per_step_async());
        assert!(Durability::Exit.save_on_exit());
    }

    #[test]
    fn every_stride_skips_intermediate() {
        let d = Durability::Every {
            n: 3,
            mode: Box::new(Durability::Sync),
        };
        assert_eq!(d.decide(0, false), DurabilityDecision::Sync);
        assert_eq!(d.decide(1, false), DurabilityDecision::Skip);
        assert_eq!(d.decide(2, false), DurabilityDecision::Skip);
        assert_eq!(d.decide(3, false), DurabilityDecision::Sync);
        // Terminal always saves.
        assert_eq!(d.decide(7, true), DurabilityDecision::Sync);
    }

    #[test]
    fn custom_hook_is_invoked() {
        let d = Durability::Custom(Arc::new(|step: u64, terminal: bool| {
            if terminal || step.is_multiple_of(2) {
                DurabilityDecision::Sync
            } else {
                DurabilityDecision::Skip
            }
        }));
        assert_eq!(d.decide(0, false), DurabilityDecision::Sync);
        assert_eq!(d.decide(1, false), DurabilityDecision::Skip);
        assert_eq!(d.decide(99, true), DurabilityDecision::Sync);
    }
}
