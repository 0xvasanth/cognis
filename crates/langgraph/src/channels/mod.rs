//! State channels for the LangGraph Pregel execution engine.
//!
//! Channels are the core communication primitive between nodes in a graph.
//! Each channel type defines a different strategy for how values from multiple
//! nodes are combined and made available to downstream consumers.
//!
//! # Channel Types
//!
//! - [`LastValue`] — Stores the most recent value; errors if more than one value
//!   is written per step.
//! - [`AnyValue`] — Stores the last value; assumes multiple values are equal.
//!   Clears on empty update.
//! - [`BinaryOperatorAggregate`] — Folds values using a configurable binary
//!   operator (add, extend, replace, or custom).
//! - [`Topic`] — Pub/sub channel that collects values into a list. Optionally
//!   accumulates across steps.
//! - [`EphemeralValue`] — Stores a value for one step only; cleared after consumption.
//! - [`NamedBarrierValue`] — Synchronization barrier that waits for all named
//!   participants before becoming available.
//! - [`UntrackedValue`] — Like `LastValue` but never included in checkpoints.
//! - [`Broadcast`](broadcast::BroadcastChannel) — Pub/sub channel with topic filtering and
//!   configurable delivery guarantees.
//! - [`reducers`] — State reducers (append, last-value, merge, binary-op) for composing
//!   channel updates.
//! - [`state_reducers`] — Schema-validated state reducers with field-level reducer
//!   specifications, type checking, and composable reduction strategies.
//!
//! # The `BaseChannel` Trait
//!
//! All channels implement the [`BaseChannel`] trait, which provides a uniform
//! interface for the Pregel engine to interact with channels regardless of
//! their internal strategy.

pub mod any_value;
pub mod base;
pub mod binop;
pub mod broadcast;
pub mod ephemeral_value;
pub mod last_value;
pub mod named_barrier;
pub mod reducers;
pub mod state;
pub mod state_reducers;
pub mod topic;
pub mod untracked;

pub use any_value::AnyValue;
pub use base::BaseChannel;
pub use binop::{BinOp, BinaryOperatorAggregate};
pub use broadcast::{
    BroadcastChannel, BroadcastConfig, BroadcastFilter, BroadcastMessage, DeliveryGuarantee,
};
pub use ephemeral_value::EphemeralValue;
pub use last_value::{LastValue, LastValueAfterFinish};
pub use named_barrier::{NamedBarrierValue, NamedBarrierValueAfterFinish};
pub use reducers::{
    reduce_state, AppendReducer, BinaryOpReducer, FieldSpec, LastValueReducer, MergeReducer,
    Reducer, StateSchema, StateSchemaBuilder,
};
pub use topic::{
    DeadLetterQueue, Topic, TopicBus, TopicBusStats, TopicChannel, TopicFilter, TopicMessage,
    TopicRouter,
};
pub use state::{
    BinaryChannel, ChannelManager, ChannelProtocol, ChannelReducer, ChannelValue, LastValueChannel,
    StateChannel,
};
pub use untracked::UntrackedValue;
