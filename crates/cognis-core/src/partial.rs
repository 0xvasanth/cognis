//! The [`Partial`] trait linking a type to its all-optional streaming mirror.
//!
//! Reach for it with typed partial-object streaming: derive it via
//! `#[derive(cognis_macros::Partial)]` and stream `T::Partial` snapshots.

/// Links a fully-specified type to its all-fields-optional streaming mirror.
///
/// While a structured response streams in, required fields are not yet
/// present, so a snapshot cannot deserialize into `Self`. The mirror type
/// wraps every field in `Option` so each snapshot decodes. Implemented by
/// `#[derive(cognis_macros::Partial)]`.
pub trait Partial {
    /// The mirror type with every field `Option<...>`.
    type Partial: serde::de::DeserializeOwned + Send + 'static;
}
