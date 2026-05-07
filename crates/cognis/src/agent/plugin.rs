//! Static-linked agent plugin trait.
//!
//! Rust-native: no `.so` loading, no dynamic registry. A plugin is a
//! Rust crate that exposes an `AgentPlugin` type whose `install` method
//! mutates an [`AgentBuilder`](super::AgentBuilder). Composes via
//! `.install_plugin(p)` in the builder.

use super::AgentBuilder;

/// Pluggable extension that mutates an `AgentBuilder` with a coherent
/// bundle of tools / middleware / config. Lets crates ship "plugin
/// presets" (e.g. a `web-tools` crate that registers HTTP, search,
/// scrape under one install call).
pub trait AgentPlugin {
    /// Mutate the builder. Typical body: register tools, push middleware,
    /// set system-prompt fragments.
    fn install(self, builder: AgentBuilder) -> AgentBuilder;
}

/// Closure-based plugin. Use when you don't want a named type.
pub struct FnPlugin<F: FnOnce(AgentBuilder) -> AgentBuilder>(pub F);

impl<F: FnOnce(AgentBuilder) -> AgentBuilder> AgentPlugin for FnPlugin<F> {
    fn install(self, builder: AgentBuilder) -> AgentBuilder {
        (self.0)(builder)
    }
}
