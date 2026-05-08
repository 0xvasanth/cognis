//! Plugin registry with lifecycle and dependency resolution.
//!
//! Builds on [`super::AgentPlugin`] (single-shot consume-self installer)
//! with a richer [`LifecyclePlugin`] trait that supports:
//!
//! - **Named plugins** — each plugin reports a unique `name()`.
//! - **Declared dependencies** — `deps()` returns the names of plugins
//!   that must be active before this one. The registry topo-sorts the
//!   activation order and rejects cycles.
//! - **Lifecycle** — `register` / `activate` / `deactivate` / `unregister`,
//!   so plugins can clean up resources or be selectively disabled at
//!   runtime.
//!
//! Use `LifecyclePlugin` when you ship multiple cooperating plugins
//! that depend on each other; use the simpler [`super::AgentPlugin`]
//! / [`super::FnPlugin`] when you just want one inline mutator.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use cognis_core::{CognisError, Result};

use super::AgentBuilder;

/// A plugin that participates in the lifecycle managed by a
/// [`PluginRegistry`]. Implementations are `&self` so the registry can
/// activate/deactivate the same plugin multiple times.
pub trait LifecyclePlugin: Send + Sync {
    /// Stable, unique plugin name.
    fn name(&self) -> &str;

    /// Names of plugins that must be active before this one. The registry
    /// uses these to compute activation order.
    fn deps(&self) -> Vec<String> {
        Vec::new()
    }

    /// Mutate the builder. Called by [`PluginRegistry::activate_all`]
    /// in topo-sorted order. Return an error to abort activation
    /// (the registry rolls back to the pre-activation state).
    fn activate(&self, builder: AgentBuilder) -> Result<AgentBuilder>;

    /// Optional teardown — called on `deactivate`. Default is a no-op.
    fn deactivate(&self) -> Result<()> {
        Ok(())
    }
}

/// Closure-based [`LifecyclePlugin`] for inline registration.
pub struct ClosurePlugin {
    name: String,
    deps: Vec<String>,
    #[allow(clippy::type_complexity)]
    activate_fn: Box<dyn Fn(AgentBuilder) -> Result<AgentBuilder> + Send + Sync>,
}

impl ClosurePlugin {
    /// Build with a name + activation closure. Defaults to no deps.
    pub fn new<F>(name: impl Into<String>, activate: F) -> Self
    where
        F: Fn(AgentBuilder) -> Result<AgentBuilder> + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            deps: Vec::new(),
            activate_fn: Box::new(activate),
        }
    }

    /// Declare a dependency on another plugin. Builder-style.
    pub fn after(mut self, dep: impl Into<String>) -> Self {
        self.deps.push(dep.into());
        self
    }
}

impl LifecyclePlugin for ClosurePlugin {
    fn name(&self) -> &str {
        &self.name
    }
    fn deps(&self) -> Vec<String> {
        self.deps.clone()
    }
    fn activate(&self, builder: AgentBuilder) -> Result<AgentBuilder> {
        (self.activate_fn)(builder)
    }
}

/// Registry of [`LifecyclePlugin`]s. Cheap to clone (plugins live behind
/// `Arc`).
#[derive(Clone, Default)]
pub struct PluginRegistry {
    plugins: HashMap<String, Arc<dyn LifecyclePlugin>>,
    active: HashSet<String>,
}

impl PluginRegistry {
    /// Empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a plugin under its `name()`. Errors on duplicate names.
    pub fn register(&mut self, plugin: Arc<dyn LifecyclePlugin>) -> Result<()> {
        let name = plugin.name().to_string();
        if self.plugins.contains_key(&name) {
            return Err(CognisError::Configuration(format!(
                "PluginRegistry: duplicate plugin `{name}`"
            )));
        }
        self.plugins.insert(name, plugin);
        Ok(())
    }

    /// Unregister a plugin by name. If the plugin is currently active,
    /// it is deactivated first.
    pub fn unregister(&mut self, name: &str) -> Result<()> {
        if self.active.contains(name) {
            if let Some(p) = self.plugins.get(name) {
                p.deactivate()?;
            }
            self.active.remove(name);
        }
        self.plugins.remove(name);
        Ok(())
    }

    /// Names of every registered plugin (sorted).
    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.plugins.keys().cloned().collect();
        v.sort();
        v
    }

    /// Names of currently active plugins (sorted).
    pub fn active(&self) -> Vec<String> {
        let mut v: Vec<String> = self.active.iter().cloned().collect();
        v.sort();
        v
    }

    /// Activate every registered plugin in dependency order. Returns the
    /// mutated builder. Errors if any plugin's `activate` fails or if
    /// the dependency graph contains a cycle / unknown name. On error,
    /// already-activated plugins are deactivated to keep registry state
    /// consistent.
    pub fn activate_all(&mut self, builder: AgentBuilder) -> Result<AgentBuilder> {
        let order = self.topo_order()?;
        let mut current = builder;
        let mut newly_active: Vec<String> = Vec::new();
        for name in order {
            if self.active.contains(&name) {
                continue;
            }
            let plugin = self.plugins.get(&name).expect("topo only returns known");
            match plugin.activate(current) {
                Ok(b) => {
                    current = b;
                    self.active.insert(name.clone());
                    newly_active.push(name);
                }
                Err(e) => {
                    // Roll back the plugins we activated this call so the
                    // registry doesn't end up half-on.
                    for n in newly_active.iter().rev() {
                        if let Some(p) = self.plugins.get(n) {
                            let _ = p.deactivate();
                        }
                        self.active.remove(n);
                    }
                    return Err(e);
                }
            }
        }
        Ok(current)
    }

    /// Deactivate every active plugin in reverse activation order.
    pub fn deactivate_all(&mut self) -> Result<()> {
        // Reverse-topo for shutdown so dependents tear down before their deps.
        let mut order = self.topo_order()?;
        order.reverse();
        for name in order {
            if !self.active.remove(&name) {
                continue;
            }
            if let Some(p) = self.plugins.get(&name) {
                p.deactivate()?;
            }
        }
        Ok(())
    }

    /// Topo-sort of the registered plugins by `deps`. Errors on cycles
    /// or references to unknown plugins.
    fn topo_order(&self) -> Result<Vec<String>> {
        let mut indeg: HashMap<String, usize> =
            self.plugins.keys().map(|n| (n.clone(), 0)).collect();
        let mut rev: HashMap<String, Vec<String>> = HashMap::new();
        for (name, p) in &self.plugins {
            for d in p.deps() {
                if !self.plugins.contains_key(&d) {
                    return Err(CognisError::Configuration(format!(
                        "PluginRegistry: `{name}` depends on unknown plugin `{d}`"
                    )));
                }
                *indeg.get_mut(name).unwrap() += 1;
                rev.entry(d).or_default().push(name.clone());
            }
        }
        let mut ready: VecDeque<String> = indeg
            .iter()
            .filter_map(|(n, &k)| if k == 0 { Some(n.clone()) } else { None })
            .collect::<Vec<_>>()
            .into_iter()
            .collect();
        // Sort the initial frontier deterministically.
        let mut ready_vec: Vec<String> = ready.drain(..).collect();
        ready_vec.sort();
        let mut ready: VecDeque<String> = ready_vec.into_iter().collect();

        let mut out = Vec::with_capacity(self.plugins.len());
        while let Some(name) = ready.pop_front() {
            out.push(name.clone());
            if let Some(downstream) = rev.get(&name) {
                let mut newly_ready: Vec<String> = Vec::new();
                for d in downstream {
                    let n = indeg.get_mut(d).unwrap();
                    *n -= 1;
                    if *n == 0 {
                        newly_ready.push(d.clone());
                    }
                }
                newly_ready.sort();
                for n in newly_ready {
                    ready.push_back(n);
                }
            }
        }
        if out.len() != self.plugins.len() {
            return Err(CognisError::Configuration(
                "PluginRegistry: dependency cycle".into(),
            ));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::super::AgentBuilder;
    use super::*;

    fn closure(
        name: &str,
        deps: Vec<&str>,
        order: Arc<AtomicUsize>,
        log: Arc<std::sync::Mutex<Vec<(String, usize)>>>,
    ) -> Arc<dyn LifecyclePlugin> {
        let n = name.to_string();
        let mut p = ClosurePlugin::new(name, move |b| {
            let i = order.fetch_add(1, Ordering::Relaxed);
            log.lock().unwrap().push((n.clone(), i));
            Ok(b)
        });
        for d in deps {
            p = p.after(d);
        }
        Arc::new(p)
    }

    #[test]
    fn duplicate_name_rejected() {
        let mut reg = PluginRegistry::new();
        reg.register(Arc::new(ClosurePlugin::new("a", Ok))).unwrap();
        let err = reg
            .register(Arc::new(ClosurePlugin::new("a", Ok)))
            .unwrap_err();
        assert!(err.to_string().contains("duplicate"));
    }

    #[test]
    fn unknown_dep_rejected() {
        let mut reg = PluginRegistry::new();
        reg.register(Arc::new(ClosurePlugin::new("a", Ok).after("missing")))
            .unwrap();
        let err = match reg.activate_all(AgentBuilder::new()) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("unknown plugin"), "got: {err}");
    }

    #[test]
    fn cycle_rejected() {
        let mut reg = PluginRegistry::new();
        reg.register(Arc::new(ClosurePlugin::new("a", Ok).after("b")))
            .unwrap();
        reg.register(Arc::new(ClosurePlugin::new("b", Ok).after("a")))
            .unwrap();
        let err = match reg.activate_all(AgentBuilder::new()) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("cycle"), "got: {err}");
    }

    #[test]
    fn topo_orders_diamond_correctly() {
        // a → b → d
        // a → c → d
        let order = Arc::new(AtomicUsize::new(0));
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut reg = PluginRegistry::new();
        reg.register(closure("a", vec![], order.clone(), log.clone()))
            .unwrap();
        reg.register(closure("b", vec!["a"], order.clone(), log.clone()))
            .unwrap();
        reg.register(closure("c", vec!["a"], order.clone(), log.clone()))
            .unwrap();
        reg.register(closure("d", vec!["b", "c"], order.clone(), log.clone()))
            .unwrap();
        reg.activate_all(AgentBuilder::new()).unwrap();
        let log = log.lock().unwrap().clone();
        let pos = |n: &str| log.iter().find(|(name, _)| name == n).unwrap().1;
        assert!(pos("a") < pos("b"));
        assert!(pos("a") < pos("c"));
        assert!(pos("b") < pos("d"));
        assert!(pos("c") < pos("d"));
    }

    #[test]
    fn activate_rolls_back_on_error() {
        let activated = Arc::new(AtomicUsize::new(0));
        let act_clone = activated.clone();
        let act_clone2 = activated.clone();
        let mut reg = PluginRegistry::new();
        // First plugin succeeds (and increments the counter).
        reg.register(Arc::new(ClosurePlugin::new("ok", move |b| {
            act_clone.fetch_add(1, Ordering::Relaxed);
            Ok(b)
        })))
        .unwrap();
        // Second plugin (depends on first) fails.
        reg.register(Arc::new(
            ClosurePlugin::new("boom", move |_| {
                act_clone2.fetch_add(1, Ordering::Relaxed);
                Err(CognisError::Internal("nope".into()))
            })
            .after("ok"),
        ))
        .unwrap();

        let res = reg.activate_all(AgentBuilder::new());
        assert!(res.is_err());
        // Both plugins ran their activate (one succeeded, one failed)
        assert_eq!(activated.load(Ordering::Relaxed), 2);
        // But neither remains active after rollback.
        assert!(reg.active().is_empty());
    }

    #[test]
    fn unregister_deactivates_first() {
        let deactivated = Arc::new(AtomicUsize::new(0));
        struct Counted(Arc<AtomicUsize>);
        impl LifecyclePlugin for Counted {
            fn name(&self) -> &str {
                "counted"
            }
            fn activate(&self, b: AgentBuilder) -> Result<AgentBuilder> {
                Ok(b)
            }
            fn deactivate(&self) -> Result<()> {
                self.0.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
        }
        let mut reg = PluginRegistry::new();
        reg.register(Arc::new(Counted(deactivated.clone())))
            .unwrap();
        reg.activate_all(AgentBuilder::new()).unwrap();
        assert_eq!(reg.active(), vec!["counted".to_string()]);
        reg.unregister("counted").unwrap();
        assert_eq!(deactivated.load(Ordering::Relaxed), 1);
        assert!(reg.active().is_empty());
        assert!(reg.names().is_empty());
    }

    #[test]
    fn deactivate_all_walks_in_reverse_topo() {
        let order = Arc::new(AtomicUsize::new(0));
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        struct LoggingPlugin {
            name: String,
            deps: Vec<String>,
            order: Arc<AtomicUsize>,
            log: Arc<std::sync::Mutex<Vec<String>>>,
        }
        impl LifecyclePlugin for LoggingPlugin {
            fn name(&self) -> &str {
                &self.name
            }
            fn deps(&self) -> Vec<String> {
                self.deps.clone()
            }
            fn activate(&self, b: AgentBuilder) -> Result<AgentBuilder> {
                Ok(b)
            }
            fn deactivate(&self) -> Result<()> {
                let _ = self.order.fetch_add(1, Ordering::Relaxed);
                self.log.lock().unwrap().push(self.name.clone());
                Ok(())
            }
        }
        let mut reg = PluginRegistry::new();
        reg.register(Arc::new(LoggingPlugin {
            name: "a".into(),
            deps: vec![],
            order: order.clone(),
            log: log.clone(),
        }))
        .unwrap();
        reg.register(Arc::new(LoggingPlugin {
            name: "b".into(),
            deps: vec!["a".into()],
            order: order.clone(),
            log: log.clone(),
        }))
        .unwrap();
        reg.activate_all(AgentBuilder::new()).unwrap();
        reg.deactivate_all().unwrap();
        let log = log.lock().unwrap().clone();
        // Dependent (b) tears down before its dep (a).
        let pos = |n: &str| log.iter().position(|x| x == n).unwrap();
        assert!(pos("b") < pos("a"));
    }
}
