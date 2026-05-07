//! Per-node counters + simple timing aggregation as Observers.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

use uuid::Uuid;

use cognis2_core::{Event, Observer};

/// Per-node execution counts and error counts.
#[derive(Debug, Default, Clone)]
pub struct GraphMetrics {
    /// Times each node has finished successfully.
    pub node_executions: HashMap<String, u64>,
    /// Times each node errored.
    pub errors: HashMap<String, u64>,
    /// Total supersteps observed (`OnNodeEnd` events).
    pub total_steps: u64,
}

/// Observer that maintains a [`GraphMetrics`] under a `Mutex`.
pub struct MetricsObserver {
    inner: Mutex<GraphMetrics>,
}

impl Default for MetricsObserver {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsObserver {
    /// Empty observer.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(GraphMetrics::default()),
        }
    }

    /// Snapshot the current metrics.
    pub fn snapshot(&self) -> GraphMetrics {
        self.inner.lock().map(|g| g.clone()).unwrap_or_default()
    }
}

impl Observer for MetricsObserver {
    fn on_event(&self, event: &Event) {
        let mut g = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        match event {
            Event::OnNodeEnd { node, .. } => {
                *g.node_executions.entry(node.clone()).or_insert(0) += 1;
                g.total_steps += 1;
            }
            Event::OnError { error, .. } => {
                *g.errors.entry(error.clone()).or_insert(0) += 1;
            }
            _ => {}
        }
    }
}

/// Per-node timing aggregator. Pairs `OnNodeStart` / `OnNodeEnd` events
/// keyed by `(run_id, step, node)` to compute durations.
pub struct ProfilingObserver {
    pending: Mutex<HashMap<(Uuid, u64, String), Instant>>,
    totals: Mutex<HashMap<String, NodeTiming>>,
}

/// One node's timing aggregate.
#[derive(Debug, Default, Clone)]
pub struct NodeTiming {
    /// Number of finished invocations seen.
    pub count: u64,
    /// Total elapsed nanoseconds across invocations.
    pub total_ns: u128,
    /// Slowest single invocation.
    pub max_ns: u128,
    /// Fastest single invocation.
    pub min_ns: u128,
}

impl NodeTiming {
    /// Mean nanoseconds per invocation. Returns `0` for zero count.
    pub fn mean_ns(&self) -> u128 {
        if self.count == 0 {
            0
        } else {
            self.total_ns / self.count as u128
        }
    }
}

impl Default for ProfilingObserver {
    fn default() -> Self {
        Self::new()
    }
}

impl ProfilingObserver {
    /// Empty profiler.
    pub fn new() -> Self {
        Self {
            pending: Mutex::new(HashMap::new()),
            totals: Mutex::new(HashMap::new()),
        }
    }

    /// Snapshot of per-node timings.
    pub fn snapshot(&self) -> HashMap<String, NodeTiming> {
        self.totals.lock().map(|m| m.clone()).unwrap_or_default()
    }
}

impl Observer for ProfilingObserver {
    fn on_event(&self, event: &Event) {
        match event {
            Event::OnNodeStart { node, step, run_id } => {
                if let Ok(mut p) = self.pending.lock() {
                    p.insert((*run_id, *step, node.clone()), Instant::now());
                }
            }
            Event::OnNodeEnd {
                node, step, run_id, ..
            } => {
                let mut p = match self.pending.lock() {
                    Ok(p) => p,
                    Err(_) => return,
                };
                let key = (*run_id, *step, node.clone());
                let started = match p.remove(&key) {
                    Some(t) => t,
                    None => return,
                };
                let elapsed_ns = started.elapsed().as_nanos();
                drop(p);
                let mut t = match self.totals.lock() {
                    Ok(t) => t,
                    Err(_) => return,
                };
                let e = t.entry(node.clone()).or_insert_with(|| NodeTiming {
                    min_ns: u128::MAX,
                    ..Default::default()
                });
                e.count += 1;
                e.total_ns += elapsed_ns;
                e.max_ns = e.max_ns.max(elapsed_ns);
                e.min_ns = e.min_ns.min(elapsed_ns);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev_node_end(node: &str) -> Event {
        Event::OnNodeEnd {
            node: node.into(),
            step: 0,
            output: serde_json::Value::Null,
            run_id: Uuid::nil(),
        }
    }

    #[test]
    fn metrics_count_executions() {
        let m = MetricsObserver::new();
        m.on_event(&ev_node_end("a"));
        m.on_event(&ev_node_end("a"));
        m.on_event(&ev_node_end("b"));
        m.on_event(&Event::OnError {
            error: "boom".into(),
            run_id: Uuid::nil(),
        });
        let snap = m.snapshot();
        assert_eq!(snap.node_executions["a"], 2);
        assert_eq!(snap.node_executions["b"], 1);
        assert_eq!(snap.total_steps, 3);
        assert_eq!(snap.errors["boom"], 1);
    }

    #[test]
    fn profiler_pairs_start_and_end() {
        let p = ProfilingObserver::new();
        let id = Uuid::nil();
        p.on_event(&Event::OnNodeStart {
            node: "n".into(),
            step: 0,
            run_id: id,
        });
        std::thread::sleep(std::time::Duration::from_millis(2));
        p.on_event(&Event::OnNodeEnd {
            node: "n".into(),
            step: 0,
            output: serde_json::Value::Null,
            run_id: id,
        });
        let snap = p.snapshot();
        let t = snap.get("n").unwrap();
        assert_eq!(t.count, 1);
        assert!(t.total_ns > 0);
    }
}
