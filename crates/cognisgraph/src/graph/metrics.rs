//! Graph execution metrics collection, profiling, and reporting.
//!
//! This module provides a pluggable metrics system for tracking node and edge
//! statistics during graph execution. It supports:
//!
//! - Per-node timing (total, avg, min, max, percentiles)
//! - Edge traversal counts and hot/cold path detection
//! - Execution timeline profiling
//! - Bottleneck detection and summary reports
//! - JSON export via serde
//! - Aggregation across multiple runs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};

// ---------------------------------------------------------------------------
// NodeMetrics
// ---------------------------------------------------------------------------

/// Per-node execution statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetrics {
    /// Name of the node.
    pub node_name: String,
    /// Number of times this node was executed.
    pub execution_count: u64,
    /// Total wall-clock time spent executing this node.
    pub total_duration: Duration,
    /// Minimum single-execution duration.
    pub min_duration: Duration,
    /// Maximum single-execution duration.
    pub max_duration: Duration,
    /// Number of executions that ended in error.
    pub error_count: u64,
    /// Timestamp of the last execution (as millis since UNIX epoch).
    pub last_executed: Option<u64>,
    /// Raw durations kept for percentile computation.
    #[serde(skip)]
    pub durations: Vec<Duration>,
}

impl NodeMetrics {
    /// Create a new, empty `NodeMetrics` for the given node name.
    pub fn new(node_name: impl Into<String>) -> Self {
        Self {
            node_name: node_name.into(),
            execution_count: 0,
            total_duration: Duration::ZERO,
            min_duration: Duration::MAX,
            max_duration: Duration::ZERO,
            error_count: 0,
            last_executed: None,
            durations: Vec::new(),
        }
    }

    /// Record a single execution.
    pub fn record(&mut self, duration: Duration, is_error: bool) {
        self.execution_count += 1;
        self.total_duration += duration;
        if duration < self.min_duration {
            self.min_duration = duration;
        }
        if duration > self.max_duration {
            self.max_duration = duration;
        }
        if is_error {
            self.error_count += 1;
        }
        self.last_executed = Some(
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_millis() as u64,
        );
        self.durations.push(duration);
    }

    /// Average execution duration. Returns `Duration::ZERO` when no executions.
    pub fn avg_duration(&self) -> Duration {
        if self.execution_count == 0 {
            Duration::ZERO
        } else {
            self.total_duration / self.execution_count as u32
        }
    }

    /// Compute a percentile (0–100) over recorded durations.
    pub fn percentile(&self, p: f64) -> Duration {
        if self.durations.is_empty() {
            return Duration::ZERO;
        }
        let mut sorted: Vec<Duration> = self.durations.clone();
        sorted.sort();
        let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round().max(0.0) as usize;
        let idx = idx.min(sorted.len() - 1);
        sorted[idx]
    }

    /// Shorthand for p50.
    pub fn p50(&self) -> Duration {
        self.percentile(50.0)
    }

    /// Shorthand for p95.
    pub fn p95(&self) -> Duration {
        self.percentile(95.0)
    }

    /// Shorthand for p99.
    pub fn p99(&self) -> Duration {
        self.percentile(99.0)
    }

    /// Error rate as a fraction in [0, 1].
    pub fn error_rate(&self) -> f64 {
        if self.execution_count == 0 {
            0.0
        } else {
            self.error_count as f64 / self.execution_count as f64
        }
    }
}

// ---------------------------------------------------------------------------
// EdgeMetrics
// ---------------------------------------------------------------------------

/// Per-edge traversal statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeMetrics {
    /// Source node.
    pub from: String,
    /// Target node.
    pub to: String,
    /// Number of times this edge was traversed.
    pub traversal_count: u64,
}

impl EdgeMetrics {
    pub fn new(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            traversal_count: 0,
        }
    }

    pub fn record_traversal(&mut self) {
        self.traversal_count += 1;
    }
}

// ---------------------------------------------------------------------------
// ExecutionProfile
// ---------------------------------------------------------------------------

/// A single entry in the execution timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileEntry {
    pub node_name: String,
    /// Offset from the start of the graph run.
    pub offset: Duration,
    pub duration: Duration,
    pub is_error: bool,
}

/// Timeline of node executions within a single graph run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionProfile {
    pub entries: Vec<ProfileEntry>,
    #[serde(skip)]
    pub(crate) run_start: Option<Instant>,
}

impl ExecutionProfile {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            run_start: None,
        }
    }

    /// Mark the start of a graph run.
    pub fn start(&mut self) {
        self.run_start = Some(Instant::now());
    }

    /// Record a node execution entry.
    pub fn record(&mut self, node_name: impl Into<String>, duration: Duration, is_error: bool) {
        let offset = self
            .run_start
            .map(|s| s.elapsed())
            .unwrap_or(Duration::ZERO);
        self.entries.push(ProfileEntry {
            node_name: node_name.into(),
            offset,
            duration,
            is_error,
        });
    }

    /// Total wall-clock time of all recorded entries.
    pub fn total_duration(&self) -> Duration {
        self.entries.iter().map(|e| e.duration).sum()
    }

    /// Return node names in execution order.
    pub fn execution_order(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.node_name.as_str()).collect()
    }
}

// ---------------------------------------------------------------------------
// GraphMetrics
// ---------------------------------------------------------------------------

/// Aggregated metrics for a single graph execution run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetrics {
    pub run_id: String,
    pub node_metrics: HashMap<String, NodeMetrics>,
    pub edge_metrics: HashMap<String, EdgeMetrics>,
    pub profile: ExecutionProfile,
    pub total_duration: Duration,
    pub total_nodes_executed: u64,
    pub total_errors: u64,
}

impl GraphMetrics {
    pub fn new(run_id: impl Into<String>) -> Self {
        Self {
            run_id: run_id.into(),
            node_metrics: HashMap::new(),
            edge_metrics: HashMap::new(),
            profile: ExecutionProfile::new(),
            total_duration: Duration::ZERO,
            total_nodes_executed: 0,
            total_errors: 0,
        }
    }

    /// Begin profiling.
    pub fn start(&mut self) {
        self.profile.start();
    }

    /// Record a node execution.
    pub fn record_node(&mut self, node_name: &str, duration: Duration, is_error: bool) {
        let nm = self
            .node_metrics
            .entry(node_name.to_string())
            .or_insert_with(|| NodeMetrics::new(node_name));
        nm.record(duration, is_error);
        self.profile.record(node_name, duration, is_error);
        self.total_nodes_executed += 1;
        if is_error {
            self.total_errors += 1;
        }
    }

    /// Record an edge traversal.
    pub fn record_edge(&mut self, from: &str, to: &str) {
        let key = format!("{}->{}", from, to);
        let em = self
            .edge_metrics
            .entry(key)
            .or_insert_with(|| EdgeMetrics::new(from, to));
        em.record_traversal();
    }

    /// Finalize the run, setting total_duration.
    pub fn finish(&mut self) {
        self.total_duration = self
            .profile
            .run_start
            .map(|s| s.elapsed())
            .unwrap_or_else(|| self.profile.total_duration());
    }

    /// The node with the highest total duration — the primary bottleneck.
    pub fn bottleneck(&self) -> Option<&NodeMetrics> {
        self.node_metrics
            .values()
            .max_by_key(|nm| nm.total_duration)
    }

    /// The edge with the highest traversal count.
    pub fn hot_edge(&self) -> Option<&EdgeMetrics> {
        self.edge_metrics
            .values()
            .max_by_key(|em| em.traversal_count)
    }

    /// The edge with the lowest traversal count (non-zero).
    pub fn cold_edge(&self) -> Option<&EdgeMetrics> {
        self.edge_metrics
            .values()
            .filter(|em| em.traversal_count > 0)
            .min_by_key(|em| em.traversal_count)
    }

    /// Nodes sorted by total duration descending.
    pub fn nodes_by_duration(&self) -> Vec<&NodeMetrics> {
        let mut v: Vec<_> = self.node_metrics.values().collect();
        v.sort_by(|a, b| b.total_duration.cmp(&a.total_duration));
        v
    }

    /// Edges sorted by traversal count descending.
    pub fn edges_by_traversal(&self) -> Vec<&EdgeMetrics> {
        let mut v: Vec<_> = self.edge_metrics.values().collect();
        v.sort_by(|a, b| b.traversal_count.cmp(&a.traversal_count));
        v
    }
}

// ---------------------------------------------------------------------------
// MetricsCollector trait
// ---------------------------------------------------------------------------

/// Pluggable sink for metrics events.
pub trait MetricsCollector: Send + Sync {
    /// Called when a node execution completes.
    fn on_node_executed(&mut self, node_name: &str, duration: Duration, is_error: bool);
    /// Called when an edge is traversed.
    fn on_edge_traversed(&mut self, from: &str, to: &str);
    /// Called when a graph run starts.
    fn on_run_start(&mut self, run_id: &str);
    /// Called when a graph run finishes.
    fn on_run_finish(&mut self, run_id: &str);
    /// Retrieve collected metrics snapshot.
    fn snapshot(&self) -> GraphMetrics;
}

// ---------------------------------------------------------------------------
// InMemoryMetricsCollector
// ---------------------------------------------------------------------------

/// In-memory implementation of [`MetricsCollector`].
#[derive(Debug)]
pub struct InMemoryMetricsCollector {
    current: GraphMetrics,
    history: Vec<GraphMetrics>,
}

impl InMemoryMetricsCollector {
    pub fn new() -> Self {
        Self {
            current: GraphMetrics::new(""),
            history: Vec::new(),
        }
    }

    /// Get all historically finished runs.
    pub fn history(&self) -> &[GraphMetrics] {
        &self.history
    }

    /// Number of completed runs.
    pub fn completed_runs(&self) -> usize {
        self.history.len()
    }
}

impl Default for InMemoryMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsCollector for InMemoryMetricsCollector {
    fn on_node_executed(&mut self, node_name: &str, duration: Duration, is_error: bool) {
        self.current.record_node(node_name, duration, is_error);
    }

    fn on_edge_traversed(&mut self, from: &str, to: &str) {
        self.current.record_edge(from, to);
    }

    fn on_run_start(&mut self, run_id: &str) {
        self.current = GraphMetrics::new(run_id);
        self.current.start();
    }

    fn on_run_finish(&mut self, run_id: &str) {
        self.current.finish();
        if self.current.run_id == run_id {
            self.history.push(self.current.clone());
        }
    }

    fn snapshot(&self) -> GraphMetrics {
        self.current.clone()
    }
}

// ---------------------------------------------------------------------------
// MetricsAggregator
// ---------------------------------------------------------------------------

/// Combines metrics from multiple graph runs into a single summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsAggregator {
    pub total_runs: u64,
    pub total_duration: Duration,
    pub total_nodes_executed: u64,
    pub total_errors: u64,
    pub node_metrics: HashMap<String, NodeMetrics>,
    pub edge_metrics: HashMap<String, EdgeMetrics>,
}

impl MetricsAggregator {
    pub fn new() -> Self {
        Self {
            total_runs: 0,
            total_duration: Duration::ZERO,
            total_nodes_executed: 0,
            total_errors: 0,
            node_metrics: HashMap::new(),
            edge_metrics: HashMap::new(),
        }
    }

    /// Merge a single `GraphMetrics` into the aggregate.
    pub fn merge(&mut self, gm: &GraphMetrics) {
        self.total_runs += 1;
        self.total_duration += gm.total_duration;
        self.total_nodes_executed += gm.total_nodes_executed;
        self.total_errors += gm.total_errors;

        for (name, nm) in &gm.node_metrics {
            let entry = self
                .node_metrics
                .entry(name.clone())
                .or_insert_with(|| NodeMetrics::new(name));
            entry.execution_count += nm.execution_count;
            entry.total_duration += nm.total_duration;
            entry.error_count += nm.error_count;
            if nm.min_duration < entry.min_duration {
                entry.min_duration = nm.min_duration;
            }
            if nm.max_duration > entry.max_duration {
                entry.max_duration = nm.max_duration;
            }
            if nm.last_executed > entry.last_executed {
                entry.last_executed = nm.last_executed;
            }
            entry.durations.extend_from_slice(&nm.durations);
        }

        for (key, em) in &gm.edge_metrics {
            let entry = self
                .edge_metrics
                .entry(key.clone())
                .or_insert_with(|| EdgeMetrics::new(&em.from, &em.to));
            entry.traversal_count += em.traversal_count;
        }
    }

    /// Merge multiple runs.
    pub fn merge_all(&mut self, runs: &[GraphMetrics]) {
        for run in runs {
            self.merge(run);
        }
    }

    /// Average run duration.
    pub fn avg_run_duration(&self) -> Duration {
        if self.total_runs == 0 {
            Duration::ZERO
        } else {
            self.total_duration / self.total_runs as u32
        }
    }

    /// Overall error rate.
    pub fn error_rate(&self) -> f64 {
        if self.total_nodes_executed == 0 {
            0.0
        } else {
            self.total_errors as f64 / self.total_nodes_executed as f64
        }
    }

    /// Top N nodes by total duration.
    pub fn top_nodes_by_duration(&self, n: usize) -> Vec<&NodeMetrics> {
        let mut v: Vec<_> = self.node_metrics.values().collect();
        v.sort_by(|a, b| b.total_duration.cmp(&a.total_duration));
        v.truncate(n);
        v
    }

    /// Top N edges by traversal count.
    pub fn top_edges_by_traversal(&self, n: usize) -> Vec<&EdgeMetrics> {
        let mut v: Vec<_> = self.edge_metrics.values().collect();
        v.sort_by(|a, b| b.traversal_count.cmp(&a.traversal_count));
        v.truncate(n);
        v
    }
}

impl Default for MetricsAggregator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MetricsReport
// ---------------------------------------------------------------------------

/// Human-readable summary report generated from metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsReport {
    pub total_runs: u64,
    pub total_nodes_executed: u64,
    pub total_errors: u64,
    pub total_duration: Duration,
    pub avg_run_duration: Duration,
    pub error_rate: f64,
    pub bottleneck_node: Option<String>,
    pub bottleneck_duration: Option<Duration>,
    pub hot_path: Vec<String>,
    pub cold_path: Vec<String>,
    pub node_summaries: Vec<NodeSummary>,
}

/// Summary for a single node inside a report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSummary {
    pub name: String,
    pub execution_count: u64,
    pub total_duration: Duration,
    pub avg_duration: Duration,
    pub p50: Duration,
    pub p95: Duration,
    pub p99: Duration,
    pub error_count: u64,
    pub error_rate: f64,
}

impl MetricsReport {
    /// Build a report from a `MetricsAggregator`.
    pub fn from_aggregator(agg: &MetricsAggregator) -> Self {
        let bottleneck = agg.top_nodes_by_duration(1).first().cloned();
        let hot_edges = agg.top_edges_by_traversal(3);
        let cold_edges = {
            let mut v: Vec<_> = agg
                .edge_metrics
                .values()
                .filter(|em| em.traversal_count > 0)
                .collect();
            v.sort_by_key(|em| em.traversal_count);
            v.truncate(3);
            v
        };

        let node_summaries: Vec<NodeSummary> = {
            let mut nodes: Vec<_> = agg.node_metrics.values().collect();
            nodes.sort_by(|a, b| b.total_duration.cmp(&a.total_duration));
            nodes
                .iter()
                .map(|nm| NodeSummary {
                    name: nm.node_name.clone(),
                    execution_count: nm.execution_count,
                    total_duration: nm.total_duration,
                    avg_duration: nm.avg_duration(),
                    p50: nm.p50(),
                    p95: nm.p95(),
                    p99: nm.p99(),
                    error_count: nm.error_count,
                    error_rate: nm.error_rate(),
                })
                .collect()
        };

        Self {
            total_runs: agg.total_runs,
            total_nodes_executed: agg.total_nodes_executed,
            total_errors: agg.total_errors,
            total_duration: agg.total_duration,
            avg_run_duration: agg.avg_run_duration(),
            error_rate: agg.error_rate(),
            bottleneck_node: bottleneck.map(|n| n.node_name.clone()),
            bottleneck_duration: bottleneck.map(|n| n.total_duration),
            hot_path: hot_edges
                .iter()
                .map(|e| format!("{}->{}", e.from, e.to))
                .collect(),
            cold_path: cold_edges
                .iter()
                .map(|e| format!("{}->{}", e.from, e.to))
                .collect(),
            node_summaries,
        }
    }

    /// Build a report directly from a single `GraphMetrics`.
    pub fn from_graph_metrics(gm: &GraphMetrics) -> Self {
        let mut agg = MetricsAggregator::new();
        agg.merge(gm);
        Self::from_aggregator(&agg)
    }
}

// ---------------------------------------------------------------------------
// MetricsExporter
// ---------------------------------------------------------------------------

/// Serializes metrics to JSON.
pub struct MetricsExporter;

impl MetricsExporter {
    /// Export `GraphMetrics` as a JSON string.
    pub fn to_json(gm: &GraphMetrics) -> serde_json::Result<String> {
        serde_json::to_string_pretty(gm)
    }

    /// Export a `MetricsReport` as a JSON string.
    pub fn report_to_json(report: &MetricsReport) -> serde_json::Result<String> {
        serde_json::to_string_pretty(report)
    }

    /// Export a `MetricsAggregator` as a JSON string.
    pub fn aggregator_to_json(agg: &MetricsAggregator) -> serde_json::Result<String> {
        serde_json::to_string_pretty(agg)
    }

    /// Export `NodeMetrics` as JSON.
    pub fn node_to_json(nm: &NodeMetrics) -> serde_json::Result<String> {
        serde_json::to_string_pretty(nm)
    }

    /// Export `EdgeMetrics` as JSON.
    pub fn edge_to_json(em: &EdgeMetrics) -> serde_json::Result<String> {
        serde_json::to_string_pretty(em)
    }

    /// Export `ExecutionProfile` as JSON.
    pub fn profile_to_json(profile: &ExecutionProfile) -> serde_json::Result<String> {
        serde_json::to_string_pretty(profile)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- NodeMetrics ---------------------------------------------------------

    #[test]
    fn node_metrics_new_defaults() {
        let nm = NodeMetrics::new("test");
        assert_eq!(nm.node_name, "test");
        assert_eq!(nm.execution_count, 0);
        assert_eq!(nm.total_duration, Duration::ZERO);
        assert_eq!(nm.min_duration, Duration::MAX);
        assert_eq!(nm.max_duration, Duration::ZERO);
        assert_eq!(nm.error_count, 0);
        assert!(nm.last_executed.is_none());
        assert!(nm.durations.is_empty());
    }

    #[test]
    fn node_metrics_record_success() {
        let mut nm = NodeMetrics::new("n");
        nm.record(Duration::from_millis(100), false);
        assert_eq!(nm.execution_count, 1);
        assert_eq!(nm.total_duration, Duration::from_millis(100));
        assert_eq!(nm.min_duration, Duration::from_millis(100));
        assert_eq!(nm.max_duration, Duration::from_millis(100));
        assert_eq!(nm.error_count, 0);
        assert!(nm.last_executed.is_some());
    }

    #[test]
    fn node_metrics_record_error() {
        let mut nm = NodeMetrics::new("n");
        nm.record(Duration::from_millis(50), true);
        assert_eq!(nm.error_count, 1);
        assert_eq!(nm.execution_count, 1);
    }

    #[test]
    fn node_metrics_min_max() {
        let mut nm = NodeMetrics::new("n");
        nm.record(Duration::from_millis(200), false);
        nm.record(Duration::from_millis(50), false);
        nm.record(Duration::from_millis(150), false);
        assert_eq!(nm.min_duration, Duration::from_millis(50));
        assert_eq!(nm.max_duration, Duration::from_millis(200));
    }

    #[test]
    fn node_metrics_avg_duration() {
        let mut nm = NodeMetrics::new("n");
        nm.record(Duration::from_millis(100), false);
        nm.record(Duration::from_millis(200), false);
        assert_eq!(nm.avg_duration(), Duration::from_millis(150));
    }

    #[test]
    fn node_metrics_avg_duration_empty() {
        let nm = NodeMetrics::new("n");
        assert_eq!(nm.avg_duration(), Duration::ZERO);
    }

    #[test]
    fn node_metrics_percentile_empty() {
        let nm = NodeMetrics::new("n");
        assert_eq!(nm.percentile(50.0), Duration::ZERO);
    }

    #[test]
    fn node_metrics_percentile_single() {
        let mut nm = NodeMetrics::new("n");
        nm.record(Duration::from_millis(42), false);
        assert_eq!(nm.percentile(50.0), Duration::from_millis(42));
        assert_eq!(nm.percentile(99.0), Duration::from_millis(42));
    }

    #[test]
    fn node_metrics_p50() {
        let mut nm = NodeMetrics::new("n");
        for v in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] {
            nm.record(Duration::from_millis(v), false);
        }
        let p = nm.p50();
        // median of sorted [10..100] is around 50-60
        assert!(p >= Duration::from_millis(50) && p <= Duration::from_millis(60));
    }

    #[test]
    fn node_metrics_p95() {
        let mut nm = NodeMetrics::new("n");
        for v in 1..=100 {
            nm.record(Duration::from_millis(v), false);
        }
        let p = nm.p95();
        assert!(p >= Duration::from_millis(95));
    }

    #[test]
    fn node_metrics_p99() {
        let mut nm = NodeMetrics::new("n");
        for v in 1..=100 {
            nm.record(Duration::from_millis(v), false);
        }
        let p = nm.p99();
        assert!(p >= Duration::from_millis(98));
    }

    #[test]
    fn node_metrics_error_rate_zero() {
        let nm = NodeMetrics::new("n");
        assert_eq!(nm.error_rate(), 0.0);
    }

    #[test]
    fn node_metrics_error_rate_half() {
        let mut nm = NodeMetrics::new("n");
        nm.record(Duration::from_millis(10), false);
        nm.record(Duration::from_millis(10), true);
        assert!((nm.error_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn node_metrics_error_rate_all() {
        let mut nm = NodeMetrics::new("n");
        nm.record(Duration::from_millis(10), true);
        nm.record(Duration::from_millis(10), true);
        assert!((nm.error_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn node_metrics_total_duration_accumulates() {
        let mut nm = NodeMetrics::new("n");
        nm.record(Duration::from_millis(10), false);
        nm.record(Duration::from_millis(20), false);
        nm.record(Duration::from_millis(30), false);
        assert_eq!(nm.total_duration, Duration::from_millis(60));
    }

    #[test]
    fn node_metrics_last_executed_updates() {
        let mut nm = NodeMetrics::new("n");
        nm.record(Duration::from_millis(1), false);
        let first = nm.last_executed;
        nm.record(Duration::from_millis(1), false);
        let second = nm.last_executed;
        assert!(second >= first);
    }

    // -- EdgeMetrics ---------------------------------------------------------

    #[test]
    fn edge_metrics_new() {
        let em = EdgeMetrics::new("a", "b");
        assert_eq!(em.from, "a");
        assert_eq!(em.to, "b");
        assert_eq!(em.traversal_count, 0);
    }

    #[test]
    fn edge_metrics_record() {
        let mut em = EdgeMetrics::new("a", "b");
        em.record_traversal();
        em.record_traversal();
        assert_eq!(em.traversal_count, 2);
    }

    // -- ExecutionProfile ----------------------------------------------------

    #[test]
    fn profile_new_is_empty() {
        let p = ExecutionProfile::new();
        assert!(p.entries.is_empty());
        assert!(p.run_start.is_none());
    }

    #[test]
    fn profile_record_entries() {
        let mut p = ExecutionProfile::new();
        p.start();
        p.record("a", Duration::from_millis(10), false);
        p.record("b", Duration::from_millis(20), true);
        assert_eq!(p.entries.len(), 2);
        assert_eq!(p.entries[0].node_name, "a");
        assert!(!p.entries[0].is_error);
        assert!(p.entries[1].is_error);
    }

    #[test]
    fn profile_total_duration() {
        let mut p = ExecutionProfile::new();
        p.record("a", Duration::from_millis(10), false);
        p.record("b", Duration::from_millis(20), false);
        assert_eq!(p.total_duration(), Duration::from_millis(30));
    }

    #[test]
    fn profile_execution_order() {
        let mut p = ExecutionProfile::new();
        p.record("x", Duration::from_millis(1), false);
        p.record("y", Duration::from_millis(1), false);
        p.record("z", Duration::from_millis(1), false);
        assert_eq!(p.execution_order(), vec!["x", "y", "z"]);
    }

    #[test]
    fn profile_default() {
        let p = ExecutionProfile::default();
        assert!(p.entries.is_empty());
    }

    // -- GraphMetrics --------------------------------------------------------

    #[test]
    fn graph_metrics_new() {
        let gm = GraphMetrics::new("run-1");
        assert_eq!(gm.run_id, "run-1");
        assert!(gm.node_metrics.is_empty());
        assert!(gm.edge_metrics.is_empty());
        assert_eq!(gm.total_nodes_executed, 0);
        assert_eq!(gm.total_errors, 0);
    }

    #[test]
    fn graph_metrics_record_node() {
        let mut gm = GraphMetrics::new("r");
        gm.record_node("a", Duration::from_millis(10), false);
        gm.record_node("a", Duration::from_millis(20), true);
        assert_eq!(gm.total_nodes_executed, 2);
        assert_eq!(gm.total_errors, 1);
        let nm = gm.node_metrics.get("a").unwrap();
        assert_eq!(nm.execution_count, 2);
        assert_eq!(nm.error_count, 1);
    }

    #[test]
    fn graph_metrics_record_edge() {
        let mut gm = GraphMetrics::new("r");
        gm.record_edge("a", "b");
        gm.record_edge("a", "b");
        gm.record_edge("b", "c");
        let ab = gm.edge_metrics.get("a->b").unwrap();
        assert_eq!(ab.traversal_count, 2);
        let bc = gm.edge_metrics.get("b->c").unwrap();
        assert_eq!(bc.traversal_count, 1);
    }

    #[test]
    fn graph_metrics_bottleneck() {
        let mut gm = GraphMetrics::new("r");
        gm.record_node("fast", Duration::from_millis(10), false);
        gm.record_node("slow", Duration::from_millis(200), false);
        let b = gm.bottleneck().unwrap();
        assert_eq!(b.node_name, "slow");
    }

    #[test]
    fn graph_metrics_bottleneck_empty() {
        let gm = GraphMetrics::new("r");
        assert!(gm.bottleneck().is_none());
    }

    #[test]
    fn graph_metrics_hot_edge() {
        let mut gm = GraphMetrics::new("r");
        gm.record_edge("a", "b");
        gm.record_edge("a", "b");
        gm.record_edge("b", "c");
        let h = gm.hot_edge().unwrap();
        assert_eq!(h.from, "a");
        assert_eq!(h.to, "b");
    }

    #[test]
    fn graph_metrics_cold_edge() {
        let mut gm = GraphMetrics::new("r");
        gm.record_edge("a", "b");
        gm.record_edge("a", "b");
        gm.record_edge("b", "c");
        let c = gm.cold_edge().unwrap();
        assert_eq!(c.from, "b");
        assert_eq!(c.to, "c");
    }

    #[test]
    fn graph_metrics_hot_edge_empty() {
        let gm = GraphMetrics::new("r");
        assert!(gm.hot_edge().is_none());
    }

    #[test]
    fn graph_metrics_cold_edge_empty() {
        let gm = GraphMetrics::new("r");
        assert!(gm.cold_edge().is_none());
    }

    #[test]
    fn graph_metrics_nodes_by_duration() {
        let mut gm = GraphMetrics::new("r");
        gm.record_node("a", Duration::from_millis(10), false);
        gm.record_node("b", Duration::from_millis(50), false);
        gm.record_node("c", Duration::from_millis(30), false);
        let sorted = gm.nodes_by_duration();
        assert_eq!(sorted[0].node_name, "b");
        assert_eq!(sorted[1].node_name, "c");
        assert_eq!(sorted[2].node_name, "a");
    }

    #[test]
    fn graph_metrics_edges_by_traversal() {
        let mut gm = GraphMetrics::new("r");
        gm.record_edge("a", "b");
        gm.record_edge("b", "c");
        gm.record_edge("b", "c");
        gm.record_edge("b", "c");
        let sorted = gm.edges_by_traversal();
        assert_eq!(sorted[0].from, "b");
        assert_eq!(sorted[0].to, "c");
    }

    #[test]
    fn graph_metrics_start_and_finish() {
        let mut gm = GraphMetrics::new("r");
        gm.start();
        gm.record_node("a", Duration::from_millis(10), false);
        gm.finish();
        // total_duration should be >= 0 after finish
        assert!(gm.total_duration >= Duration::ZERO);
    }

    #[test]
    fn graph_metrics_profile_populated() {
        let mut gm = GraphMetrics::new("r");
        gm.start();
        gm.record_node("x", Duration::from_millis(5), false);
        gm.record_node("y", Duration::from_millis(15), false);
        assert_eq!(gm.profile.entries.len(), 2);
    }

    // -- InMemoryMetricsCollector --------------------------------------------

    #[test]
    fn collector_new() {
        let c = InMemoryMetricsCollector::new();
        assert_eq!(c.completed_runs(), 0);
    }

    #[test]
    fn collector_default() {
        let c = InMemoryMetricsCollector::default();
        assert_eq!(c.completed_runs(), 0);
    }

    #[test]
    fn collector_run_lifecycle() {
        let mut c = InMemoryMetricsCollector::new();
        c.on_run_start("run-1");
        c.on_node_executed("a", Duration::from_millis(10), false);
        c.on_edge_traversed("a", "b");
        c.on_run_finish("run-1");
        assert_eq!(c.completed_runs(), 1);
    }

    #[test]
    fn collector_snapshot() {
        let mut c = InMemoryMetricsCollector::new();
        c.on_run_start("r");
        c.on_node_executed("n", Duration::from_millis(5), false);
        let snap = c.snapshot();
        assert_eq!(snap.run_id, "r");
        assert_eq!(snap.total_nodes_executed, 1);
    }

    #[test]
    fn collector_multiple_runs() {
        let mut c = InMemoryMetricsCollector::new();
        c.on_run_start("r1");
        c.on_node_executed("a", Duration::from_millis(10), false);
        c.on_run_finish("r1");
        c.on_run_start("r2");
        c.on_node_executed("b", Duration::from_millis(20), false);
        c.on_run_finish("r2");
        assert_eq!(c.completed_runs(), 2);
        assert_eq!(c.history()[0].run_id, "r1");
        assert_eq!(c.history()[1].run_id, "r2");
    }

    #[test]
    fn collector_history_contains_all_node_data() {
        let mut c = InMemoryMetricsCollector::new();
        c.on_run_start("r");
        c.on_node_executed("x", Duration::from_millis(100), true);
        c.on_run_finish("r");
        let h = &c.history()[0];
        assert_eq!(h.total_errors, 1);
        let nm = h.node_metrics.get("x").unwrap();
        assert_eq!(nm.error_count, 1);
    }

    #[test]
    fn collector_mismatched_finish_ignored() {
        let mut c = InMemoryMetricsCollector::new();
        c.on_run_start("r1");
        c.on_run_finish("wrong-id");
        // should not record because IDs don't match
        assert_eq!(c.completed_runs(), 0);
    }

    // -- MetricsAggregator ---------------------------------------------------

    #[test]
    fn aggregator_new() {
        let a = MetricsAggregator::new();
        assert_eq!(a.total_runs, 0);
    }

    #[test]
    fn aggregator_default() {
        let a = MetricsAggregator::default();
        assert_eq!(a.total_runs, 0);
    }

    #[test]
    fn aggregator_merge_single() {
        let mut gm = GraphMetrics::new("r");
        gm.record_node("a", Duration::from_millis(100), false);
        gm.record_edge("a", "b");
        gm.total_duration = Duration::from_millis(100);

        let mut agg = MetricsAggregator::new();
        agg.merge(&gm);
        assert_eq!(agg.total_runs, 1);
        assert_eq!(agg.total_nodes_executed, 1);
        assert_eq!(agg.total_duration, Duration::from_millis(100));
    }

    #[test]
    fn aggregator_merge_multiple() {
        let mut g1 = GraphMetrics::new("r1");
        g1.record_node("a", Duration::from_millis(10), false);
        g1.total_duration = Duration::from_millis(10);

        let mut g2 = GraphMetrics::new("r2");
        g2.record_node("a", Duration::from_millis(30), false);
        g2.total_duration = Duration::from_millis(30);

        let mut agg = MetricsAggregator::new();
        agg.merge_all(&[g1, g2]);
        assert_eq!(agg.total_runs, 2);
        let nm = agg.node_metrics.get("a").unwrap();
        assert_eq!(nm.execution_count, 2);
        assert_eq!(nm.total_duration, Duration::from_millis(40));
    }

    #[test]
    fn aggregator_avg_run_duration() {
        let mut agg = MetricsAggregator::new();
        let mut g1 = GraphMetrics::new("r1");
        g1.total_duration = Duration::from_millis(100);
        let mut g2 = GraphMetrics::new("r2");
        g2.total_duration = Duration::from_millis(200);
        agg.merge(&g1);
        agg.merge(&g2);
        assert_eq!(agg.avg_run_duration(), Duration::from_millis(150));
    }

    #[test]
    fn aggregator_avg_run_duration_empty() {
        let agg = MetricsAggregator::new();
        assert_eq!(agg.avg_run_duration(), Duration::ZERO);
    }

    #[test]
    fn aggregator_error_rate() {
        let mut gm = GraphMetrics::new("r");
        gm.record_node("a", Duration::from_millis(10), false);
        gm.record_node("b", Duration::from_millis(10), true);
        let mut agg = MetricsAggregator::new();
        agg.merge(&gm);
        assert!((agg.error_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn aggregator_error_rate_empty() {
        let agg = MetricsAggregator::new();
        assert_eq!(agg.error_rate(), 0.0);
    }

    #[test]
    fn aggregator_top_nodes() {
        let mut gm = GraphMetrics::new("r");
        gm.record_node("fast", Duration::from_millis(5), false);
        gm.record_node("slow", Duration::from_millis(500), false);
        gm.record_node("mid", Duration::from_millis(50), false);
        let mut agg = MetricsAggregator::new();
        agg.merge(&gm);
        let top = agg.top_nodes_by_duration(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].node_name, "slow");
        assert_eq!(top[1].node_name, "mid");
    }

    #[test]
    fn aggregator_top_edges() {
        let mut gm = GraphMetrics::new("r");
        gm.record_edge("a", "b");
        gm.record_edge("a", "b");
        gm.record_edge("c", "d");
        let mut agg = MetricsAggregator::new();
        agg.merge(&gm);
        let top = agg.top_edges_by_traversal(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].from, "a");
    }

    #[test]
    fn aggregator_min_max_across_runs() {
        let mut g1 = GraphMetrics::new("r1");
        g1.record_node("n", Duration::from_millis(100), false);
        let mut g2 = GraphMetrics::new("r2");
        g2.record_node("n", Duration::from_millis(50), false);
        let mut agg = MetricsAggregator::new();
        agg.merge(&g1);
        agg.merge(&g2);
        let nm = agg.node_metrics.get("n").unwrap();
        assert_eq!(nm.min_duration, Duration::from_millis(50));
        assert_eq!(nm.max_duration, Duration::from_millis(100));
    }

    #[test]
    fn aggregator_durations_combined() {
        let mut g1 = GraphMetrics::new("r1");
        g1.record_node("n", Duration::from_millis(10), false);
        let mut g2 = GraphMetrics::new("r2");
        g2.record_node("n", Duration::from_millis(20), false);
        let mut agg = MetricsAggregator::new();
        agg.merge_all(&[g1, g2]);
        let nm = agg.node_metrics.get("n").unwrap();
        assert_eq!(nm.durations.len(), 2);
    }

    // -- MetricsReport -------------------------------------------------------

    #[test]
    fn report_from_aggregator() {
        let mut gm = GraphMetrics::new("r");
        gm.record_node("a", Duration::from_millis(100), false);
        gm.record_node("b", Duration::from_millis(200), true);
        gm.record_edge("a", "b");
        gm.total_duration = Duration::from_millis(300);
        let mut agg = MetricsAggregator::new();
        agg.merge(&gm);
        let report = MetricsReport::from_aggregator(&agg);
        assert_eq!(report.total_runs, 1);
        assert_eq!(report.total_nodes_executed, 2);
        assert_eq!(report.total_errors, 1);
        assert_eq!(report.bottleneck_node.as_deref(), Some("b"));
        assert!(!report.node_summaries.is_empty());
    }

    #[test]
    fn report_from_graph_metrics() {
        let mut gm = GraphMetrics::new("r");
        gm.record_node("x", Duration::from_millis(50), false);
        gm.total_duration = Duration::from_millis(50);
        let report = MetricsReport::from_graph_metrics(&gm);
        assert_eq!(report.total_runs, 1);
        assert_eq!(report.bottleneck_node.as_deref(), Some("x"));
    }

    #[test]
    fn report_hot_path() {
        let mut gm = GraphMetrics::new("r");
        gm.record_edge("a", "b");
        gm.record_edge("a", "b");
        gm.record_edge("b", "c");
        let report = MetricsReport::from_graph_metrics(&gm);
        assert!(!report.hot_path.is_empty());
        assert_eq!(report.hot_path[0], "a->b");
    }

    #[test]
    fn report_cold_path() {
        let mut gm = GraphMetrics::new("r");
        gm.record_edge("a", "b");
        gm.record_edge("a", "b");
        gm.record_edge("b", "c");
        let report = MetricsReport::from_graph_metrics(&gm);
        assert!(!report.cold_path.is_empty());
        assert_eq!(report.cold_path[0], "b->c");
    }

    #[test]
    fn report_node_summaries_sorted() {
        let mut gm = GraphMetrics::new("r");
        gm.record_node("fast", Duration::from_millis(10), false);
        gm.record_node("slow", Duration::from_millis(100), false);
        let report = MetricsReport::from_graph_metrics(&gm);
        assert_eq!(report.node_summaries[0].name, "slow");
        assert_eq!(report.node_summaries[1].name, "fast");
    }

    #[test]
    fn report_error_rate() {
        let mut gm = GraphMetrics::new("r");
        gm.record_node("a", Duration::from_millis(10), true);
        gm.record_node("b", Duration::from_millis(10), false);
        gm.record_node("c", Duration::from_millis(10), true);
        let report = MetricsReport::from_graph_metrics(&gm);
        assert!((report.error_rate - 2.0 / 3.0).abs() < 0.01);
    }

    // -- MetricsExporter -----------------------------------------------------

    #[test]
    fn export_graph_metrics_json() {
        let gm = GraphMetrics::new("r");
        let json = MetricsExporter::to_json(&gm).unwrap();
        assert!(json.contains("\"run_id\""));
    }

    #[test]
    fn export_report_json() {
        let gm = GraphMetrics::new("r");
        let report = MetricsReport::from_graph_metrics(&gm);
        let json = MetricsExporter::report_to_json(&report).unwrap();
        assert!(json.contains("\"total_runs\""));
    }

    #[test]
    fn export_aggregator_json() {
        let agg = MetricsAggregator::new();
        let json = MetricsExporter::aggregator_to_json(&agg).unwrap();
        assert!(json.contains("\"total_runs\""));
    }

    #[test]
    fn export_node_json() {
        let nm = NodeMetrics::new("test");
        let json = MetricsExporter::node_to_json(&nm).unwrap();
        assert!(json.contains("\"node_name\""));
    }

    #[test]
    fn export_edge_json() {
        let em = EdgeMetrics::new("a", "b");
        let json = MetricsExporter::edge_to_json(&em).unwrap();
        assert!(json.contains("\"from\""));
    }

    #[test]
    fn export_profile_json() {
        let p = ExecutionProfile::new();
        let json = MetricsExporter::profile_to_json(&p).unwrap();
        assert!(json.contains("\"entries\""));
    }

    #[test]
    fn export_roundtrip_graph_metrics() {
        let mut gm = GraphMetrics::new("roundtrip");
        gm.record_node("a", Duration::from_millis(42), false);
        gm.record_edge("a", "b");
        gm.total_duration = Duration::from_millis(42);
        let json = MetricsExporter::to_json(&gm).unwrap();
        let parsed: GraphMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.run_id, "roundtrip");
        assert_eq!(parsed.total_nodes_executed, 1);
    }

    #[test]
    fn export_roundtrip_report() {
        let mut gm = GraphMetrics::new("r");
        gm.record_node("n", Duration::from_millis(10), false);
        gm.total_duration = Duration::from_millis(10);
        let report = MetricsReport::from_graph_metrics(&gm);
        let json = MetricsExporter::report_to_json(&report).unwrap();
        let parsed: MetricsReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.total_runs, 1);
    }

    // -- Percentile edge cases -----------------------------------------------

    #[test]
    fn percentile_p0() {
        let mut nm = NodeMetrics::new("n");
        for v in [10, 20, 30] {
            nm.record(Duration::from_millis(v), false);
        }
        assert_eq!(nm.percentile(0.0), Duration::from_millis(10));
    }

    #[test]
    fn percentile_p100() {
        let mut nm = NodeMetrics::new("n");
        for v in [10, 20, 30] {
            nm.record(Duration::from_millis(v), false);
        }
        assert_eq!(nm.percentile(100.0), Duration::from_millis(30));
    }

    // -- Integration-style tests ---------------------------------------------

    #[test]
    fn full_pipeline_single_run() {
        let mut collector = InMemoryMetricsCollector::new();
        collector.on_run_start("integration-1");
        collector.on_node_executed("start", Duration::from_millis(5), false);
        collector.on_edge_traversed("start", "process");
        collector.on_node_executed("process", Duration::from_millis(50), false);
        collector.on_edge_traversed("process", "end");
        collector.on_node_executed("end", Duration::from_millis(2), false);
        collector.on_run_finish("integration-1");

        let history = collector.history();
        assert_eq!(history.len(), 1);
        let gm = &history[0];
        assert_eq!(gm.total_nodes_executed, 3);
        assert_eq!(gm.total_errors, 0);

        let report = MetricsReport::from_graph_metrics(gm);
        assert_eq!(report.bottleneck_node.as_deref(), Some("process"));
        assert_eq!(report.node_summaries.len(), 3);
    }

    #[test]
    fn full_pipeline_multi_run_aggregation() {
        let mut collector = InMemoryMetricsCollector::new();

        for i in 0..5 {
            let id = format!("run-{}", i);
            collector.on_run_start(&id);
            collector.on_node_executed("a", Duration::from_millis(10 * (i as u64 + 1)), false);
            collector.on_edge_traversed("a", "b");
            collector.on_node_executed("b", Duration::from_millis(5), i % 2 == 0);
            collector.on_run_finish(&id);
        }

        assert_eq!(collector.completed_runs(), 5);

        let mut agg = MetricsAggregator::new();
        agg.merge_all(collector.history());
        assert_eq!(agg.total_runs, 5);
        assert_eq!(agg.total_nodes_executed, 10);
        // Errors on runs 0, 2, 4 => 3 errors
        assert_eq!(agg.total_errors, 3);
        let report = MetricsReport::from_aggregator(&agg);
        assert_eq!(report.total_runs, 5);
        assert!(report.error_rate > 0.0);
    }
}
