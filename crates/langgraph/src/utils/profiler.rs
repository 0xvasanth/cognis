//! Graph execution profiler.
//!
//! Provides [`GraphProfiler`] for tracking execution time, node call counts,
//! edge traversals, and bottleneck detection during graph execution.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// ProfilerOptions
// ---------------------------------------------------------------------------

/// Configuration options for the graph profiler.
#[derive(Debug, Clone)]
pub struct ProfilerOptions {
    /// Whether profiling is enabled. When `false`, all recording methods
    /// become no-ops.
    pub enabled: bool,
    /// Whether to record edge traversals.
    pub track_edges: bool,
    /// Whether to record errors.
    pub track_errors: bool,
    /// Sampling rate in the range `[0.0, 1.0]`. A value of `1.0` records
    /// every event; `0.5` records roughly half.
    pub sample_rate: f64,
}

impl Default for ProfilerOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            track_edges: true,
            track_errors: true,
            sample_rate: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// NodeProfile
// ---------------------------------------------------------------------------

/// Aggregated profile data for a single node.
#[derive(Debug, Clone)]
pub struct NodeProfile {
    /// Node name.
    pub name: String,
    /// Number of times the node was called.
    pub call_count: usize,
    /// Sum of all call durations.
    pub total_duration: Duration,
    /// Shortest call duration.
    pub min_duration: Duration,
    /// Longest call duration.
    pub max_duration: Duration,
    /// Average call duration (`total_duration / call_count`).
    pub avg_duration: Duration,
    /// Number of errors recorded for this node.
    pub error_count: usize,
    /// Percentage of total graph execution time spent in this node.
    pub percentage_of_total: f64,
}

// ---------------------------------------------------------------------------
// EdgeTraversal
// ---------------------------------------------------------------------------

/// A recorded edge traversal between two nodes.
#[derive(Debug, Clone)]
pub struct EdgeTraversal {
    /// Source node.
    pub from: String,
    /// Destination node.
    pub to: String,
    /// Number of times this edge was traversed.
    pub count: usize,
    /// Timestamp of the most recent traversal.
    pub timestamp: Instant,
}

// ---------------------------------------------------------------------------
// ProfileReport
// ---------------------------------------------------------------------------

/// Complete profiling report for a graph execution.
#[derive(Debug, Clone)]
pub struct ProfileReport {
    /// Total wall-clock duration of the profiled execution.
    pub total_duration: Duration,
    /// Per-node profile data.
    pub node_profiles: HashMap<String, NodeProfile>,
    /// Recorded edge traversals.
    pub edge_traversals: Vec<EdgeTraversal>,
    /// The node that consumed the most total time.
    pub bottleneck_node: Option<String>,
    /// The critical path — longest chain of nodes by cumulative duration.
    pub hotpath: Vec<String>,
    /// Total number of node calls across all nodes.
    pub total_node_calls: usize,
    /// Total number of errors across all nodes.
    pub error_count: usize,
}

impl ProfileReport {
    /// Return a human-readable summary string.
    pub fn summary(&self) -> String {
        format_report(self)
    }
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

/// Timing information for a single in-progress node call.
#[derive(Debug, Clone)]
struct NodeTiming {
    start: Instant,
}

/// Completed duration record for a node call.
#[derive(Debug, Clone)]
struct NodeRecord {
    duration: Duration,
}

/// Internal edge key for aggregation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct EdgeKey {
    from: String,
    to: String,
}

/// Internal edge data.
#[derive(Debug, Clone)]
struct EdgeData {
    count: usize,
    last_timestamp: Instant,
}

/// Error record for a node.
#[derive(Debug, Clone)]
struct ErrorRecord {
    _node: String,
    _error: String,
}

/// All mutable profiler state behind a lock.
#[derive(Debug)]
struct ProfilerState {
    /// When the profiler was created or last reset.
    start_time: Instant,
    /// Currently in-progress node calls.
    active: HashMap<String, Vec<NodeTiming>>,
    /// Completed call records per node.
    records: HashMap<String, Vec<NodeRecord>>,
    /// Aggregated edge data.
    edges: HashMap<EdgeKey, EdgeData>,
    /// Ordered list of edge keys for stable output.
    edge_order: Vec<EdgeKey>,
    /// Error records.
    errors: Vec<ErrorRecord>,
    /// Simple counter used for sampling.
    sample_counter: u64,
}

impl ProfilerState {
    fn new() -> Self {
        Self {
            start_time: Instant::now(),
            active: HashMap::new(),
            records: HashMap::new(),
            edges: HashMap::new(),
            edge_order: Vec::new(),
            errors: Vec::new(),
            sample_counter: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// GraphProfiler
// ---------------------------------------------------------------------------

/// Thread-safe graph execution profiler.
///
/// Records node execution times, edge traversals, and errors, then produces
/// a [`ProfileReport`] with bottleneck and hot-path analysis.
///
/// # Example
///
/// ```
/// use langgraph::utils::profiler::{GraphProfiler, ProfilerOptions};
///
/// let profiler = GraphProfiler::new(ProfilerOptions::default());
/// profiler.start_node("agent");
/// // ... node work ...
/// profiler.end_node("agent");
/// let report = profiler.get_report();
/// println!("{}", report.summary());
/// ```
#[derive(Debug, Clone)]
pub struct GraphProfiler {
    options: ProfilerOptions,
    state: Arc<Mutex<ProfilerState>>,
}

impl GraphProfiler {
    /// Create a new profiler with the given options.
    pub fn new(options: ProfilerOptions) -> Self {
        Self {
            options,
            state: Arc::new(Mutex::new(ProfilerState::new())),
        }
    }

    /// Whether this event should be sampled (recorded).
    fn should_sample(&self) -> bool {
        if !self.options.enabled {
            return false;
        }
        if (self.options.sample_rate - 1.0).abs() < f64::EPSILON {
            return true;
        }
        if self.options.sample_rate <= 0.0 {
            return false;
        }
        let mut state = self.state.lock().unwrap();
        state.sample_counter += 1;
        let threshold = (1.0 / self.options.sample_rate) as u64;
        state.sample_counter % threshold == 0
    }

    /// Record the start of a node execution.
    pub fn start_node(&self, node_name: &str) {
        if !self.should_sample() {
            return;
        }
        let mut state = self.state.lock().unwrap();
        state
            .active
            .entry(node_name.to_string())
            .or_default()
            .push(NodeTiming {
                start: Instant::now(),
            });
    }

    /// Record the end of a node execution. Pairs with the most recent
    /// [`start_node`](Self::start_node) call for the same name.
    pub fn end_node(&self, node_name: &str) {
        let mut state = self.state.lock().unwrap();
        if let Some(timings) = state.active.get_mut(node_name) {
            if let Some(timing) = timings.pop() {
                let duration = timing.start.elapsed();
                state
                    .records
                    .entry(node_name.to_string())
                    .or_default()
                    .push(NodeRecord { duration });
            }
        }
    }

    /// Record an edge traversal from one node to another.
    pub fn record_edge(&self, from: &str, to: &str) {
        if !self.options.enabled || !self.options.track_edges {
            return;
        }
        let mut state = self.state.lock().unwrap();
        let key = EdgeKey {
            from: from.to_string(),
            to: to.to_string(),
        };
        if !state.edges.contains_key(&key) {
            state.edge_order.push(key.clone());
            state.edges.insert(
                key,
                EdgeData {
                    count: 1,
                    last_timestamp: Instant::now(),
                },
            );
        } else {
            let entry = state.edges.get_mut(&key).unwrap();
            entry.count += 1;
            entry.last_timestamp = Instant::now();
        }
    }

    /// Record an error that occurred in a node.
    pub fn record_error(&self, node_name: &str, error: &str) {
        if !self.options.enabled || !self.options.track_errors {
            return;
        }
        let mut state = self.state.lock().unwrap();
        state.errors.push(ErrorRecord {
            _node: node_name.to_string(),
            _error: error.to_string(),
        });
    }

    /// Create an RAII scope guard that automatically records start/end for
    /// the named node.
    pub fn scope(&self, node_name: &str) -> ProfileScope {
        self.start_node(node_name);
        ProfileScope {
            profiler: self.clone(),
            node_name: node_name.to_string(),
        }
    }

    /// Generate a profiling report from the data collected so far.
    pub fn get_report(&self) -> ProfileReport {
        let state = self.state.lock().unwrap();
        let total_duration = state.start_time.elapsed();

        // Build node profiles.
        let mut node_profiles = HashMap::new();
        let mut total_node_calls: usize = 0;

        // Compute total node time first for percentage calculation.
        let total_node_time: Duration = state
            .records
            .values()
            .flat_map(|recs| recs.iter())
            .map(|r| r.duration)
            .sum();

        for (name, records) in &state.records {
            if records.is_empty() {
                continue;
            }
            let call_count = records.len();
            total_node_calls += call_count;
            let total_dur: Duration = records.iter().map(|r| r.duration).sum();
            let min_dur = records.iter().map(|r| r.duration).min().unwrap();
            let max_dur = records.iter().map(|r| r.duration).max().unwrap();
            let avg_dur = total_dur / call_count as u32;

            let pct = if total_node_time.as_nanos() > 0 {
                (total_dur.as_nanos() as f64 / total_node_time.as_nanos() as f64) * 100.0
            } else {
                0.0
            };

            // Count errors for this node.
            let error_count = state
                .errors
                .iter()
                .filter(|e| e._node == *name)
                .count();

            node_profiles.insert(
                name.clone(),
                NodeProfile {
                    name: name.clone(),
                    call_count,
                    total_duration: total_dur,
                    min_duration: min_dur,
                    max_duration: max_dur,
                    avg_duration: avg_dur,
                    error_count,
                    percentage_of_total: pct,
                },
            );
        }

        // Bottleneck: node with highest total_duration.
        let bottleneck_node = node_profiles
            .values()
            .max_by_key(|np| np.total_duration)
            .map(|np| np.name.clone());

        // Edge traversals.
        let edge_traversals: Vec<EdgeTraversal> = state
            .edge_order
            .iter()
            .filter_map(|key| {
                state.edges.get(key).map(|data| EdgeTraversal {
                    from: key.from.clone(),
                    to: key.to.clone(),
                    count: data.count,
                    timestamp: data.last_timestamp,
                })
            })
            .collect();

        // Hot path: build an adjacency list from edges and find the longest
        // path by summing average node durations. We use a simple DFS since
        // graph execution DAGs are typically small.
        let hotpath = compute_hotpath(&node_profiles, &edge_traversals);

        let error_count = state.errors.len();

        ProfileReport {
            total_duration,
            node_profiles,
            edge_traversals,
            bottleneck_node,
            hotpath,
            total_node_calls,
            error_count,
        }
    }

    /// Clear all collected profiling data.
    pub fn reset(&self) {
        let mut state = self.state.lock().unwrap();
        *state = ProfilerState::new();
    }
}

// ---------------------------------------------------------------------------
// Hot path computation
// ---------------------------------------------------------------------------

/// Find the critical path through the graph — the chain of nodes with the
/// largest cumulative total duration.
fn compute_hotpath(
    profiles: &HashMap<String, NodeProfile>,
    edges: &[EdgeTraversal],
) -> Vec<String> {
    if profiles.is_empty() {
        return Vec::new();
    }

    // Build adjacency list.
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut has_incoming: std::collections::HashSet<&str> = std::collections::HashSet::new();

    for edge in edges {
        adj.entry(edge.from.as_str())
            .or_default()
            .push(edge.to.as_str());
        has_incoming.insert(edge.to.as_str());
    }

    // Find roots (nodes with no incoming edges that appear in profiles).
    let roots: Vec<&str> = profiles
        .keys()
        .filter(|k| !has_incoming.contains(k.as_str()))
        .map(|k| k.as_str())
        .collect();

    if roots.is_empty() && !profiles.is_empty() {
        // No edge info — fall back to single bottleneck node.
        return profiles
            .values()
            .max_by_key(|np| np.total_duration)
            .map(|np| vec![np.name.clone()])
            .unwrap_or_default();
    }

    // DFS to find longest path by total_duration.
    fn dfs<'a>(
        node: &'a str,
        adj: &HashMap<&'a str, Vec<&'a str>>,
        profiles: &HashMap<String, NodeProfile>,
        visited: &mut std::collections::HashSet<&'a str>,
    ) -> (Duration, Vec<String>) {
        let self_dur = profiles
            .get(node)
            .map(|p| p.total_duration)
            .unwrap_or_default();
        visited.insert(node);

        let mut best_dur = Duration::ZERO;
        let mut best_path: Vec<String> = Vec::new();

        if let Some(neighbors) = adj.get(node) {
            for &next in neighbors {
                if visited.contains(next) {
                    continue;
                }
                let (dur, path) = dfs(next, adj, profiles, visited);
                if dur > best_dur {
                    best_dur = dur;
                    best_path = path;
                }
            }
        }

        visited.remove(node);

        let mut result = vec![node.to_string()];
        result.extend(best_path);
        (self_dur + best_dur, result)
    }

    let mut best_dur = Duration::ZERO;
    let mut best_path = Vec::new();

    let mut visited = std::collections::HashSet::new();
    for root in &roots {
        let (dur, path) = dfs(root, &adj, profiles, &mut visited);
        if dur > best_dur {
            best_dur = dur;
            best_path = path;
        }
    }

    best_path
}

// ---------------------------------------------------------------------------
// ProfileScope (RAII guard)
// ---------------------------------------------------------------------------

/// RAII guard that records node start on creation and node end on drop.
///
/// Obtained via [`GraphProfiler::scope`].
#[derive(Debug)]
pub struct ProfileScope {
    profiler: GraphProfiler,
    node_name: String,
}

impl Drop for ProfileScope {
    fn drop(&mut self) {
        self.profiler.end_node(&self.node_name);
    }
}

// ---------------------------------------------------------------------------
// format_report
// ---------------------------------------------------------------------------

/// Format a [`ProfileReport`] as a human-readable table.
pub fn format_report(report: &ProfileReport) -> String {
    let mut out = String::new();

    // Collect and sort nodes by total duration descending.
    let mut nodes: Vec<&NodeProfile> = report.node_profiles.values().collect();
    nodes.sort_by(|a, b| b.total_duration.cmp(&a.total_duration));

    // Determine column widths.
    let node_width = nodes
        .iter()
        .map(|n| n.name.len())
        .max()
        .unwrap_or(4)
        .max(4);
    let calls_width = 5;
    let total_width = 9;
    let avg_width = 9;

    let inner_width = node_width + calls_width + total_width + avg_width + 9; // separators

    // Top border.
    out.push_str(&format!("\u{250c}{}\u{2510}\n", "\u{2500}".repeat(inner_width)));

    // Title.
    let title = "Graph Execution Profile";
    let pad = (inner_width.saturating_sub(title.len())) / 2;
    out.push_str(&format!(
        "\u{2502}{}{}{}\u{2502}\n",
        " ".repeat(pad),
        title,
        " ".repeat(inner_width - pad - title.len())
    ));

    // Header separator.
    out.push_str(&format!(
        "\u{251c}{}\u{252c}{}\u{252c}{}\u{252c}{}\u{2524}\n",
        "\u{2500}".repeat(node_width + 2),
        "\u{2500}".repeat(calls_width + 2),
        "\u{2500}".repeat(total_width + 2),
        "\u{2500}".repeat(avg_width + 2),
    ));

    // Column headers.
    out.push_str(&format!(
        "\u{2502} {:<nw$} \u{2502} {:>cw$} \u{2502} {:>tw$} \u{2502} {:>aw$} \u{2502}\n",
        "Node",
        "Calls",
        "Total",
        "Avg",
        nw = node_width,
        cw = calls_width,
        tw = total_width,
        aw = avg_width,
    ));

    // Header/data separator.
    out.push_str(&format!(
        "\u{251c}{}\u{253c}{}\u{253c}{}\u{253c}{}\u{2524}\n",
        "\u{2500}".repeat(node_width + 2),
        "\u{2500}".repeat(calls_width + 2),
        "\u{2500}".repeat(total_width + 2),
        "\u{2500}".repeat(avg_width + 2),
    ));

    // Data rows.
    for node in &nodes {
        out.push_str(&format!(
            "\u{2502} {:<nw$} \u{2502} {:>cw$} \u{2502} {:>tw$} \u{2502} {:>aw$} \u{2502}\n",
            node.name,
            node.call_count,
            format_duration(node.total_duration),
            format_duration(node.avg_duration),
            nw = node_width,
            cw = calls_width,
            tw = total_width,
            aw = avg_width,
        ));
    }

    // Footer separator.
    out.push_str(&format!(
        "\u{251c}{}\u{2534}{}\u{2534}{}\u{2534}{}\u{2524}\n",
        "\u{2500}".repeat(node_width + 2),
        "\u{2500}".repeat(calls_width + 2),
        "\u{2500}".repeat(total_width + 2),
        "\u{2500}".repeat(avg_width + 2),
    ));

    // Bottleneck line.
    if let Some(ref bn) = report.bottleneck_node {
        let pct = report
            .node_profiles
            .get(bn)
            .map(|p| p.percentage_of_total)
            .unwrap_or(0.0);
        let line = format!("Bottleneck: {} ({:.1}%)", bn, pct);
        out.push_str(&format!(
            "\u{2502} {:<w$}\u{2502}\n",
            line,
            w = inner_width - 1
        ));
    }

    // Summary line.
    let summary = format!(
        "Total: {}, {} calls, {} errors",
        format_duration(report.total_duration),
        report.total_node_calls,
        report.error_count,
    );
    out.push_str(&format!(
        "\u{2502} {:<w$}\u{2502}\n",
        summary,
        w = inner_width - 1
    ));

    // Bottom border.
    out.push_str(&format!("\u{2514}{}\u{2518}\n", "\u{2500}".repeat(inner_width)));

    out
}

/// Format a [`Duration`] as a compact human-readable string.
fn format_duration(d: Duration) -> String {
    let micros = d.as_micros();
    if micros < 1_000 {
        format!("{}us", micros)
    } else if micros < 1_000_000 {
        format!("{:.1}ms", micros as f64 / 1_000.0)
    } else {
        format!("{:.2}s", micros as f64 / 1_000_000.0)
    }
}

impl fmt::Display for ProfileReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    // 1. Start/end node recording
    #[test]
    fn test_start_end_node() {
        let profiler = GraphProfiler::new(ProfilerOptions::default());
        profiler.start_node("agent");
        thread::sleep(Duration::from_millis(10));
        profiler.end_node("agent");

        let report = profiler.get_report();
        assert!(report.node_profiles.contains_key("agent"));
        assert_eq!(report.node_profiles["agent"].call_count, 1);
        assert!(report.node_profiles["agent"].total_duration >= Duration::from_millis(5));
    }

    // 2. Multiple calls to same node
    #[test]
    fn test_multiple_calls_same_node() {
        let profiler = GraphProfiler::new(ProfilerOptions::default());
        for _ in 0..3 {
            profiler.start_node("tools");
            thread::sleep(Duration::from_millis(5));
            profiler.end_node("tools");
        }

        let report = profiler.get_report();
        assert_eq!(report.node_profiles["tools"].call_count, 3);
        assert_eq!(report.total_node_calls, 3);
    }

    // 3. NodeProfile statistics (min/max/avg)
    #[test]
    fn test_node_profile_statistics() {
        let profiler = GraphProfiler::new(ProfilerOptions::default());

        // First call: short
        profiler.start_node("compute");
        thread::sleep(Duration::from_millis(10));
        profiler.end_node("compute");

        // Second call: longer
        profiler.start_node("compute");
        thread::sleep(Duration::from_millis(30));
        profiler.end_node("compute");

        let report = profiler.get_report();
        let np = &report.node_profiles["compute"];
        assert_eq!(np.call_count, 2);
        assert!(np.min_duration < np.max_duration);
        // avg should be between min and max
        assert!(np.avg_duration >= np.min_duration);
        assert!(np.avg_duration <= np.max_duration);
    }

    // 4. Edge traversal tracking
    #[test]
    fn test_edge_traversal() {
        let profiler = GraphProfiler::new(ProfilerOptions::default());
        profiler.record_edge("agent", "tools");
        profiler.record_edge("tools", "agent");
        profiler.record_edge("agent", "tools");

        let report = profiler.get_report();
        assert_eq!(report.edge_traversals.len(), 2);

        let agent_to_tools = report
            .edge_traversals
            .iter()
            .find(|e| e.from == "agent" && e.to == "tools")
            .unwrap();
        assert_eq!(agent_to_tools.count, 2);

        let tools_to_agent = report
            .edge_traversals
            .iter()
            .find(|e| e.from == "tools" && e.to == "agent")
            .unwrap();
        assert_eq!(tools_to_agent.count, 1);
    }

    // 5. Error recording
    #[test]
    fn test_error_recording() {
        let profiler = GraphProfiler::new(ProfilerOptions::default());
        profiler.record_error("agent", "timeout");
        profiler.record_error("agent", "rate_limit");
        profiler.record_error("tools", "not_found");

        // Also add node recordings so they show up in the report.
        profiler.start_node("agent");
        profiler.end_node("agent");

        let report = profiler.get_report();
        assert_eq!(report.error_count, 3);
        assert_eq!(report.node_profiles["agent"].error_count, 2);
    }

    // 6. Bottleneck detection
    #[test]
    fn test_bottleneck_detection() {
        let profiler = GraphProfiler::new(ProfilerOptions::default());

        profiler.start_node("fast");
        thread::sleep(Duration::from_millis(5));
        profiler.end_node("fast");

        profiler.start_node("slow");
        thread::sleep(Duration::from_millis(30));
        profiler.end_node("slow");

        let report = profiler.get_report();
        assert_eq!(report.bottleneck_node, Some("slow".to_string()));
    }

    // 7. Hot path detection
    #[test]
    fn test_hotpath_detection() {
        let profiler = GraphProfiler::new(ProfilerOptions::default());

        // Create a graph: start -> middle -> end
        profiler.start_node("start");
        thread::sleep(Duration::from_millis(10));
        profiler.end_node("start");

        profiler.start_node("middle");
        thread::sleep(Duration::from_millis(10));
        profiler.end_node("middle");

        profiler.start_node("end");
        thread::sleep(Duration::from_millis(10));
        profiler.end_node("end");

        profiler.record_edge("start", "middle");
        profiler.record_edge("middle", "end");

        let report = profiler.get_report();
        assert_eq!(report.hotpath, vec!["start", "middle", "end"]);
    }

    // 8. Report generation
    #[test]
    fn test_report_generation() {
        let profiler = GraphProfiler::new(ProfilerOptions::default());
        profiler.start_node("node_a");
        profiler.end_node("node_a");
        profiler.start_node("node_b");
        profiler.end_node("node_b");

        let report = profiler.get_report();
        assert_eq!(report.node_profiles.len(), 2);
        assert_eq!(report.total_node_calls, 2);
        assert!(report.total_duration > Duration::ZERO);
    }

    // 9. Summary formatting
    #[test]
    fn test_summary_formatting() {
        let profiler = GraphProfiler::new(ProfilerOptions::default());
        profiler.start_node("agent");
        thread::sleep(Duration::from_millis(5));
        profiler.end_node("agent");
        profiler.record_edge("agent", "tools");

        let report = profiler.get_report();
        let summary = report.summary();

        assert!(summary.contains("Graph Execution Profile"));
        assert!(summary.contains("agent"));
        assert!(summary.contains("Bottleneck"));
        assert!(summary.contains("Total"));
    }

    // 10. ProfileScope RAII
    #[test]
    fn test_profile_scope_raii() {
        let profiler = GraphProfiler::new(ProfilerOptions::default());
        {
            let _scope = profiler.scope("scoped_node");
            thread::sleep(Duration::from_millis(10));
        } // _scope dropped here, records end_node

        let report = profiler.get_report();
        assert!(report.node_profiles.contains_key("scoped_node"));
        assert_eq!(report.node_profiles["scoped_node"].call_count, 1);
        assert!(report.node_profiles["scoped_node"].total_duration >= Duration::from_millis(5));
    }

    // 11. Reset clears all data
    #[test]
    fn test_reset_clears_data() {
        let profiler = GraphProfiler::new(ProfilerOptions::default());
        profiler.start_node("x");
        profiler.end_node("x");
        profiler.record_edge("x", "y");
        profiler.record_error("x", "boom");

        let report = profiler.get_report();
        assert_eq!(report.total_node_calls, 1);
        assert_eq!(report.error_count, 1);

        profiler.reset();

        let report = profiler.get_report();
        assert_eq!(report.total_node_calls, 0);
        assert_eq!(report.error_count, 0);
        assert!(report.node_profiles.is_empty());
        assert!(report.edge_traversals.is_empty());
    }

    // 12. Options: disabled profiler
    #[test]
    fn test_disabled_profiler() {
        let opts = ProfilerOptions {
            enabled: false,
            ..Default::default()
        };
        let profiler = GraphProfiler::new(opts);
        profiler.start_node("agent");
        profiler.end_node("agent");
        profiler.record_edge("a", "b");
        profiler.record_error("a", "err");

        let report = profiler.get_report();
        assert_eq!(report.total_node_calls, 0);
        assert!(report.edge_traversals.is_empty());
        assert_eq!(report.error_count, 0);
    }

    // 13. Sample rate filtering
    #[test]
    fn test_sample_rate_filtering() {
        let opts = ProfilerOptions {
            sample_rate: 0.5,
            ..Default::default()
        };
        let profiler = GraphProfiler::new(opts);

        // With sample_rate=0.5, roughly every other call is recorded.
        for _ in 0..10 {
            profiler.start_node("sampled");
            profiler.end_node("sampled");
        }

        let report = profiler.get_report();
        // At 0.5 rate, we expect about 5 out of 10, but the exact count
        // depends on the sampling logic. Just verify it's less than 10.
        let calls = report
            .node_profiles
            .get("sampled")
            .map(|p| p.call_count)
            .unwrap_or(0);
        assert!(calls < 10, "expected fewer calls due to sampling, got {calls}");
        assert!(calls > 0, "expected some calls to be sampled");
    }

    // 14. Percentage calculation
    #[test]
    fn test_percentage_calculation() {
        let profiler = GraphProfiler::new(ProfilerOptions::default());

        profiler.start_node("alpha");
        thread::sleep(Duration::from_millis(20));
        profiler.end_node("alpha");

        profiler.start_node("beta");
        thread::sleep(Duration::from_millis(20));
        profiler.end_node("beta");

        let report = profiler.get_report();
        let total_pct: f64 = report
            .node_profiles
            .values()
            .map(|p| p.percentage_of_total)
            .sum();

        // Total percentages should sum to ~100%.
        assert!(
            (total_pct - 100.0).abs() < 1.0,
            "total percentage was {total_pct}, expected ~100.0"
        );
    }

    // 15. Concurrent profiling (thread safety)
    #[test]
    fn test_concurrent_profiling() {
        let profiler = GraphProfiler::new(ProfilerOptions::default());
        let mut handles = Vec::new();

        for i in 0..4 {
            let p = profiler.clone();
            let node_name = format!("thread_{i}");
            handles.push(thread::spawn(move || {
                for _ in 0..5 {
                    p.start_node(&node_name);
                    thread::sleep(Duration::from_millis(1));
                    p.end_node(&node_name);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        let report = profiler.get_report();
        assert_eq!(report.node_profiles.len(), 4);
        assert_eq!(report.total_node_calls, 20);
    }

    // 16. Display trait for ProfileReport
    #[test]
    fn test_display_trait() {
        let profiler = GraphProfiler::new(ProfilerOptions::default());
        profiler.start_node("n");
        profiler.end_node("n");
        let report = profiler.get_report();
        let display = format!("{}", report);
        assert!(!display.is_empty());
        assert!(display.contains("Graph Execution Profile"));
    }
}
