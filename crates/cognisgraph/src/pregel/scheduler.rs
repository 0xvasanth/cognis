//! Node scheduler for the Pregel execution engine.
//!
//! Manages execution order, priorities, and resource allocation for graph nodes.
//! Supports multiple scheduling strategies including FIFO, priority-based,
//! shortest-job-first, and round-robin scheduling.
//!
//! # Example
//!
//! ```rust
//! use cognisgraph::pregel::scheduler::{
//!     NodePriority, ScheduledNode, Scheduler, SchedulingStrategy,
//! };
//! use std::collections::HashSet;
//!
//! let mut scheduler = Scheduler::new(SchedulingStrategy::Priority);
//! scheduler.add_node(ScheduledNode::new("critical_task", NodePriority::Critical));
//! scheduler.add_node(ScheduledNode::new("background_task", NodePriority::Background));
//!
//! let completed = HashSet::new();
//! let next = scheduler.next(&completed);
//! assert_eq!(next.unwrap().name, "critical_task");
//! ```

use std::collections::HashSet;
use std::time::Duration;

use serde_json::Value;

/// Priority level for a scheduled node.
///
/// Determines execution order when using priority-based scheduling.
/// Higher priority nodes are executed before lower priority ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodePriority {
    /// Highest priority — must execute immediately.
    Critical,
    /// High priority — execute before normal tasks.
    High,
    /// Default priority level.
    Normal,
    /// Low priority — execute after normal tasks.
    Low,
    /// Lowest priority — execute only when no other work is pending.
    Background,
}

impl NodePriority {
    /// Returns the numeric weight for this priority level.
    ///
    /// Higher values indicate higher priority.
    pub fn weight(&self) -> u32 {
        match self {
            NodePriority::Critical => 100,
            NodePriority::High => 75,
            NodePriority::Normal => 50,
            NodePriority::Low => 25,
            NodePriority::Background => 10,
        }
    }
}

impl PartialOrd for NodePriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NodePriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.weight().cmp(&other.weight())
    }
}

/// A node scheduled for execution within the Pregel engine.
///
/// Contains metadata about the node including its priority, estimated cost,
/// dependencies on other nodes, and an optional deadline.
#[derive(Debug, Clone)]
pub struct ScheduledNode {
    /// The name of the node to execute.
    pub name: String,
    /// The priority level of this node.
    pub priority: NodePriority,
    /// Estimated execution cost (e.g., memory in MB or compute units).
    pub estimated_cost: u64,
    /// Names of nodes that must complete before this node can execute.
    pub dependencies: Vec<String>,
    /// Optional deadline by which this node should complete.
    pub deadline: Option<Duration>,
}

impl ScheduledNode {
    /// Create a new scheduled node with the given name and priority.
    ///
    /// Defaults to zero cost, no dependencies, and no deadline.
    pub fn new(name: impl Into<String>, priority: NodePriority) -> Self {
        Self {
            name: name.into(),
            priority,
            estimated_cost: 0,
            dependencies: Vec::new(),
            deadline: None,
        }
    }

    /// Set the estimated execution cost.
    pub fn with_cost(mut self, cost: u64) -> Self {
        self.estimated_cost = cost;
        self
    }

    /// Add a dependency on another node.
    pub fn with_dependency(mut self, dep: impl Into<String>) -> Self {
        self.dependencies.push(dep.into());
        self
    }

    /// Set an execution deadline.
    pub fn with_deadline(mut self, duration: Duration) -> Self {
        self.deadline = Some(duration);
        self
    }

    /// Check if all dependencies have been completed.
    ///
    /// Returns `true` if every node listed in `dependencies` is present in
    /// the `completed` set, or if there are no dependencies.
    pub fn is_ready(&self, completed: &HashSet<String>) -> bool {
        self.dependencies.iter().all(|dep| completed.contains(dep))
    }
}

/// Strategy used to determine node execution order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedulingStrategy {
    /// First-in, first-out — nodes are executed in the order they were added.
    FIFO,
    /// Priority-based — higher priority nodes execute first.
    Priority,
    /// Shortest job first — nodes with the lowest estimated cost execute first.
    ShortestJobFirst,
    /// Round-robin — nodes are cycled through in order.
    RoundRobin,
    /// Custom strategy identified by name.
    Custom(String),
}

/// Node scheduler that manages execution order based on a chosen strategy.
///
/// The scheduler maintains a queue of nodes and selects the next node(s) to
/// execute based on the configured [`SchedulingStrategy`] and dependency state.
pub struct Scheduler {
    /// The scheduling strategy in use.
    strategy: SchedulingStrategy,
    /// Nodes waiting to be scheduled.
    pending: Vec<ScheduledNode>,
    /// Original nodes for reset support.
    original: Vec<ScheduledNode>,
    /// Round-robin index tracker.
    rr_index: usize,
}

impl Scheduler {
    /// Create a new scheduler with the given strategy.
    pub fn new(strategy: SchedulingStrategy) -> Self {
        Self {
            strategy,
            pending: Vec::new(),
            original: Vec::new(),
            rr_index: 0,
        }
    }

    /// Add a node to the scheduler.
    pub fn add_node(&mut self, node: ScheduledNode) {
        self.original.push(node.clone());
        self.pending.push(node);
    }

    /// Return the next node to execute based on the current strategy.
    ///
    /// Only nodes whose dependencies are fully satisfied (present in `completed`)
    /// are considered. Returns `None` if no node is ready.
    pub fn next(&mut self, completed: &HashSet<String>) -> Option<ScheduledNode> {
        let ready_indices: Vec<usize> = self
            .pending
            .iter()
            .enumerate()
            .filter(|(_, n)| n.is_ready(completed))
            .map(|(i, _)| i)
            .collect();

        if ready_indices.is_empty() {
            return None;
        }

        let chosen_idx = match &self.strategy {
            SchedulingStrategy::FIFO => ready_indices[0],
            SchedulingStrategy::Priority => *ready_indices
                .iter()
                .max_by(|&&a, &&b| self.pending[a].priority.cmp(&self.pending[b].priority))
                .unwrap(),
            SchedulingStrategy::ShortestJobFirst => *ready_indices
                .iter()
                .min_by_key(|&&i| self.pending[i].estimated_cost)
                .unwrap(),
            SchedulingStrategy::RoundRobin => {
                // Pick the ready node at or after rr_index
                let idx = ready_indices
                    .iter()
                    .find(|&&i| i >= self.rr_index)
                    .copied()
                    .unwrap_or(ready_indices[0]);
                self.rr_index = idx;
                idx
            }
            SchedulingStrategy::Custom(_) => {
                // Custom strategies fall back to FIFO ordering.
                ready_indices[0]
            }
        };

        Some(self.pending.remove(chosen_idx))
    }

    /// Return a batch of ready nodes for parallel execution.
    ///
    /// Selects up to `max_parallel` nodes whose dependencies are satisfied,
    /// ordered according to the current strategy.
    pub fn next_batch(
        &mut self,
        completed: &HashSet<String>,
        max_parallel: usize,
    ) -> Vec<ScheduledNode> {
        let mut batch = Vec::new();

        for _ in 0..max_parallel {
            match self.next(completed) {
                Some(node) => batch.push(node),
                None => break,
            }
        }

        batch
    }

    /// Return the number of nodes still pending execution.
    pub fn remaining(&self) -> usize {
        self.pending.len()
    }

    /// Check if there are no more pending nodes.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Reset the scheduler, re-queuing all originally added nodes.
    pub fn reset(&mut self) {
        self.pending = self.original.clone();
        self.rr_index = 0;
    }
}

/// Metrics collected during scheduling operations.
#[derive(Debug, Clone)]
pub struct SchedulerMetrics {
    /// Total number of nodes that have been scheduled.
    pub nodes_scheduled: usize,
    /// Average wait time for nodes (placeholder for integration with timing).
    pub average_wait_time: Duration,
    /// Total estimated cost of all scheduled nodes.
    pub total_cost: u64,
    /// Number of batches created.
    pub batches_created: usize,
}

impl SchedulerMetrics {
    /// Create a new empty metrics instance.
    pub fn new() -> Self {
        Self {
            nodes_scheduled: 0,
            average_wait_time: Duration::ZERO,
            total_cost: 0,
            batches_created: 0,
        }
    }

    /// Record that nodes were scheduled with a given total cost.
    pub fn record_scheduling(&mut self, node_count: usize, cost: u64) {
        self.nodes_scheduled += node_count;
        self.total_cost += cost;
    }

    /// Record that a batch was created.
    pub fn record_batch(&mut self) {
        self.batches_created += 1;
    }

    /// Serialize metrics to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "nodes_scheduled": self.nodes_scheduled,
            "average_wait_time_ms": self.average_wait_time.as_millis(),
            "total_cost": self.total_cost,
            "batches_created": self.batches_created,
        })
    }
}

impl Default for SchedulerMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// A pool of execution resources with concurrency and memory limits.
///
/// Tracks how many concurrent slots and how much memory (in MB) are available
/// for node execution.
#[derive(Debug, Clone)]
pub struct ResourcePool {
    /// Maximum number of concurrent executions.
    max_concurrent: usize,
    /// Currently active executions.
    active: usize,
    /// Maximum available memory in MB.
    max_memory_mb: u64,
    /// Currently allocated memory in MB.
    used_memory_mb: u64,
}

impl ResourcePool {
    /// Create a new resource pool with the given limits.
    pub fn new(max_concurrent: usize, max_memory_mb: u64) -> Self {
        Self {
            max_concurrent,
            active: 0,
            max_memory_mb,
            used_memory_mb: 0,
        }
    }

    /// Attempt to acquire resources for a node with the given cost.
    ///
    /// Returns `true` if there is an available slot and enough memory.
    /// The resources are reserved upon success.
    pub fn acquire(&mut self, cost: u64) -> bool {
        if self.active < self.max_concurrent && self.used_memory_mb + cost <= self.max_memory_mb {
            self.active += 1;
            self.used_memory_mb += cost;
            true
        } else {
            false
        }
    }

    /// Release resources after a node completes execution.
    pub fn release(&mut self, cost: u64) {
        self.active = self.active.saturating_sub(1);
        self.used_memory_mb = self.used_memory_mb.saturating_sub(cost);
    }

    /// Return the number of available concurrent execution slots.
    pub fn available_slots(&self) -> usize {
        self.max_concurrent - self.active
    }

    /// Return the available memory in MB.
    pub fn available_memory(&self) -> u64 {
        self.max_memory_mb - self.used_memory_mb
    }

    /// Return the current resource utilization as a fraction (0.0 to 1.0).
    ///
    /// Computed as the average of slot utilization and memory utilization.
    pub fn utilization(&self) -> f64 {
        let slot_util = if self.max_concurrent > 0 {
            self.active as f64 / self.max_concurrent as f64
        } else {
            0.0
        };
        let mem_util = if self.max_memory_mb > 0 {
            self.used_memory_mb as f64 / self.max_memory_mb as f64
        } else {
            0.0
        };
        (slot_util + mem_util) / 2.0
    }
}

/// A scheduler that is aware of available execution resources.
///
/// Combines a [`Scheduler`] with a [`ResourcePool`] to ensure that nodes are
/// only scheduled when sufficient resources (concurrency slots, memory) are
/// available.
pub struct ResourceAwareScheduler {
    /// The underlying node scheduler.
    scheduler: Scheduler,
    /// The resource pool tracking available capacity.
    pool: ResourcePool,
    /// Scheduling metrics.
    metrics: SchedulerMetrics,
}

impl ResourceAwareScheduler {
    /// Create a new resource-aware scheduler.
    pub fn new(strategy: SchedulingStrategy, pool: ResourcePool) -> Self {
        Self {
            scheduler: Scheduler::new(strategy),
            pool,
            metrics: SchedulerMetrics::new(),
        }
    }

    /// Add a node to the underlying scheduler.
    pub fn add_node(&mut self, node: ScheduledNode) {
        self.scheduler.add_node(node);
    }

    /// Schedule the next batch of nodes, considering resource constraints.
    ///
    /// Returns nodes that are ready (dependencies met) and for which the
    /// resource pool has capacity.
    pub fn schedule_next(&mut self, completed: &HashSet<String>) -> Vec<ScheduledNode> {
        let max = self.pool.available_slots();
        if max == 0 {
            return Vec::new();
        }

        let mut batch = Vec::new();

        // We need to try nodes one at a time, checking resource availability.
        loop {
            if batch.len() >= max {
                break;
            }

            match self.scheduler.next(completed) {
                Some(node) => {
                    if self.pool.acquire(node.estimated_cost) {
                        self.metrics.record_scheduling(1, node.estimated_cost);
                        batch.push(node);
                    } else {
                        // Can't acquire resources — put the node back and stop.
                        self.scheduler.add_node(node);
                        break;
                    }
                }
                None => break,
            }
        }

        if !batch.is_empty() {
            self.metrics.record_batch();
        }

        batch
    }

    /// Release resources after a node with the given cost completes.
    pub fn release_resources(&mut self, cost: u64) {
        self.pool.release(cost);
    }

    /// Return a reference to the collected metrics.
    pub fn metrics(&self) -> &SchedulerMetrics {
        &self.metrics
    }

    /// Return the number of remaining pending nodes.
    pub fn remaining(&self) -> usize {
        self.scheduler.remaining()
    }

    /// Check if the scheduler has no more pending nodes.
    pub fn is_empty(&self) -> bool {
        self.scheduler.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── NodePriority tests ──────────────────────────────────────────────

    #[test]
    fn test_priority_weights() {
        assert_eq!(NodePriority::Critical.weight(), 100);
        assert_eq!(NodePriority::High.weight(), 75);
        assert_eq!(NodePriority::Normal.weight(), 50);
        assert_eq!(NodePriority::Low.weight(), 25);
        assert_eq!(NodePriority::Background.weight(), 10);
    }

    #[test]
    fn test_priority_ordering() {
        assert!(NodePriority::Critical > NodePriority::High);
        assert!(NodePriority::High > NodePriority::Normal);
        assert!(NodePriority::Normal > NodePriority::Low);
        assert!(NodePriority::Low > NodePriority::Background);
    }

    #[test]
    fn test_priority_equality() {
        assert_eq!(NodePriority::Normal, NodePriority::Normal);
        assert_ne!(NodePriority::Critical, NodePriority::Low);
    }

    #[test]
    fn test_priority_sort() {
        let mut priorities = vec![
            NodePriority::Low,
            NodePriority::Critical,
            NodePriority::Normal,
            NodePriority::Background,
            NodePriority::High,
        ];
        priorities.sort();
        assert_eq!(
            priorities,
            vec![
                NodePriority::Background,
                NodePriority::Low,
                NodePriority::Normal,
                NodePriority::High,
                NodePriority::Critical,
            ]
        );
    }

    // ── ScheduledNode tests ─────────────────────────────────────────────

    #[test]
    fn test_scheduled_node_new() {
        let node = ScheduledNode::new("test_node", NodePriority::Normal);
        assert_eq!(node.name, "test_node");
        assert_eq!(node.priority, NodePriority::Normal);
        assert_eq!(node.estimated_cost, 0);
        assert!(node.dependencies.is_empty());
        assert!(node.deadline.is_none());
    }

    #[test]
    fn test_scheduled_node_builder() {
        let node = ScheduledNode::new("node", NodePriority::High)
            .with_cost(512)
            .with_dependency("dep_a")
            .with_dependency("dep_b")
            .with_deadline(Duration::from_secs(30));

        assert_eq!(node.estimated_cost, 512);
        assert_eq!(node.dependencies, vec!["dep_a", "dep_b"]);
        assert_eq!(node.deadline, Some(Duration::from_secs(30)));
    }

    #[test]
    fn test_is_ready_no_deps() {
        let node = ScheduledNode::new("node", NodePriority::Normal);
        let completed = HashSet::new();
        assert!(node.is_ready(&completed));
    }

    #[test]
    fn test_is_ready_with_deps_satisfied() {
        let node = ScheduledNode::new("node", NodePriority::Normal)
            .with_dependency("a")
            .with_dependency("b");

        let completed: HashSet<String> = ["a".to_string(), "b".to_string()].into();
        assert!(node.is_ready(&completed));
    }

    #[test]
    fn test_is_ready_with_deps_unsatisfied() {
        let node = ScheduledNode::new("node", NodePriority::Normal)
            .with_dependency("a")
            .with_dependency("b");

        let completed: HashSet<String> = ["a".to_string()].into();
        assert!(!node.is_ready(&completed));
    }

    #[test]
    fn test_is_ready_empty_completed() {
        let node = ScheduledNode::new("node", NodePriority::Normal).with_dependency("dep");
        let completed = HashSet::new();
        assert!(!node.is_ready(&completed));
    }

    // ── FIFO scheduling tests ───────────────────────────────────────────

    #[test]
    fn test_fifo_ordering() {
        let mut scheduler = Scheduler::new(SchedulingStrategy::FIFO);
        scheduler.add_node(ScheduledNode::new("first", NodePriority::Low));
        scheduler.add_node(ScheduledNode::new("second", NodePriority::Critical));
        scheduler.add_node(ScheduledNode::new("third", NodePriority::Normal));

        let completed = HashSet::new();
        assert_eq!(scheduler.next(&completed).unwrap().name, "first");
        assert_eq!(scheduler.next(&completed).unwrap().name, "second");
        assert_eq!(scheduler.next(&completed).unwrap().name, "third");
        assert!(scheduler.next(&completed).is_none());
    }

    // ── Priority scheduling tests ───────────────────────────────────────

    #[test]
    fn test_priority_scheduling() {
        let mut scheduler = Scheduler::new(SchedulingStrategy::Priority);
        scheduler.add_node(ScheduledNode::new("low", NodePriority::Low));
        scheduler.add_node(ScheduledNode::new("critical", NodePriority::Critical));
        scheduler.add_node(ScheduledNode::new("normal", NodePriority::Normal));
        scheduler.add_node(ScheduledNode::new("high", NodePriority::High));

        let completed = HashSet::new();
        assert_eq!(scheduler.next(&completed).unwrap().name, "critical");
        assert_eq!(scheduler.next(&completed).unwrap().name, "high");
        assert_eq!(scheduler.next(&completed).unwrap().name, "normal");
        assert_eq!(scheduler.next(&completed).unwrap().name, "low");
    }

    // ── ShortestJobFirst scheduling tests ───────────────────────────────

    #[test]
    fn test_shortest_job_first() {
        let mut scheduler = Scheduler::new(SchedulingStrategy::ShortestJobFirst);
        scheduler.add_node(ScheduledNode::new("expensive", NodePriority::Normal).with_cost(1000));
        scheduler.add_node(ScheduledNode::new("cheap", NodePriority::Normal).with_cost(10));
        scheduler.add_node(ScheduledNode::new("medium", NodePriority::Normal).with_cost(500));

        let completed = HashSet::new();
        assert_eq!(scheduler.next(&completed).unwrap().name, "cheap");
        assert_eq!(scheduler.next(&completed).unwrap().name, "medium");
        assert_eq!(scheduler.next(&completed).unwrap().name, "expensive");
    }

    // ── RoundRobin scheduling tests ─────────────────────────────────────

    #[test]
    fn test_round_robin() {
        let mut scheduler = Scheduler::new(SchedulingStrategy::RoundRobin);
        scheduler.add_node(ScheduledNode::new("a", NodePriority::Normal));
        scheduler.add_node(ScheduledNode::new("b", NodePriority::Normal));
        scheduler.add_node(ScheduledNode::new("c", NodePriority::Normal));

        let completed = HashSet::new();
        assert_eq!(scheduler.next(&completed).unwrap().name, "a");
        assert_eq!(scheduler.next(&completed).unwrap().name, "b");
        assert_eq!(scheduler.next(&completed).unwrap().name, "c");
    }

    // ── next_batch tests ────────────────────────────────────────────────

    #[test]
    fn test_next_batch_parallel() {
        let mut scheduler = Scheduler::new(SchedulingStrategy::FIFO);
        scheduler.add_node(ScheduledNode::new("a", NodePriority::Normal));
        scheduler.add_node(ScheduledNode::new("b", NodePriority::Normal));
        scheduler.add_node(ScheduledNode::new("c", NodePriority::Normal));

        let completed = HashSet::new();
        let batch = scheduler.next_batch(&completed, 2);
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].name, "a");
        assert_eq!(batch[1].name, "b");
        assert_eq!(scheduler.remaining(), 1);
    }

    #[test]
    fn test_next_batch_fewer_ready_than_max() {
        let mut scheduler = Scheduler::new(SchedulingStrategy::FIFO);
        scheduler.add_node(ScheduledNode::new("a", NodePriority::Normal));

        let completed = HashSet::new();
        let batch = scheduler.next_batch(&completed, 5);
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_next_batch_with_dependencies() {
        let mut scheduler = Scheduler::new(SchedulingStrategy::FIFO);
        scheduler.add_node(ScheduledNode::new("a", NodePriority::Normal));
        scheduler.add_node(ScheduledNode::new("b", NodePriority::Normal).with_dependency("a"));
        scheduler.add_node(ScheduledNode::new("c", NodePriority::Normal));

        let completed = HashSet::new();
        let batch = scheduler.next_batch(&completed, 3);
        // "b" depends on "a", so only "a" and "c" are ready
        assert_eq!(batch.len(), 2);
        let names: Vec<&str> = batch.iter().map(|n| n.name.as_str()).collect();
        assert!(names.contains(&"a"));
        assert!(names.contains(&"c"));
        assert!(!names.contains(&"b"));
    }

    // ── Dependency-aware scheduling tests ───────────────────────────────

    #[test]
    fn test_dependency_chain() {
        let mut scheduler = Scheduler::new(SchedulingStrategy::FIFO);
        scheduler.add_node(ScheduledNode::new("step1", NodePriority::Normal));
        scheduler
            .add_node(ScheduledNode::new("step2", NodePriority::Normal).with_dependency("step1"));
        scheduler
            .add_node(ScheduledNode::new("step3", NodePriority::Normal).with_dependency("step2"));

        let mut completed = HashSet::new();

        // Only step1 is ready initially
        let n1 = scheduler.next(&completed).unwrap();
        assert_eq!(n1.name, "step1");
        assert!(scheduler.next(&completed).is_none());

        // After step1 completes, step2 is ready
        completed.insert("step1".to_string());
        let n2 = scheduler.next(&completed).unwrap();
        assert_eq!(n2.name, "step2");

        // After step2 completes, step3 is ready
        completed.insert("step2".to_string());
        let n3 = scheduler.next(&completed).unwrap();
        assert_eq!(n3.name, "step3");
    }

    #[test]
    fn test_no_ready_nodes() {
        let mut scheduler = Scheduler::new(SchedulingStrategy::FIFO);
        scheduler.add_node(
            ScheduledNode::new("blocked", NodePriority::Normal).with_dependency("missing"),
        );

        let completed = HashSet::new();
        assert!(scheduler.next(&completed).is_none());
        assert_eq!(scheduler.remaining(), 1);
    }

    #[test]
    fn test_all_nodes_ready() {
        let mut scheduler = Scheduler::new(SchedulingStrategy::Priority);
        scheduler.add_node(ScheduledNode::new("a", NodePriority::High));
        scheduler.add_node(ScheduledNode::new("b", NodePriority::Critical));
        scheduler.add_node(ScheduledNode::new("c", NodePriority::Normal));

        let completed = HashSet::new();
        let batch = scheduler.next_batch(&completed, 10);
        assert_eq!(batch.len(), 3);
        // Priority order
        assert_eq!(batch[0].name, "b");
        assert_eq!(batch[1].name, "a");
        assert_eq!(batch[2].name, "c");
    }

    #[test]
    fn test_circular_deps_handled_gracefully() {
        let mut scheduler = Scheduler::new(SchedulingStrategy::FIFO);
        scheduler.add_node(ScheduledNode::new("a", NodePriority::Normal).with_dependency("b"));
        scheduler.add_node(ScheduledNode::new("b", NodePriority::Normal).with_dependency("a"));

        let completed = HashSet::new();
        // Neither node can be scheduled — circular dependency
        assert!(scheduler.next(&completed).is_none());
        assert_eq!(scheduler.remaining(), 2);
        assert!(!scheduler.is_empty());
    }

    // ── Scheduler remaining / is_empty / reset tests ────────────────────

    #[test]
    fn test_remaining_and_is_empty() {
        let mut scheduler = Scheduler::new(SchedulingStrategy::FIFO);
        assert_eq!(scheduler.remaining(), 0);
        assert!(scheduler.is_empty());

        scheduler.add_node(ScheduledNode::new("a", NodePriority::Normal));
        assert_eq!(scheduler.remaining(), 1);
        assert!(!scheduler.is_empty());

        let completed = HashSet::new();
        scheduler.next(&completed);
        assert_eq!(scheduler.remaining(), 0);
        assert!(scheduler.is_empty());
    }

    #[test]
    fn test_reset() {
        let mut scheduler = Scheduler::new(SchedulingStrategy::FIFO);
        scheduler.add_node(ScheduledNode::new("a", NodePriority::Normal));
        scheduler.add_node(ScheduledNode::new("b", NodePriority::Normal));

        let completed = HashSet::new();
        scheduler.next(&completed);
        assert_eq!(scheduler.remaining(), 1);

        scheduler.reset();
        assert_eq!(scheduler.remaining(), 2);

        // Can schedule again from the beginning
        assert_eq!(scheduler.next(&completed).unwrap().name, "a");
    }

    // ── ResourcePool tests ──────────────────────────────────────────────

    #[test]
    fn test_resource_pool_acquire_release() {
        let mut pool = ResourcePool::new(2, 1024);
        assert_eq!(pool.available_slots(), 2);
        assert_eq!(pool.available_memory(), 1024);

        assert!(pool.acquire(256));
        assert_eq!(pool.available_slots(), 1);
        assert_eq!(pool.available_memory(), 768);

        assert!(pool.acquire(256));
        assert_eq!(pool.available_slots(), 0);

        // No more slots
        assert!(!pool.acquire(256));

        pool.release(256);
        assert_eq!(pool.available_slots(), 1);
        assert_eq!(pool.available_memory(), 1024 - 256);
    }

    #[test]
    fn test_resource_pool_memory_limit() {
        let mut pool = ResourcePool::new(10, 100);
        assert!(pool.acquire(50));
        assert!(pool.acquire(50));
        // No memory left
        assert!(!pool.acquire(1));
    }

    #[test]
    fn test_resource_pool_utilization() {
        let mut pool = ResourcePool::new(4, 1000);
        assert_eq!(pool.utilization(), 0.0);

        pool.acquire(500);
        // slot_util = 1/4 = 0.25, mem_util = 500/1000 = 0.5, avg = 0.375
        let util = pool.utilization();
        assert!((util - 0.375).abs() < 1e-10);
    }

    #[test]
    fn test_resource_pool_release_saturating() {
        let mut pool = ResourcePool::new(2, 100);
        // Release without acquire should not underflow
        pool.release(50);
        assert_eq!(pool.available_slots(), 2);
        assert_eq!(pool.available_memory(), 100);
    }

    // ── ResourceAwareScheduler tests ────────────────────────────────────

    #[test]
    fn test_resource_aware_scheduler_basic() {
        let pool = ResourcePool::new(2, 1024);
        let mut scheduler = ResourceAwareScheduler::new(SchedulingStrategy::Priority, pool);

        scheduler.add_node(ScheduledNode::new("a", NodePriority::High).with_cost(256));
        scheduler.add_node(ScheduledNode::new("b", NodePriority::Normal).with_cost(256));
        scheduler.add_node(ScheduledNode::new("c", NodePriority::Low).with_cost(256));

        let completed = HashSet::new();
        let batch = scheduler.schedule_next(&completed);
        // Only 2 slots available
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].name, "a"); // higher priority first
        assert_eq!(batch[1].name, "b");
        assert_eq!(scheduler.remaining(), 1);
    }

    #[test]
    fn test_resource_aware_scheduler_memory_constrained() {
        let pool = ResourcePool::new(10, 300);
        let mut scheduler = ResourceAwareScheduler::new(SchedulingStrategy::FIFO, pool);

        scheduler.add_node(ScheduledNode::new("big", NodePriority::Normal).with_cost(200));
        scheduler.add_node(ScheduledNode::new("also_big", NodePriority::Normal).with_cost(200));

        let completed = HashSet::new();
        let batch = scheduler.schedule_next(&completed);
        // Only enough memory for one
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].name, "big");
    }

    #[test]
    fn test_resource_aware_scheduler_release_and_reschedule() {
        let pool = ResourcePool::new(1, 1024);
        let mut scheduler = ResourceAwareScheduler::new(SchedulingStrategy::FIFO, pool);

        scheduler.add_node(ScheduledNode::new("a", NodePriority::Normal).with_cost(100));
        scheduler.add_node(ScheduledNode::new("b", NodePriority::Normal).with_cost(100));

        let completed = HashSet::new();

        let batch1 = scheduler.schedule_next(&completed);
        assert_eq!(batch1.len(), 1);
        assert_eq!(batch1[0].name, "a");

        // Release resources
        scheduler.release_resources(100);

        let batch2 = scheduler.schedule_next(&completed);
        assert_eq!(batch2.len(), 1);
        assert_eq!(batch2[0].name, "b");
    }

    #[test]
    fn test_resource_aware_scheduler_no_slots() {
        let pool = ResourcePool::new(0, 1024);
        let mut scheduler = ResourceAwareScheduler::new(SchedulingStrategy::FIFO, pool);
        scheduler.add_node(ScheduledNode::new("a", NodePriority::Normal));

        let completed = HashSet::new();
        let batch = scheduler.schedule_next(&completed);
        assert!(batch.is_empty());
    }

    // ── SchedulerMetrics tests ──────────────────────────────────────────

    #[test]
    fn test_metrics_new() {
        let metrics = SchedulerMetrics::new();
        assert_eq!(metrics.nodes_scheduled, 0);
        assert_eq!(metrics.total_cost, 0);
        assert_eq!(metrics.batches_created, 0);
    }

    #[test]
    fn test_metrics_record() {
        let mut metrics = SchedulerMetrics::new();
        metrics.record_scheduling(3, 500);
        metrics.record_batch();
        metrics.record_scheduling(2, 300);
        metrics.record_batch();

        assert_eq!(metrics.nodes_scheduled, 5);
        assert_eq!(metrics.total_cost, 800);
        assert_eq!(metrics.batches_created, 2);
    }

    #[test]
    fn test_metrics_to_json() {
        let mut metrics = SchedulerMetrics::new();
        metrics.record_scheduling(4, 1000);
        metrics.record_batch();

        let json = metrics.to_json();
        assert_eq!(json["nodes_scheduled"], 4);
        assert_eq!(json["total_cost"], 1000);
        assert_eq!(json["batches_created"], 1);
        assert_eq!(json["average_wait_time_ms"], 0);
    }

    #[test]
    fn test_resource_aware_scheduler_metrics_tracking() {
        let pool = ResourcePool::new(3, 2048);
        let mut scheduler = ResourceAwareScheduler::new(SchedulingStrategy::FIFO, pool);

        scheduler.add_node(ScheduledNode::new("a", NodePriority::Normal).with_cost(100));
        scheduler.add_node(ScheduledNode::new("b", NodePriority::Normal).with_cost(200));

        let completed = HashSet::new();
        let _batch = scheduler.schedule_next(&completed);

        let metrics = scheduler.metrics();
        assert_eq!(metrics.nodes_scheduled, 2);
        assert_eq!(metrics.total_cost, 300);
        assert_eq!(metrics.batches_created, 1);
    }

    // ── Custom strategy test ────────────────────────────────────────────

    #[test]
    fn test_custom_strategy_falls_back_to_fifo() {
        let mut scheduler = Scheduler::new(SchedulingStrategy::Custom("my_strategy".to_string()));
        scheduler.add_node(ScheduledNode::new("first", NodePriority::Low));
        scheduler.add_node(ScheduledNode::new("second", NodePriority::Critical));

        let completed = HashSet::new();
        assert_eq!(scheduler.next(&completed).unwrap().name, "first");
        assert_eq!(scheduler.next(&completed).unwrap().name, "second");
    }

    // ── Edge case: empty scheduler ──────────────────────────────────────

    #[test]
    fn test_empty_scheduler_next_returns_none() {
        let mut scheduler = Scheduler::new(SchedulingStrategy::FIFO);
        let completed = HashSet::new();
        assert!(scheduler.next(&completed).is_none());
    }

    #[test]
    fn test_empty_scheduler_next_batch_returns_empty() {
        let mut scheduler = Scheduler::new(SchedulingStrategy::FIFO);
        let completed = HashSet::new();
        let batch = scheduler.next_batch(&completed, 5);
        assert!(batch.is_empty());
    }
}
