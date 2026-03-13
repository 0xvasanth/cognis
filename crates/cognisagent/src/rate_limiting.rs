//! Advanced rate limiting for agent operations.
//!
//! This module provides composable rate limiting primitives:
//!
//! - [`TokenBucket`] — classic token bucket with configurable capacity and refill rate
//! - [`SlidingWindowLimiter`] — tracks requests across per-second, per-minute, and per-hour windows
//! - [`CostBasedLimiter`] — monitors API spend against a budget cap
//! - [`CompositeLimiter`] — combines multiple limiters (all must pass)
//! - [`UsageTracker`] — records consumption over time and produces [`UsageReport`]s
//! - [`QuotaManager`] — enforces per-model or per-provider quotas
//! - [`RateLimitMiddleware`] — trait for integration with the agent pipeline

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// RateLimitPolicy
// ---------------------------------------------------------------------------

/// Policy to apply when a rate limit is exceeded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RateLimitPolicy {
    /// Reject the request immediately with an error.
    Reject,
    /// Queue the request until capacity is available.
    Queue,
    /// Back off with exponential delay.
    Backoff {
        /// Base delay before the first retry.
        base_delay: Duration,
        /// Maximum delay cap.
        max_delay: Duration,
        /// Multiplicative factor per retry.
        multiplier: f64,
    },
    /// Throttle to a reduced rate rather than rejecting.
    Throttle {
        /// The reduced rate (requests per second) to enforce.
        reduced_rate: f64,
    },
}

impl Default for RateLimitPolicy {
    fn default() -> Self {
        Self::Reject
    }
}

// ---------------------------------------------------------------------------
// RateLimitResult
// ---------------------------------------------------------------------------

/// The outcome of a rate limit check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitResult {
    /// Whether the request is allowed.
    pub allowed: bool,
    /// How long the caller should wait before retrying (zero if allowed).
    pub wait_time: Duration,
    /// Remaining tokens in the bucket (if applicable).
    pub remaining_tokens: f64,
    /// Remaining cost budget (if applicable).
    pub cost_remaining: f64,
    /// Human-readable reason when the request is denied.
    pub reason: Option<String>,
}

impl RateLimitResult {
    /// Convenience constructor for an allowed result.
    pub fn allowed(remaining_tokens: f64, cost_remaining: f64) -> Self {
        Self {
            allowed: true,
            wait_time: Duration::ZERO,
            remaining_tokens,
            cost_remaining,
            reason: None,
        }
    }

    /// Convenience constructor for a denied result.
    pub fn denied(wait_time: Duration, remaining_tokens: f64, cost_remaining: f64, reason: impl Into<String>) -> Self {
        Self {
            allowed: false,
            wait_time,
            remaining_tokens,
            cost_remaining,
            reason: Some(reason.into()),
        }
    }
}

// ---------------------------------------------------------------------------
// TokenBucket
// ---------------------------------------------------------------------------

/// A token-bucket rate limiter.
///
/// Tokens refill continuously at `refill_rate` tokens per second up to
/// `capacity`. Callers acquire tokens before performing work.
pub struct TokenBucket {
    /// Maximum number of tokens the bucket can hold.
    capacity: f64,
    /// Tokens added per second.
    refill_rate: f64,
    /// Current available tokens.
    available: Mutex<f64>,
    /// Timestamp of the last refill.
    last_refill: Mutex<Instant>,
}

impl TokenBucket {
    /// Create a new token bucket starting at full capacity.
    pub fn new(capacity: f64, refill_rate: f64) -> Self {
        Self {
            capacity,
            refill_rate,
            available: Mutex::new(capacity),
            last_refill: Mutex::new(Instant::now()),
        }
    }

    /// Refill tokens based on elapsed time since last refill.
    fn refill(&self) {
        let mut last = self.last_refill.lock().unwrap();
        let mut avail = self.available.lock().unwrap();
        let now = Instant::now();
        let elapsed = now.duration_since(*last).as_secs_f64();
        *avail = (*avail + elapsed * self.refill_rate).min(self.capacity);
        *last = now;
    }

    /// Try to acquire `n` tokens. Returns a [`RateLimitResult`].
    pub fn try_acquire(&self, n: f64) -> RateLimitResult {
        self.refill();
        let mut avail = self.available.lock().unwrap();
        if *avail >= n {
            *avail -= n;
            RateLimitResult::allowed(*avail, f64::INFINITY)
        } else {
            let deficit = n - *avail;
            let wait = if self.refill_rate > 0.0 {
                Duration::from_secs_f64(deficit / self.refill_rate)
            } else {
                Duration::from_secs(u64::MAX / 2)
            };
            RateLimitResult::denied(wait, *avail, f64::INFINITY, "token bucket exhausted")
        }
    }

    /// Force-acquire tokens even if it drives available below zero.
    /// Useful for recording after-the-fact usage.
    pub fn force_acquire(&self, n: f64) {
        self.refill();
        let mut avail = self.available.lock().unwrap();
        *avail -= n;
    }

    /// Return current available tokens (after refill).
    pub fn available(&self) -> f64 {
        self.refill();
        *self.available.lock().unwrap()
    }

    /// Return the bucket capacity.
    pub fn capacity(&self) -> f64 {
        self.capacity
    }

    /// Return the refill rate in tokens per second.
    pub fn refill_rate(&self) -> f64 {
        self.refill_rate
    }

    /// Calculate wait time needed to accumulate `n` tokens.
    pub fn wait_time_for(&self, n: f64) -> Duration {
        self.refill();
        let avail = *self.available.lock().unwrap();
        if avail >= n {
            return Duration::ZERO;
        }
        let deficit = n - avail;
        if self.refill_rate <= 0.0 {
            return Duration::from_secs(u64::MAX / 2);
        }
        Duration::from_secs_f64(deficit / self.refill_rate)
    }

    /// Reset the bucket to full capacity.
    pub fn reset(&self) {
        *self.available.lock().unwrap() = self.capacity;
        *self.last_refill.lock().unwrap() = Instant::now();
    }
}

// ---------------------------------------------------------------------------
// TimeWindow
// ---------------------------------------------------------------------------

/// Predefined time windows for the sliding window limiter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimeWindow {
    /// One-second window.
    PerSecond,
    /// One-minute window.
    PerMinute,
    /// One-hour window.
    PerHour,
}

impl TimeWindow {
    /// Return the duration of this window.
    pub fn duration(&self) -> Duration {
        match self {
            Self::PerSecond => Duration::from_secs(1),
            Self::PerMinute => Duration::from_secs(60),
            Self::PerHour => Duration::from_secs(3600),
        }
    }
}

// ---------------------------------------------------------------------------
// SlidingWindowLimiter
// ---------------------------------------------------------------------------

/// Entry tracking a single window's state.
struct WindowState {
    /// Maximum allowed count in this window.
    limit: u64,
    /// Timestamps of requests within the window.
    timestamps: Vec<Instant>,
}

/// Rate limiter that tracks request counts across sliding time windows.
pub struct SlidingWindowLimiter {
    windows: Mutex<Vec<(TimeWindow, WindowState)>>,
    policy: RateLimitPolicy,
}

impl SlidingWindowLimiter {
    /// Create a new sliding window limiter with no windows configured.
    pub fn new(policy: RateLimitPolicy) -> Self {
        Self {
            windows: Mutex::new(Vec::new()),
            policy,
        }
    }

    /// Add a window constraint.
    pub fn add_window(&self, window: TimeWindow, limit: u64) {
        let mut windows = self.windows.lock().unwrap();
        windows.push((window, WindowState { limit, timestamps: Vec::new() }));
    }

    /// Prune expired timestamps from all windows.
    fn prune(windows: &mut [(TimeWindow, WindowState)]) {
        let now = Instant::now();
        for (tw, state) in windows.iter_mut() {
            let cutoff = now.checked_sub(tw.duration()).unwrap_or(now);
            state.timestamps.retain(|t| *t > cutoff);
        }
    }

    /// Check whether a request is allowed and record it if so.
    pub fn check_and_record(&self) -> RateLimitResult {
        let mut windows = self.windows.lock().unwrap();
        Self::prune(&mut windows);

        // Find the most constrained window.
        let mut worst_wait = Duration::ZERO;
        let mut min_remaining: u64 = u64::MAX;
        let mut denied_reason: Option<String> = None;

        for (tw, state) in windows.iter() {
            let count = state.timestamps.len() as u64;
            let remaining = state.limit.saturating_sub(count);
            if remaining < min_remaining {
                min_remaining = remaining;
            }
            if count >= state.limit {
                // Estimate wait: time until the oldest entry expires.
                if let Some(oldest) = state.timestamps.first() {
                    let elapsed = oldest.elapsed();
                    let window_dur = tw.duration();
                    if elapsed < window_dur {
                        let wait = window_dur - elapsed;
                        if wait > worst_wait {
                            worst_wait = wait;
                        }
                    }
                }
                denied_reason = Some(format!(
                    "{:?} limit of {} exceeded",
                    tw, state.limit
                ));
            }
        }

        if let Some(reason) = denied_reason {
            return RateLimitResult::denied(worst_wait, min_remaining as f64, f64::INFINITY, reason);
        }

        // Record the request.
        let now = Instant::now();
        for (_tw, state) in windows.iter_mut() {
            state.timestamps.push(now);
        }

        RateLimitResult::allowed(min_remaining.saturating_sub(1) as f64, f64::INFINITY)
    }

    /// Get the current count for a specific window.
    pub fn count_for(&self, window: TimeWindow) -> u64 {
        let mut windows = self.windows.lock().unwrap();
        Self::prune(&mut windows);
        for (tw, state) in windows.iter() {
            if *tw == window {
                return state.timestamps.len() as u64;
            }
        }
        0
    }

    /// Get the policy.
    pub fn policy(&self) -> &RateLimitPolicy {
        &self.policy
    }

    /// Reset all windows.
    pub fn reset(&self) {
        let mut windows = self.windows.lock().unwrap();
        for (_tw, state) in windows.iter_mut() {
            state.timestamps.clear();
        }
    }
}

// ---------------------------------------------------------------------------
// CostBasedLimiter
// ---------------------------------------------------------------------------

/// Rate limiter that tracks API costs against a budget cap.
pub struct CostBasedLimiter {
    /// Maximum budget in dollars.
    budget: f64,
    /// Amount spent so far.
    spent: Mutex<f64>,
    /// Per-model cost overrides (model_name -> cost_per_call).
    model_costs: Mutex<HashMap<String, f64>>,
    /// Default cost per call when no model-specific cost is set.
    default_cost: f64,
    /// Policy when budget is exceeded.
    policy: RateLimitPolicy,
}

impl CostBasedLimiter {
    /// Create a new cost-based limiter with the given budget.
    pub fn new(budget: f64, default_cost: f64, policy: RateLimitPolicy) -> Self {
        Self {
            budget,
            spent: Mutex::new(0.0),
            model_costs: Mutex::new(HashMap::new()),
            default_cost,
            policy,
        }
    }

    /// Set the cost for a specific model.
    pub fn set_model_cost(&self, model: impl Into<String>, cost: f64) {
        self.model_costs.lock().unwrap().insert(model.into(), cost);
    }

    /// Get the cost for a model, falling back to the default.
    pub fn cost_for_model(&self, model: &str) -> f64 {
        self.model_costs
            .lock()
            .unwrap()
            .get(model)
            .copied()
            .unwrap_or(self.default_cost)
    }

    /// Check whether the budget allows a call with the given cost.
    pub fn check(&self, cost: f64) -> RateLimitResult {
        let spent = *self.spent.lock().unwrap();
        let remaining = self.budget - spent;
        if remaining >= cost {
            RateLimitResult::allowed(f64::INFINITY, remaining - cost)
        } else {
            RateLimitResult::denied(
                Duration::ZERO,
                f64::INFINITY,
                remaining,
                format!("budget exceeded: spent ${:.4} of ${:.4}", spent, self.budget),
            )
        }
    }

    /// Record a cost expenditure.
    pub fn record_cost(&self, cost: f64) {
        let mut spent = self.spent.lock().unwrap();
        *spent += cost;
    }

    /// Check and record a cost in one atomic operation.
    pub fn check_and_record(&self, cost: f64) -> RateLimitResult {
        let mut spent = self.spent.lock().unwrap();
        let remaining = self.budget - *spent;
        if remaining >= cost {
            *spent += cost;
            RateLimitResult::allowed(f64::INFINITY, remaining - cost)
        } else {
            RateLimitResult::denied(
                Duration::ZERO,
                f64::INFINITY,
                remaining,
                format!("budget exceeded: spent ${:.4} of ${:.4}", *spent, self.budget),
            )
        }
    }

    /// Return the total amount spent.
    pub fn total_spent(&self) -> f64 {
        *self.spent.lock().unwrap()
    }

    /// Return the remaining budget.
    pub fn remaining_budget(&self) -> f64 {
        self.budget - *self.spent.lock().unwrap()
    }

    /// Return the total budget.
    pub fn budget(&self) -> f64 {
        self.budget
    }

    /// Get the policy.
    pub fn policy(&self) -> &RateLimitPolicy {
        &self.policy
    }

    /// Reset spent amount to zero.
    pub fn reset(&self) {
        *self.spent.lock().unwrap() = 0.0;
    }
}

// ---------------------------------------------------------------------------
// CompositeLimiter
// ---------------------------------------------------------------------------

/// A limiter that combines multiple sub-limiters. A request is only
/// allowed when **all** sub-limiters approve.
pub struct CompositeLimiter {
    limiters: Vec<Box<dyn Limiter + Send + Sync>>,
}

/// Trait abstracting a single rate limit check.
pub trait Limiter: Send + Sync {
    /// Check whether a request is allowed.
    fn check(&self) -> RateLimitResult;
    /// Record that a request was made.
    fn record(&self);
    /// Reset the limiter state.
    fn reset(&self);
    /// Return a descriptive name.
    fn name(&self) -> &str;
}

// --- Limiter impls for our concrete types ---

impl Limiter for TokenBucket {
    fn check(&self) -> RateLimitResult {
        // Peek without consuming.
        self.refill();
        let avail = *self.available.lock().unwrap();
        if avail >= 1.0 {
            RateLimitResult::allowed(avail, f64::INFINITY)
        } else {
            let wait = self.wait_time_for(1.0);
            RateLimitResult::denied(wait, avail, f64::INFINITY, "token bucket exhausted")
        }
    }

    fn record(&self) {
        self.try_acquire(1.0);
    }

    fn reset(&self) {
        TokenBucket::reset(self);
    }

    fn name(&self) -> &str {
        "token_bucket"
    }
}

impl Limiter for SlidingWindowLimiter {
    fn check(&self) -> RateLimitResult {
        // We peek by checking without recording. We must clone timestamps
        // to avoid side effects, but for simplicity we check and do not record.
        let mut windows = self.windows.lock().unwrap();
        SlidingWindowLimiter::prune(&mut windows);

        let mut min_remaining: u64 = u64::MAX;
        let mut worst_wait = Duration::ZERO;
        let mut denied_reason: Option<String> = None;

        for (tw, state) in windows.iter() {
            let count = state.timestamps.len() as u64;
            let remaining = state.limit.saturating_sub(count);
            if remaining < min_remaining {
                min_remaining = remaining;
            }
            if count >= state.limit {
                if let Some(oldest) = state.timestamps.first() {
                    let elapsed = oldest.elapsed();
                    let window_dur = tw.duration();
                    if elapsed < window_dur {
                        let wait = window_dur - elapsed;
                        if wait > worst_wait {
                            worst_wait = wait;
                        }
                    }
                }
                denied_reason = Some(format!("{:?} limit exceeded", tw));
            }
        }

        if let Some(reason) = denied_reason {
            RateLimitResult::denied(worst_wait, min_remaining as f64, f64::INFINITY, reason)
        } else {
            RateLimitResult::allowed(min_remaining as f64, f64::INFINITY)
        }
    }

    fn record(&self) {
        let mut windows = self.windows.lock().unwrap();
        let now = Instant::now();
        for (_tw, state) in windows.iter_mut() {
            state.timestamps.push(now);
        }
    }

    fn reset(&self) {
        SlidingWindowLimiter::reset(self);
    }

    fn name(&self) -> &str {
        "sliding_window"
    }
}

impl Limiter for CostBasedLimiter {
    fn check(&self) -> RateLimitResult {
        CostBasedLimiter::check(self, self.default_cost)
    }

    fn record(&self) {
        self.record_cost(self.default_cost);
    }

    fn reset(&self) {
        CostBasedLimiter::reset(self);
    }

    fn name(&self) -> &str {
        "cost_based"
    }
}

impl CompositeLimiter {
    /// Create an empty composite limiter.
    pub fn new() -> Self {
        Self {
            limiters: Vec::new(),
        }
    }

    /// Add a limiter to the composite.
    pub fn add_limiter(&mut self, limiter: Box<dyn Limiter + Send + Sync>) {
        self.limiters.push(limiter);
    }

    /// Check all limiters. Returns denied with the worst wait time if any deny.
    pub fn check_all(&self) -> RateLimitResult {
        let mut worst_wait = Duration::ZERO;
        let mut min_tokens = f64::INFINITY;
        let mut min_cost = f64::INFINITY;
        let mut denied_reasons = Vec::new();

        for limiter in &self.limiters {
            let result = limiter.check();
            if result.remaining_tokens < min_tokens {
                min_tokens = result.remaining_tokens;
            }
            if result.cost_remaining < min_cost {
                min_cost = result.cost_remaining;
            }
            if !result.allowed {
                if result.wait_time > worst_wait {
                    worst_wait = result.wait_time;
                }
                if let Some(reason) = &result.reason {
                    denied_reasons.push(format!("{}: {}", limiter.name(), reason));
                }
            }
        }

        if denied_reasons.is_empty() {
            RateLimitResult::allowed(min_tokens, min_cost)
        } else {
            RateLimitResult::denied(worst_wait, min_tokens, min_cost, denied_reasons.join("; "))
        }
    }

    /// Check all limiters and record on all if allowed.
    pub fn check_and_record(&self) -> RateLimitResult {
        let result = self.check_all();
        if result.allowed {
            for limiter in &self.limiters {
                limiter.record();
            }
        }
        result
    }

    /// Reset all sub-limiters.
    pub fn reset_all(&self) {
        for limiter in &self.limiters {
            limiter.reset();
        }
    }

    /// Return the number of sub-limiters.
    pub fn len(&self) -> usize {
        self.limiters.len()
    }

    /// Return whether there are no sub-limiters.
    pub fn is_empty(&self) -> bool {
        self.limiters.is_empty()
    }
}

impl Default for CompositeLimiter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// UsageRecord
// ---------------------------------------------------------------------------

/// A single usage data point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRecord {
    /// Elapsed time since tracker creation.
    pub elapsed: Duration,
    /// Number of requests in this record.
    pub request_count: u64,
    /// Tokens consumed.
    pub tokens_used: u64,
    /// Cost incurred.
    pub cost: f64,
    /// Optional label (e.g. model name or provider).
    pub label: Option<String>,
}

// ---------------------------------------------------------------------------
// UsageTracker
// ---------------------------------------------------------------------------

/// Tracks consumption over time for monitoring and reporting.
pub struct UsageTracker {
    start: Instant,
    records: Mutex<Vec<UsageRecord>>,
    total_requests: Mutex<u64>,
    total_tokens: Mutex<u64>,
    total_cost: Mutex<f64>,
    peak_requests_per_second: Mutex<f64>,
    label_costs: Mutex<HashMap<String, f64>>,
    label_requests: Mutex<HashMap<String, u64>>,
}

impl UsageTracker {
    /// Create a new usage tracker.
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            records: Mutex::new(Vec::new()),
            total_requests: Mutex::new(0),
            total_tokens: Mutex::new(0),
            total_cost: Mutex::new(0.0),
            peak_requests_per_second: Mutex::new(0.0),
            label_costs: Mutex::new(HashMap::new()),
            label_requests: Mutex::new(HashMap::new()),
        }
    }

    /// Record a usage event.
    pub fn record(&self, tokens: u64, cost: f64, label: Option<&str>) {
        let elapsed = self.start.elapsed();

        {
            let mut total_req = self.total_requests.lock().unwrap();
            *total_req += 1;

            // Update peak RPS.
            let secs = elapsed.as_secs_f64().max(0.001);
            let rps = *total_req as f64 / secs;
            let mut peak = self.peak_requests_per_second.lock().unwrap();
            if rps > *peak {
                *peak = rps;
            }
        }

        {
            let mut total_tok = self.total_tokens.lock().unwrap();
            *total_tok += tokens;
        }

        {
            let mut total_c = self.total_cost.lock().unwrap();
            *total_c += cost;
        }

        if let Some(lbl) = label {
            let mut lc = self.label_costs.lock().unwrap();
            *lc.entry(lbl.to_string()).or_insert(0.0) += cost;
            let mut lr = self.label_requests.lock().unwrap();
            *lr.entry(lbl.to_string()).or_insert(0) += 1;
        }

        let record = UsageRecord {
            elapsed,
            request_count: 1,
            tokens_used: tokens,
            cost,
            label: label.map(|s| s.to_string()),
        };
        self.records.lock().unwrap().push(record);
    }

    /// Return total requests recorded.
    pub fn total_requests(&self) -> u64 {
        *self.total_requests.lock().unwrap()
    }

    /// Return total tokens consumed.
    pub fn total_tokens(&self) -> u64 {
        *self.total_tokens.lock().unwrap()
    }

    /// Return total cost.
    pub fn total_cost(&self) -> f64 {
        *self.total_cost.lock().unwrap()
    }

    /// Generate a usage report.
    pub fn report(&self) -> UsageReport {
        let elapsed = self.start.elapsed();
        let total_requests = *self.total_requests.lock().unwrap();
        let total_tokens = *self.total_tokens.lock().unwrap();
        let total_cost = *self.total_cost.lock().unwrap();
        let peak_rps = *self.peak_requests_per_second.lock().unwrap();
        let cost_breakdown = self.label_costs.lock().unwrap().clone();
        let request_breakdown = self.label_requests.lock().unwrap().clone();

        let avg_rps = if elapsed.as_secs_f64() > 0.0 {
            total_requests as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        let avg_tokens_per_request = if total_requests > 0 {
            total_tokens as f64 / total_requests as f64
        } else {
            0.0
        };

        let avg_cost_per_request = if total_requests > 0 {
            total_cost / total_requests as f64
        } else {
            0.0
        };

        UsageReport {
            period: elapsed,
            total_requests,
            total_tokens,
            total_cost,
            average_rps: avg_rps,
            peak_rps,
            average_tokens_per_request: avg_tokens_per_request,
            average_cost_per_request: avg_cost_per_request,
            cost_breakdown,
            request_breakdown,
        }
    }

    /// Reset all tracking data. The start time is reset to now.
    pub fn reset(&mut self) {
        self.start = Instant::now();
        self.records.lock().unwrap().clear();
        *self.total_requests.lock().unwrap() = 0;
        *self.total_tokens.lock().unwrap() = 0;
        *self.total_cost.lock().unwrap() = 0.0;
        *self.peak_requests_per_second.lock().unwrap() = 0.0;
        self.label_costs.lock().unwrap().clear();
        self.label_requests.lock().unwrap().clear();
    }
}

impl Default for UsageTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// UsageReport
// ---------------------------------------------------------------------------

/// Summary report of usage over a period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageReport {
    /// Duration of the reporting period.
    pub period: Duration,
    /// Total number of requests.
    pub total_requests: u64,
    /// Total tokens consumed.
    pub total_tokens: u64,
    /// Total cost in dollars.
    pub total_cost: f64,
    /// Average requests per second.
    pub average_rps: f64,
    /// Peak requests per second observed.
    pub peak_rps: f64,
    /// Average tokens per request.
    pub average_tokens_per_request: f64,
    /// Average cost per request.
    pub average_cost_per_request: f64,
    /// Cost broken down by label (model/provider).
    pub cost_breakdown: HashMap<String, f64>,
    /// Request count broken down by label.
    pub request_breakdown: HashMap<String, u64>,
}

// ---------------------------------------------------------------------------
// QuotaManager
// ---------------------------------------------------------------------------

/// Per-model or per-provider quota entry.
struct QuotaEntry {
    /// Maximum requests allowed in the window.
    max_requests: u64,
    /// Current request count.
    current: u64,
    /// Maximum cost allowed.
    max_cost: f64,
    /// Current cost spent.
    current_cost: f64,
    /// Window duration.
    window: Duration,
    /// Window start time.
    window_start: Instant,
}

impl QuotaEntry {
    fn maybe_reset(&mut self) {
        if self.window_start.elapsed() >= self.window {
            self.current = 0;
            self.current_cost = 0.0;
            self.window_start = Instant::now();
        }
    }
}

/// Manages per-model or per-provider quotas.
pub struct QuotaManager {
    quotas: Mutex<HashMap<String, QuotaEntry>>,
    default_window: Duration,
}

impl QuotaManager {
    /// Create a new quota manager with the given default window.
    pub fn new(default_window: Duration) -> Self {
        Self {
            quotas: Mutex::new(HashMap::new()),
            default_window,
        }
    }

    /// Set a quota for a key (model or provider name).
    pub fn set_quota(&self, key: impl Into<String>, max_requests: u64, max_cost: f64) {
        let mut quotas = self.quotas.lock().unwrap();
        quotas.insert(
            key.into(),
            QuotaEntry {
                max_requests,
                current: 0,
                max_cost,
                current_cost: 0.0,
                window: self.default_window,
                window_start: Instant::now(),
            },
        );
    }

    /// Set a quota with a custom window duration.
    pub fn set_quota_with_window(
        &self,
        key: impl Into<String>,
        max_requests: u64,
        max_cost: f64,
        window: Duration,
    ) {
        let mut quotas = self.quotas.lock().unwrap();
        quotas.insert(
            key.into(),
            QuotaEntry {
                max_requests,
                current: 0,
                max_cost,
                current_cost: 0.0,
                window,
                window_start: Instant::now(),
            },
        );
    }

    /// Check whether a request for the given key is allowed.
    pub fn check(&self, key: &str) -> RateLimitResult {
        let mut quotas = self.quotas.lock().unwrap();
        if let Some(entry) = quotas.get_mut(key) {
            entry.maybe_reset();
            if entry.current >= entry.max_requests {
                let remaining_window = entry.window.checked_sub(entry.window_start.elapsed())
                    .unwrap_or(Duration::ZERO);
                return RateLimitResult::denied(
                    remaining_window,
                    (entry.max_requests - entry.current) as f64,
                    entry.max_cost - entry.current_cost,
                    format!("quota exceeded for '{}'", key),
                );
            }
            if entry.current_cost >= entry.max_cost {
                return RateLimitResult::denied(
                    Duration::ZERO,
                    (entry.max_requests - entry.current) as f64,
                    entry.max_cost - entry.current_cost,
                    format!("cost quota exceeded for '{}'", key),
                );
            }
            RateLimitResult::allowed(
                (entry.max_requests - entry.current) as f64,
                entry.max_cost - entry.current_cost,
            )
        } else {
            // No quota set means unlimited.
            RateLimitResult::allowed(f64::INFINITY, f64::INFINITY)
        }
    }

    /// Record a request for the given key.
    pub fn record(&self, key: &str, cost: f64) {
        let mut quotas = self.quotas.lock().unwrap();
        if let Some(entry) = quotas.get_mut(key) {
            entry.maybe_reset();
            entry.current += 1;
            entry.current_cost += cost;
        }
    }

    /// Check and record in one operation.
    pub fn check_and_record(&self, key: &str, cost: f64) -> RateLimitResult {
        let mut quotas = self.quotas.lock().unwrap();
        if let Some(entry) = quotas.get_mut(key) {
            entry.maybe_reset();
            if entry.current >= entry.max_requests {
                let remaining_window = entry.window.checked_sub(entry.window_start.elapsed())
                    .unwrap_or(Duration::ZERO);
                return RateLimitResult::denied(
                    remaining_window,
                    0.0,
                    entry.max_cost - entry.current_cost,
                    format!("quota exceeded for '{}'", key),
                );
            }
            if entry.current_cost + cost > entry.max_cost {
                return RateLimitResult::denied(
                    Duration::ZERO,
                    (entry.max_requests - entry.current) as f64,
                    entry.max_cost - entry.current_cost,
                    format!("cost quota exceeded for '{}'", key),
                );
            }
            entry.current += 1;
            entry.current_cost += cost;
            RateLimitResult::allowed(
                (entry.max_requests - entry.current) as f64,
                entry.max_cost - entry.current_cost,
            )
        } else {
            RateLimitResult::allowed(f64::INFINITY, f64::INFINITY)
        }
    }

    /// Get current usage for a key.
    pub fn usage(&self, key: &str) -> Option<(u64, f64)> {
        let mut quotas = self.quotas.lock().unwrap();
        if let Some(entry) = quotas.get_mut(key) {
            entry.maybe_reset();
            Some((entry.current, entry.current_cost))
        } else {
            None
        }
    }

    /// Reset a specific key's counters.
    pub fn reset(&self, key: &str) {
        let mut quotas = self.quotas.lock().unwrap();
        if let Some(entry) = quotas.get_mut(key) {
            entry.current = 0;
            entry.current_cost = 0.0;
            entry.window_start = Instant::now();
        }
    }

    /// Reset all quotas.
    pub fn reset_all(&self) {
        let mut quotas = self.quotas.lock().unwrap();
        for entry in quotas.values_mut() {
            entry.current = 0;
            entry.current_cost = 0.0;
            entry.window_start = Instant::now();
        }
    }

    /// Return the list of configured keys.
    pub fn keys(&self) -> Vec<String> {
        self.quotas.lock().unwrap().keys().cloned().collect()
    }
}

// ---------------------------------------------------------------------------
// RateLimitMiddleware trait
// ---------------------------------------------------------------------------

/// Trait for integrating rate limiting into the agent pipeline.
///
/// Implementors can wrap any combination of limiters and apply them
/// as a middleware step before model or tool invocations.
pub trait RateLimitMiddleware: Send + Sync {
    /// Check rate limits before a model call. Returns the result.
    fn check_model_call(&self, model: &str) -> RateLimitResult;

    /// Check rate limits before a tool call. Returns the result.
    fn check_tool_call(&self, tool_name: &str) -> RateLimitResult;

    /// Record that a model call completed with the given token usage and cost.
    fn record_model_usage(&self, model: &str, tokens: u64, cost: f64);

    /// Record that a tool call completed.
    fn record_tool_usage(&self, tool_name: &str);

    /// Get the current usage report.
    fn usage_report(&self) -> UsageReport;

    /// Reset all limiters and tracking.
    fn reset(&self);
}

// ---------------------------------------------------------------------------
// DefaultRateLimitMiddleware
// ---------------------------------------------------------------------------

/// A default implementation of [`RateLimitMiddleware`] combining a
/// [`TokenBucket`], [`QuotaManager`], and [`UsageTracker`].
pub struct DefaultRateLimitMiddleware {
    bucket: TokenBucket,
    quotas: QuotaManager,
    tracker: UsageTracker,
    tool_bucket: TokenBucket,
}

impl DefaultRateLimitMiddleware {
    /// Create with the given model and tool rate limits.
    pub fn new(
        model_capacity: f64,
        model_refill_rate: f64,
        tool_capacity: f64,
        tool_refill_rate: f64,
    ) -> Self {
        Self {
            bucket: TokenBucket::new(model_capacity, model_refill_rate),
            quotas: QuotaManager::new(Duration::from_secs(60)),
            tracker: UsageTracker::new(),
            tool_bucket: TokenBucket::new(tool_capacity, tool_refill_rate),
        }
    }

    /// Access the quota manager.
    pub fn quota_manager(&self) -> &QuotaManager {
        &self.quotas
    }
}

impl RateLimitMiddleware for DefaultRateLimitMiddleware {
    fn check_model_call(&self, model: &str) -> RateLimitResult {
        let bucket_result = self.bucket.try_acquire(1.0);
        if !bucket_result.allowed {
            return bucket_result;
        }
        self.quotas.check(model)
    }

    fn check_tool_call(&self, _tool_name: &str) -> RateLimitResult {
        self.tool_bucket.try_acquire(1.0)
    }

    fn record_model_usage(&self, model: &str, tokens: u64, cost: f64) {
        self.quotas.record(model, cost);
        self.tracker.record(tokens, cost, Some(model));
    }

    fn record_tool_usage(&self, _tool_name: &str) {
        // Tool usage is tracked by the tool bucket acquire.
    }

    fn usage_report(&self) -> UsageReport {
        self.tracker.report()
    }

    fn reset(&self) {
        self.bucket.reset();
        self.tool_bucket.reset();
        self.quotas.reset_all();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    // -----------------------------------------------------------------------
    // RateLimitPolicy tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_policy_default_is_reject() {
        let policy = RateLimitPolicy::default();
        assert_eq!(policy, RateLimitPolicy::Reject);
    }

    #[test]
    fn test_policy_backoff_variant() {
        let policy = RateLimitPolicy::Backoff {
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            multiplier: 2.0,
        };
        match policy {
            RateLimitPolicy::Backoff { base_delay, max_delay, multiplier } => {
                assert_eq!(base_delay, Duration::from_millis(100));
                assert_eq!(max_delay, Duration::from_secs(10));
                assert!((multiplier - 2.0).abs() < f64::EPSILON);
            }
            _ => panic!("expected Backoff"),
        }
    }

    #[test]
    fn test_policy_throttle_variant() {
        let policy = RateLimitPolicy::Throttle { reduced_rate: 5.0 };
        match policy {
            RateLimitPolicy::Throttle { reduced_rate } => {
                assert!((reduced_rate - 5.0).abs() < f64::EPSILON);
            }
            _ => panic!("expected Throttle"),
        }
    }

    #[test]
    fn test_policy_queue_variant() {
        let policy = RateLimitPolicy::Queue;
        assert_eq!(policy, RateLimitPolicy::Queue);
    }

    #[test]
    fn test_policy_serialization() {
        let policy = RateLimitPolicy::Reject;
        let json = serde_json::to_string(&policy).unwrap();
        let deserialized: RateLimitPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, RateLimitPolicy::Reject);
    }

    // -----------------------------------------------------------------------
    // RateLimitResult tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_result_allowed() {
        let r = RateLimitResult::allowed(10.0, 5.0);
        assert!(r.allowed);
        assert_eq!(r.wait_time, Duration::ZERO);
        assert!((r.remaining_tokens - 10.0).abs() < f64::EPSILON);
        assert!((r.cost_remaining - 5.0).abs() < f64::EPSILON);
        assert!(r.reason.is_none());
    }

    #[test]
    fn test_result_denied() {
        let r = RateLimitResult::denied(Duration::from_secs(5), 0.0, 1.0, "over limit");
        assert!(!r.allowed);
        assert_eq!(r.wait_time, Duration::from_secs(5));
        assert!((r.remaining_tokens - 0.0).abs() < f64::EPSILON);
        assert!((r.cost_remaining - 1.0).abs() < f64::EPSILON);
        assert_eq!(r.reason.as_deref(), Some("over limit"));
    }

    #[test]
    fn test_result_serialization() {
        let r = RateLimitResult::allowed(42.0, 10.0);
        let json = serde_json::to_string(&r).unwrap();
        let r2: RateLimitResult = serde_json::from_str(&json).unwrap();
        assert!(r2.allowed);
        assert!((r2.remaining_tokens - 42.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // TokenBucket tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_token_bucket_creation() {
        let bucket = TokenBucket::new(100.0, 10.0);
        assert!((bucket.capacity() - 100.0).abs() < f64::EPSILON);
        assert!((bucket.refill_rate() - 10.0).abs() < f64::EPSILON);
        assert!((bucket.available() - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_token_bucket_acquire_success() {
        let bucket = TokenBucket::new(10.0, 1.0);
        let result = bucket.try_acquire(5.0);
        assert!(result.allowed);
        assert!(bucket.available() >= 4.0 && bucket.available() <= 5.5);
    }

    #[test]
    fn test_token_bucket_acquire_failure() {
        let bucket = TokenBucket::new(5.0, 1.0);
        let r1 = bucket.try_acquire(5.0);
        assert!(r1.allowed);
        let r2 = bucket.try_acquire(5.0);
        assert!(!r2.allowed);
        assert!(r2.wait_time > Duration::ZERO);
        assert_eq!(r2.reason.as_deref(), Some("token bucket exhausted"));
    }

    #[test]
    fn test_token_bucket_partial_acquire() {
        let bucket = TokenBucket::new(10.0, 1.0);
        assert!(bucket.try_acquire(3.0).allowed);
        assert!(bucket.try_acquire(3.0).allowed);
        assert!(bucket.try_acquire(3.0).allowed);
        // Only ~1 token left, requesting 3 should fail.
        assert!(!bucket.try_acquire(3.0).allowed);
    }

    #[test]
    fn test_token_bucket_refill() {
        let bucket = TokenBucket::new(10.0, 10000.0); // Very fast refill.
        assert!(bucket.try_acquire(10.0).allowed);
        assert!(!bucket.try_acquire(1.0).allowed);
        thread::sleep(Duration::from_millis(10));
        // Should have refilled significantly.
        assert!(bucket.try_acquire(1.0).allowed);
    }

    #[test]
    fn test_token_bucket_wait_time() {
        let bucket = TokenBucket::new(10.0, 2.0);
        assert!(bucket.try_acquire(10.0).allowed);
        let wait = bucket.wait_time_for(4.0);
        // At 2 tokens/s, need ~2 seconds for 4 tokens.
        assert!(wait >= Duration::from_millis(1500));
        assert!(wait <= Duration::from_millis(2500));
    }

    #[test]
    fn test_token_bucket_wait_time_zero_when_available() {
        let bucket = TokenBucket::new(10.0, 1.0);
        let wait = bucket.wait_time_for(5.0);
        assert_eq!(wait, Duration::ZERO);
    }

    #[test]
    fn test_token_bucket_force_acquire() {
        let bucket = TokenBucket::new(5.0, 1.0);
        bucket.force_acquire(10.0);
        // Available should be negative.
        assert!(bucket.available() < 0.0);
    }

    #[test]
    fn test_token_bucket_reset() {
        let bucket = TokenBucket::new(10.0, 1.0);
        bucket.try_acquire(10.0);
        assert!(bucket.available() < 1.0);
        bucket.reset();
        assert!((bucket.available() - 10.0).abs() < 1.0);
    }

    #[test]
    fn test_token_bucket_zero_refill_rate() {
        let bucket = TokenBucket::new(5.0, 0.0);
        assert!(bucket.try_acquire(5.0).allowed);
        let result = bucket.try_acquire(1.0);
        assert!(!result.allowed);
        // Wait time should be very large.
        assert!(result.wait_time > Duration::from_secs(1000));
    }

    // -----------------------------------------------------------------------
    // SlidingWindowLimiter tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sliding_window_single_window() {
        let limiter = SlidingWindowLimiter::new(RateLimitPolicy::Reject);
        limiter.add_window(TimeWindow::PerSecond, 3);

        assert!(limiter.check_and_record().allowed);
        assert!(limiter.check_and_record().allowed);
        assert!(limiter.check_and_record().allowed);
        assert!(!limiter.check_and_record().allowed);
    }

    #[test]
    fn test_sliding_window_multiple_windows() {
        let limiter = SlidingWindowLimiter::new(RateLimitPolicy::Reject);
        limiter.add_window(TimeWindow::PerSecond, 10);
        limiter.add_window(TimeWindow::PerMinute, 3);

        assert!(limiter.check_and_record().allowed);
        assert!(limiter.check_and_record().allowed);
        assert!(limiter.check_and_record().allowed);
        // Per-minute limit should kick in.
        let result = limiter.check_and_record();
        assert!(!result.allowed);
        assert!(result.reason.as_deref().unwrap().contains("PerMinute"));
    }

    #[test]
    fn test_sliding_window_count() {
        let limiter = SlidingWindowLimiter::new(RateLimitPolicy::Reject);
        limiter.add_window(TimeWindow::PerMinute, 100);

        limiter.check_and_record();
        limiter.check_and_record();
        limiter.check_and_record();

        assert_eq!(limiter.count_for(TimeWindow::PerMinute), 3);
        assert_eq!(limiter.count_for(TimeWindow::PerHour), 0); // Not configured.
    }

    #[test]
    fn test_sliding_window_reset() {
        let limiter = SlidingWindowLimiter::new(RateLimitPolicy::Reject);
        limiter.add_window(TimeWindow::PerSecond, 2);

        limiter.check_and_record();
        limiter.check_and_record();
        assert!(!limiter.check_and_record().allowed);

        limiter.reset();
        assert!(limiter.check_and_record().allowed);
    }

    #[test]
    fn test_sliding_window_policy() {
        let limiter = SlidingWindowLimiter::new(RateLimitPolicy::Queue);
        assert_eq!(*limiter.policy(), RateLimitPolicy::Queue);
    }

    #[test]
    fn test_sliding_window_remaining_decreases() {
        let limiter = SlidingWindowLimiter::new(RateLimitPolicy::Reject);
        limiter.add_window(TimeWindow::PerMinute, 5);

        let r1 = limiter.check_and_record();
        assert!(r1.allowed);
        assert!((r1.remaining_tokens - 4.0).abs() < f64::EPSILON);

        let r2 = limiter.check_and_record();
        assert!(r2.allowed);
        assert!((r2.remaining_tokens - 3.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // TimeWindow tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_time_window_durations() {
        assert_eq!(TimeWindow::PerSecond.duration(), Duration::from_secs(1));
        assert_eq!(TimeWindow::PerMinute.duration(), Duration::from_secs(60));
        assert_eq!(TimeWindow::PerHour.duration(), Duration::from_secs(3600));
    }

    #[test]
    fn test_time_window_serialization() {
        let tw = TimeWindow::PerMinute;
        let json = serde_json::to_string(&tw).unwrap();
        let tw2: TimeWindow = serde_json::from_str(&json).unwrap();
        assert_eq!(tw2, TimeWindow::PerMinute);
    }

    // -----------------------------------------------------------------------
    // CostBasedLimiter tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cost_limiter_allows_within_budget() {
        let limiter = CostBasedLimiter::new(10.0, 0.01, RateLimitPolicy::Reject);
        let result = limiter.check(1.0);
        assert!(result.allowed);
        assert!((result.cost_remaining - 9.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cost_limiter_denies_over_budget() {
        let limiter = CostBasedLimiter::new(1.0, 0.01, RateLimitPolicy::Reject);
        limiter.record_cost(0.8);
        let result = limiter.check(0.5);
        assert!(!result.allowed);
        assert!(result.reason.as_deref().unwrap().contains("budget exceeded"));
    }

    #[test]
    fn test_cost_limiter_check_and_record() {
        let limiter = CostBasedLimiter::new(1.0, 0.01, RateLimitPolicy::Reject);
        let r1 = limiter.check_and_record(0.4);
        assert!(r1.allowed);
        assert!((limiter.total_spent() - 0.4).abs() < f64::EPSILON);

        let r2 = limiter.check_and_record(0.4);
        assert!(r2.allowed);

        let r3 = limiter.check_and_record(0.4);
        assert!(!r3.allowed);
        // Only 0.8 was actually recorded (the 0.4 that failed was not).
        assert!((limiter.total_spent() - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cost_limiter_model_costs() {
        let limiter = CostBasedLimiter::new(10.0, 0.01, RateLimitPolicy::Reject);
        limiter.set_model_cost("gpt-4", 0.03);
        limiter.set_model_cost("gpt-3.5", 0.002);

        assert!((limiter.cost_for_model("gpt-4") - 0.03).abs() < f64::EPSILON);
        assert!((limiter.cost_for_model("gpt-3.5") - 0.002).abs() < f64::EPSILON);
        assert!((limiter.cost_for_model("unknown") - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cost_limiter_remaining_budget() {
        let limiter = CostBasedLimiter::new(5.0, 0.01, RateLimitPolicy::Reject);
        limiter.record_cost(2.0);
        assert!((limiter.remaining_budget() - 3.0).abs() < f64::EPSILON);
        assert!((limiter.budget() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cost_limiter_reset() {
        let limiter = CostBasedLimiter::new(5.0, 0.01, RateLimitPolicy::Reject);
        limiter.record_cost(4.0);
        limiter.reset();
        assert!((limiter.total_spent() - 0.0).abs() < f64::EPSILON);
        assert!((limiter.remaining_budget() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cost_limiter_policy() {
        let limiter = CostBasedLimiter::new(5.0, 0.01, RateLimitPolicy::Queue);
        assert_eq!(*limiter.policy(), RateLimitPolicy::Queue);
    }

    // -----------------------------------------------------------------------
    // CompositeLimiter tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_composite_empty_allows() {
        let composite = CompositeLimiter::new();
        let result = composite.check_all();
        assert!(result.allowed);
    }

    #[test]
    fn test_composite_all_pass() {
        let mut composite = CompositeLimiter::new();
        composite.add_limiter(Box::new(TokenBucket::new(10.0, 1.0)));
        composite.add_limiter(Box::new(CostBasedLimiter::new(100.0, 0.01, RateLimitPolicy::Reject)));

        let result = composite.check_all();
        assert!(result.allowed);
    }

    #[test]
    fn test_composite_one_fails() {
        let mut composite = CompositeLimiter::new();
        let bucket = TokenBucket::new(1.0, 0.0);
        bucket.try_acquire(1.0); // Drain it.
        composite.add_limiter(Box::new(bucket));
        composite.add_limiter(Box::new(CostBasedLimiter::new(100.0, 0.01, RateLimitPolicy::Reject)));

        let result = composite.check_all();
        assert!(!result.allowed);
        assert!(result.reason.as_deref().unwrap().contains("token_bucket"));
    }

    #[test]
    fn test_composite_check_and_record() {
        let mut composite = CompositeLimiter::new();
        composite.add_limiter(Box::new(TokenBucket::new(3.0, 0.0)));

        assert!(composite.check_and_record().allowed);
        assert!(composite.check_and_record().allowed);
        assert!(composite.check_and_record().allowed);
        assert!(!composite.check_and_record().allowed);
    }

    #[test]
    fn test_composite_reset_all() {
        let mut composite = CompositeLimiter::new();
        composite.add_limiter(Box::new(TokenBucket::new(1.0, 0.0)));

        assert!(composite.check_and_record().allowed);
        assert!(!composite.check_and_record().allowed);

        composite.reset_all();
        assert!(composite.check_and_record().allowed);
    }

    #[test]
    fn test_composite_len_and_empty() {
        let mut composite = CompositeLimiter::new();
        assert!(composite.is_empty());
        assert_eq!(composite.len(), 0);

        composite.add_limiter(Box::new(TokenBucket::new(10.0, 1.0)));
        assert!(!composite.is_empty());
        assert_eq!(composite.len(), 1);
    }

    #[test]
    fn test_composite_default() {
        let composite = CompositeLimiter::default();
        assert!(composite.is_empty());
    }

    // -----------------------------------------------------------------------
    // UsageTracker tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_usage_tracker_basic() {
        let tracker = UsageTracker::new();
        tracker.record(100, 0.01, Some("gpt-4"));
        tracker.record(200, 0.02, Some("gpt-4"));
        tracker.record(50, 0.005, Some("claude"));

        assert_eq!(tracker.total_requests(), 3);
        assert_eq!(tracker.total_tokens(), 350);
        assert!((tracker.total_cost() - 0.035).abs() < 1e-9);
    }

    #[test]
    fn test_usage_tracker_report() {
        let tracker = UsageTracker::new();
        tracker.record(100, 0.01, Some("gpt-4"));
        tracker.record(200, 0.02, Some("gpt-4"));
        tracker.record(50, 0.005, Some("claude"));

        let report = tracker.report();
        assert_eq!(report.total_requests, 3);
        assert_eq!(report.total_tokens, 350);
        assert!((report.total_cost - 0.035).abs() < 1e-9);
        assert!((report.average_tokens_per_request - 116.666).abs() < 1.0);
        assert!(report.cost_breakdown.contains_key("gpt-4"));
        assert!(report.cost_breakdown.contains_key("claude"));
        assert!((report.cost_breakdown["gpt-4"] - 0.03).abs() < 1e-9);
        assert!((report.cost_breakdown["claude"] - 0.005).abs() < 1e-9);
        assert_eq!(report.request_breakdown["gpt-4"], 2);
        assert_eq!(report.request_breakdown["claude"], 1);
    }

    #[test]
    fn test_usage_tracker_no_label() {
        let tracker = UsageTracker::new();
        tracker.record(100, 0.01, None);
        assert_eq!(tracker.total_requests(), 1);
        let report = tracker.report();
        assert!(report.cost_breakdown.is_empty());
    }

    #[test]
    fn test_usage_tracker_reset() {
        let mut tracker = UsageTracker::new();
        tracker.record(100, 0.01, Some("test"));
        tracker.reset();
        assert_eq!(tracker.total_requests(), 0);
        assert_eq!(tracker.total_tokens(), 0);
        assert!((tracker.total_cost() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_usage_tracker_default() {
        let tracker = UsageTracker::default();
        assert_eq!(tracker.total_requests(), 0);
    }

    #[test]
    fn test_usage_report_serialization() {
        let tracker = UsageTracker::new();
        tracker.record(100, 0.01, Some("gpt-4"));
        let report = tracker.report();
        let json = serde_json::to_string(&report).unwrap();
        let r2: UsageReport = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.total_requests, 1);
    }

    // -----------------------------------------------------------------------
    // QuotaManager tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_quota_manager_no_quota_allows() {
        let qm = QuotaManager::new(Duration::from_secs(60));
        let result = qm.check("unknown-model");
        assert!(result.allowed);
    }

    #[test]
    fn test_quota_manager_set_and_check() {
        let qm = QuotaManager::new(Duration::from_secs(60));
        qm.set_quota("gpt-4", 3, 1.0);

        let r1 = qm.check_and_record("gpt-4", 0.1);
        assert!(r1.allowed);
        let r2 = qm.check_and_record("gpt-4", 0.1);
        assert!(r2.allowed);
        let r3 = qm.check_and_record("gpt-4", 0.1);
        assert!(r3.allowed);
        let r4 = qm.check_and_record("gpt-4", 0.1);
        assert!(!r4.allowed);
        assert!(r4.reason.as_deref().unwrap().contains("quota exceeded"));
    }

    #[test]
    fn test_quota_manager_cost_quota() {
        let qm = QuotaManager::new(Duration::from_secs(60));
        qm.set_quota("gpt-4", 100, 0.5);

        let r1 = qm.check_and_record("gpt-4", 0.3);
        assert!(r1.allowed);
        let r2 = qm.check_and_record("gpt-4", 0.3);
        assert!(!r2.allowed);
        assert!(r2.reason.as_deref().unwrap().contains("cost quota exceeded"));
    }

    #[test]
    fn test_quota_manager_usage() {
        let qm = QuotaManager::new(Duration::from_secs(60));
        qm.set_quota("model-a", 10, 5.0);
        qm.record("model-a", 0.5);
        qm.record("model-a", 0.3);

        let (count, cost) = qm.usage("model-a").unwrap();
        assert_eq!(count, 2);
        assert!((cost - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_quota_manager_usage_none() {
        let qm = QuotaManager::new(Duration::from_secs(60));
        assert!(qm.usage("nonexistent").is_none());
    }

    #[test]
    fn test_quota_manager_reset_specific() {
        let qm = QuotaManager::new(Duration::from_secs(60));
        qm.set_quota("model-a", 3, 5.0);
        qm.record("model-a", 1.0);
        qm.record("model-a", 1.0);
        qm.reset("model-a");

        let (count, cost) = qm.usage("model-a").unwrap();
        assert_eq!(count, 0);
        assert!((cost - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_quota_manager_reset_all() {
        let qm = QuotaManager::new(Duration::from_secs(60));
        qm.set_quota("a", 10, 5.0);
        qm.set_quota("b", 10, 5.0);
        qm.record("a", 1.0);
        qm.record("b", 2.0);
        qm.reset_all();

        let (a_count, _) = qm.usage("a").unwrap();
        let (b_count, _) = qm.usage("b").unwrap();
        assert_eq!(a_count, 0);
        assert_eq!(b_count, 0);
    }

    #[test]
    fn test_quota_manager_keys() {
        let qm = QuotaManager::new(Duration::from_secs(60));
        qm.set_quota("alpha", 10, 5.0);
        qm.set_quota("beta", 10, 5.0);

        let mut keys = qm.keys();
        keys.sort();
        assert_eq!(keys, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_quota_manager_custom_window() {
        let qm = QuotaManager::new(Duration::from_secs(60));
        qm.set_quota_with_window("fast", 2, 10.0, Duration::from_millis(50));

        let r1 = qm.check_and_record("fast", 0.1);
        assert!(r1.allowed);
        let r2 = qm.check_and_record("fast", 0.1);
        assert!(r2.allowed);
        let r3 = qm.check_and_record("fast", 0.1);
        assert!(!r3.allowed);

        // Wait for window to reset.
        thread::sleep(Duration::from_millis(60));
        let r4 = qm.check_and_record("fast", 0.1);
        assert!(r4.allowed);
    }

    // -----------------------------------------------------------------------
    // Limiter trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_limiter_trait_token_bucket() {
        let bucket = TokenBucket::new(3.0, 0.0);
        let limiter: &dyn Limiter = &bucket;

        assert!(limiter.check().allowed);
        limiter.record();
        limiter.record();
        limiter.record();
        assert!(!limiter.check().allowed);

        limiter.reset();
        assert!(limiter.check().allowed);
        assert_eq!(limiter.name(), "token_bucket");
    }

    #[test]
    fn test_limiter_trait_cost_based() {
        let cost_limiter = CostBasedLimiter::new(0.05, 0.01, RateLimitPolicy::Reject);
        let limiter: &dyn Limiter = &cost_limiter;

        assert!(limiter.check().allowed);
        limiter.record(); // Records 0.01
        limiter.record(); // Records 0.01
        limiter.record(); // Records 0.01
        limiter.record(); // Records 0.01
        limiter.record(); // Records 0.01 => total 0.05
        assert!(!limiter.check().allowed);

        limiter.reset();
        assert!(limiter.check().allowed);
        assert_eq!(limiter.name(), "cost_based");
    }

    #[test]
    fn test_limiter_trait_sliding_window() {
        let sw = SlidingWindowLimiter::new(RateLimitPolicy::Reject);
        sw.add_window(TimeWindow::PerMinute, 2);
        let limiter: &dyn Limiter = &sw;

        assert!(limiter.check().allowed);
        limiter.record();
        limiter.record();
        assert!(!limiter.check().allowed);

        limiter.reset();
        assert!(limiter.check().allowed);
        assert_eq!(limiter.name(), "sliding_window");
    }

    // -----------------------------------------------------------------------
    // DefaultRateLimitMiddleware tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_middleware_model_check() {
        let mw = DefaultRateLimitMiddleware::new(3.0, 0.0, 10.0, 1.0);
        assert!(mw.check_model_call("gpt-4").allowed);
        assert!(mw.check_model_call("gpt-4").allowed);
        assert!(mw.check_model_call("gpt-4").allowed);
        assert!(!mw.check_model_call("gpt-4").allowed);
    }

    #[test]
    fn test_default_middleware_tool_check() {
        let mw = DefaultRateLimitMiddleware::new(10.0, 1.0, 2.0, 0.0);
        assert!(mw.check_tool_call("search").allowed);
        assert!(mw.check_tool_call("search").allowed);
        assert!(!mw.check_tool_call("search").allowed);
    }

    #[test]
    fn test_default_middleware_record_usage() {
        let mw = DefaultRateLimitMiddleware::new(10.0, 1.0, 10.0, 1.0);
        mw.record_model_usage("gpt-4", 500, 0.05);
        mw.record_model_usage("claude", 300, 0.03);

        let report = mw.usage_report();
        assert_eq!(report.total_requests, 2);
        assert_eq!(report.total_tokens, 800);
        assert!((report.total_cost - 0.08).abs() < 1e-9);
    }

    #[test]
    fn test_default_middleware_with_quotas() {
        let mw = DefaultRateLimitMiddleware::new(100.0, 10.0, 100.0, 10.0);
        mw.quota_manager().set_quota("gpt-4", 2, 1.0);

        assert!(mw.check_model_call("gpt-4").allowed);
        mw.record_model_usage("gpt-4", 100, 0.1);
        assert!(mw.check_model_call("gpt-4").allowed);
        mw.record_model_usage("gpt-4", 100, 0.1);
        // Quota of 2 requests reached.
        let result = mw.check_model_call("gpt-4");
        assert!(!result.allowed);
    }

    #[test]
    fn test_default_middleware_reset() {
        let mw = DefaultRateLimitMiddleware::new(2.0, 0.0, 2.0, 0.0);
        mw.check_model_call("test");
        mw.check_model_call("test");
        assert!(!mw.check_model_call("test").allowed);

        mw.reset();
        assert!(mw.check_model_call("test").allowed);
    }

    // -----------------------------------------------------------------------
    // UsageRecord tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_usage_record_serialization() {
        let record = UsageRecord {
            elapsed: Duration::from_secs(5),
            request_count: 1,
            tokens_used: 100,
            cost: 0.01,
            label: Some("gpt-4".to_string()),
        };
        let json = serde_json::to_string(&record).unwrap();
        let r2: UsageRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.tokens_used, 100);
        assert_eq!(r2.label.as_deref(), Some("gpt-4"));
    }

    #[test]
    fn test_usage_record_no_label() {
        let record = UsageRecord {
            elapsed: Duration::from_secs(1),
            request_count: 1,
            tokens_used: 50,
            cost: 0.005,
            label: None,
        };
        assert!(record.label.is_none());
    }

    // -----------------------------------------------------------------------
    // Edge case / integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_token_bucket_fractional_tokens() {
        let bucket = TokenBucket::new(1.5, 0.0);
        assert!(bucket.try_acquire(1.0).allowed);
        assert!(!bucket.try_acquire(1.0).allowed);
        assert!(bucket.try_acquire(0.5).allowed);
    }

    #[test]
    fn test_sliding_window_expiry() {
        let limiter = SlidingWindowLimiter::new(RateLimitPolicy::Reject);
        limiter.add_window(TimeWindow::PerSecond, 2);

        assert!(limiter.check_and_record().allowed);
        assert!(limiter.check_and_record().allowed);
        assert!(!limiter.check_and_record().allowed);

        // Wait for the 1-second window to expire.
        thread::sleep(Duration::from_millis(1100));

        assert!(limiter.check_and_record().allowed);
    }

    #[test]
    fn test_composite_multiple_failures_aggregated() {
        let mut composite = CompositeLimiter::new();

        let b1 = TokenBucket::new(0.0, 0.0); // Always empty.
        let b2 = CostBasedLimiter::new(0.0, 0.01, RateLimitPolicy::Reject); // Zero budget.
        composite.add_limiter(Box::new(b1));
        composite.add_limiter(Box::new(b2));

        let result = composite.check_all();
        assert!(!result.allowed);
        let reason = result.reason.unwrap();
        assert!(reason.contains("token_bucket"));
        assert!(reason.contains("cost_based"));
    }

    #[test]
    fn test_usage_tracker_peak_rps() {
        let tracker = UsageTracker::new();
        // Record several requests quickly.
        for _ in 0..10 {
            tracker.record(10, 0.001, None);
        }
        let report = tracker.report();
        assert!(report.peak_rps > 0.0);
    }

    #[test]
    fn test_quota_manager_record_without_quota() {
        let qm = QuotaManager::new(Duration::from_secs(60));
        // Recording on an unknown key should be a no-op.
        qm.record("unknown", 1.0);
        assert!(qm.usage("unknown").is_none());
    }

    #[test]
    fn test_cost_limiter_exact_budget() {
        let limiter = CostBasedLimiter::new(1.0, 0.01, RateLimitPolicy::Reject);
        let result = limiter.check_and_record(1.0);
        assert!(result.allowed);
        assert!((limiter.remaining_budget() - 0.0).abs() < f64::EPSILON);
        // Exactly at budget, next should fail.
        let result2 = limiter.check_and_record(0.01);
        assert!(!result2.allowed);
    }
}
