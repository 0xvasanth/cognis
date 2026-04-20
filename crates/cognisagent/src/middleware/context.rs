//! Context manager middleware — manages contextual information injected into
//! agent conversations, including system context, user profiles, session data,
//! and dynamic context sources.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::Mutex;

use crate::middleware::{AgentState, Middleware, Result};

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// Where in the message list context should be injected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContextPosition {
    /// Insert a system message before the first system message.
    BeforeSystem,
    /// Insert a system message after the first system message.
    #[default]
    AfterSystem,
    /// Insert a system message before the first user/human message.
    BeforeUser,
    /// Insert a system message after the first user/human message.
    AfterUser,
}

/// Strategy for handling context overflow when the token budget is exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PriorityMode {
    /// Drop entries with the lowest priority first.
    #[default]
    DropLowest,
    /// Truncate all entries proportionally.
    TruncateAll,
    /// Drop the oldest entries first (by creation time).
    DropOldest,
}

/// Configuration for the [`ContextMiddleware`].
#[derive(Debug, Clone)]
pub struct ContextManagerConfig {
    /// Maximum token budget for injected context. `None` means unlimited.
    pub max_context_tokens: Option<usize>,
    /// Where to inject context in the message list.
    pub context_position: ContextPosition,
    /// Separator placed between individual context entries.
    pub separator: String,
    /// How to handle overflow when the budget is exceeded.
    pub priority_mode: PriorityMode,
}

impl Default for ContextManagerConfig {
    fn default() -> Self {
        Self {
            max_context_tokens: None,
            context_position: ContextPosition::AfterSystem,
            separator: "\n---\n".to_string(),
            priority_mode: PriorityMode::DropLowest,
        }
    }
}

// ---------------------------------------------------------------------------
// ContextEntry
// ---------------------------------------------------------------------------

/// A single piece of context produced by a [`ContextSource`].
#[derive(Debug, Clone)]
pub struct ContextEntry {
    /// The textual content to inject.
    pub content: String,
    /// Name of the source that produced this entry.
    pub source_name: String,
    /// Priority — higher values are kept first when trimming.
    pub priority: u32,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, Value>,
    /// Optional time-to-live; entries older than their TTL are expired.
    pub ttl: Option<Duration>,
    /// When the entry was created.
    pub created_at: Instant,
}

impl ContextEntry {
    /// Create a new context entry with the given content, source name, and priority.
    pub fn new(content: impl Into<String>, source_name: impl Into<String>, priority: u32) -> Self {
        Self {
            content: content.into(),
            source_name: source_name.into(),
            priority,
            metadata: HashMap::new(),
            ttl: None,
            created_at: Instant::now(),
        }
    }

    /// Returns `true` if this entry has expired based on its TTL.
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            self.created_at.elapsed() > ttl
        } else {
            false
        }
    }

    /// Approximate token count (rough estimate: 1 token per 4 chars).
    pub fn estimated_tokens(&self) -> usize {
        self.content.len().div_ceil(4)
    }
}

// ---------------------------------------------------------------------------
// ContextSource trait
// ---------------------------------------------------------------------------

/// Trait for pluggable context sources that produce [`ContextEntry`] values.
#[async_trait]
pub trait ContextSource: Send + Sync {
    /// Retrieve contextual information, optionally filtered by `query`.
    async fn get_context(&self, query: &str) -> Result<ContextEntry>;

    /// A human-readable name for this source.
    fn name(&self) -> &str;

    /// Priority of this source — higher values are kept first when trimming.
    fn priority(&self) -> u32;
}

// ---------------------------------------------------------------------------
// Built-in context sources
// ---------------------------------------------------------------------------

/// A context source that always returns fixed text.
pub struct StaticContext {
    name: String,
    content: String,
    priority: u32,
}

impl StaticContext {
    /// Create a new static context source.
    pub fn new(name: impl Into<String>, content: impl Into<String>, priority: u32) -> Self {
        Self {
            name: name.into(),
            content: content.into(),
            priority,
        }
    }
}

#[async_trait]
impl ContextSource for StaticContext {
    async fn get_context(&self, _query: &str) -> Result<ContextEntry> {
        Ok(ContextEntry::new(
            self.content.clone(),
            self.name.clone(),
            self.priority,
        ))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn priority(&self) -> u32 {
        self.priority
    }
}

/// A context source that provides user profile information.
pub struct UserProfileContext {
    name: String,
    user_name: String,
    preferences: HashMap<String, String>,
    priority: u32,
}

impl UserProfileContext {
    /// Create a new user profile context source.
    pub fn new(
        user_name: impl Into<String>,
        preferences: HashMap<String, String>,
        priority: u32,
    ) -> Self {
        Self {
            name: "user_profile".to_string(),
            user_name: user_name.into(),
            preferences,
            priority,
        }
    }
}

#[async_trait]
impl ContextSource for UserProfileContext {
    async fn get_context(&self, _query: &str) -> Result<ContextEntry> {
        let mut lines = vec![format!("User: {}", self.user_name)];
        for (k, v) in &self.preferences {
            lines.push(format!("  {k}: {v}"));
        }
        let mut entry = ContextEntry::new(lines.join("\n"), self.name.clone(), self.priority);
        entry.metadata.insert(
            "user_name".to_string(),
            Value::String(self.user_name.clone()),
        );
        Ok(entry)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn priority(&self) -> u32 {
        self.priority
    }
}

/// A context source backed by thread-safe mutable key-value state.
pub struct SessionContext {
    name: String,
    data: Arc<Mutex<HashMap<String, Value>>>,
    priority: u32,
}

impl SessionContext {
    /// Create a new session context source.
    pub fn new(priority: u32) -> Self {
        Self {
            name: "session".to_string(),
            data: Arc::new(Mutex::new(HashMap::new())),
            priority,
        }
    }

    /// Set a session value.
    pub async fn set(&self, key: impl Into<String>, value: Value) {
        self.data.lock().await.insert(key.into(), value);
    }

    /// Get a session value.
    pub async fn get(&self, key: &str) -> Option<Value> {
        self.data.lock().await.get(key).cloned()
    }

    /// Remove a session value.
    pub async fn remove(&self, key: &str) -> Option<Value> {
        self.data.lock().await.remove(key)
    }
}

#[async_trait]
impl ContextSource for SessionContext {
    async fn get_context(&self, _query: &str) -> Result<ContextEntry> {
        let data = self.data.lock().await;
        if data.is_empty() {
            return Ok(ContextEntry::new("", self.name.clone(), self.priority));
        }
        let lines: Vec<String> = data.iter().map(|(k, v)| format!("  {k}: {v}")).collect();
        let content = format!("Session data:\n{}", lines.join("\n"));
        let mut entry = ContextEntry::new(content, self.name.clone(), self.priority);
        for (k, v) in data.iter() {
            entry.metadata.insert(k.clone(), v.clone());
        }
        Ok(entry)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn priority(&self) -> u32 {
        self.priority
    }
}

/// A context source that provides the current date/time and timezone.
pub struct TimeContext {
    name: String,
    timezone: String,
    priority: u32,
}

impl TimeContext {
    /// Create a new time context source.
    pub fn new(timezone: impl Into<String>, priority: u32) -> Self {
        Self {
            name: "time".to_string(),
            timezone: timezone.into(),
            priority,
        }
    }
}

#[async_trait]
impl ContextSource for TimeContext {
    async fn get_context(&self, _query: &str) -> Result<ContextEntry> {
        let now = std::time::SystemTime::now();
        let dur = now
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        let secs = dur.as_secs();
        let content = format!(
            "Current time: {} (unix timestamp), Timezone: {}",
            secs, self.timezone
        );
        Ok(ContextEntry::new(content, self.name.clone(), self.priority))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn priority(&self) -> u32 {
        self.priority
    }
}

/// A context source that reads selected environment variables.
pub struct EnvironmentContext {
    name: String,
    var_names: Vec<String>,
    priority: u32,
}

impl EnvironmentContext {
    /// Create a new environment context source that reads the given variable names.
    pub fn new(var_names: Vec<String>, priority: u32) -> Self {
        Self {
            name: "environment".to_string(),
            var_names,
            priority,
        }
    }
}

#[async_trait]
impl ContextSource for EnvironmentContext {
    async fn get_context(&self, _query: &str) -> Result<ContextEntry> {
        let mut lines = Vec::new();
        for var in &self.var_names {
            if let Ok(val) = std::env::var(var) {
                lines.push(format!("  {var}: {val}"));
            }
        }
        let content = if lines.is_empty() {
            String::new()
        } else {
            format!("Environment:\n{}", lines.join("\n"))
        };
        Ok(ContextEntry::new(content, self.name.clone(), self.priority))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn priority(&self) -> u32 {
        self.priority
    }
}

// ---------------------------------------------------------------------------
// ContextManager
// ---------------------------------------------------------------------------

/// Manages a collection of [`ContextSource`]s and produces combined context.
pub struct ContextManager {
    sources: Arc<Mutex<Vec<Box<dyn ContextSource>>>>,
    config: ContextManagerConfig,
}

impl ContextManager {
    /// Create a new context manager with the given configuration.
    pub fn new(config: ContextManagerConfig) -> Self {
        Self {
            sources: Arc::new(Mutex::new(Vec::new())),
            config,
        }
    }

    /// Add a context source.
    pub async fn add_source(&self, source: Box<dyn ContextSource>) {
        self.sources.lock().await.push(source);
    }

    /// Remove a context source by name. Returns `true` if a source was removed.
    pub async fn remove_source(&self, name: &str) -> bool {
        let mut sources = self.sources.lock().await;
        let before = sources.len();
        sources.retain(|s| s.name() != name);
        sources.len() < before
    }

    /// Gather context from all sources, filtered by TTL expiration.
    pub async fn get_all_context(&self, query: &str) -> Vec<ContextEntry> {
        let sources = self.sources.lock().await;
        let mut entries = Vec::new();
        for source in sources.iter() {
            if let Ok(entry) = source.get_context(query).await {
                if !entry.is_expired() && !entry.content.is_empty() {
                    entries.push(entry);
                }
            }
        }
        // Sort by priority descending.
        entries.sort_by_key(|b| std::cmp::Reverse(b.priority));
        entries
    }

    /// Gather and format context into a single string, respecting the token budget.
    pub async fn get_formatted_context(&self, query: &str) -> String {
        let mut entries = self.get_all_context(query).await;

        if let Some(budget) = self.config.max_context_tokens {
            entries = self.apply_budget(entries, budget);
        }

        entries
            .iter()
            .map(|e| e.content.as_str())
            .collect::<Vec<_>>()
            .join(&self.config.separator)
    }

    /// Apply the token budget according to the configured priority mode.
    fn apply_budget(&self, mut entries: Vec<ContextEntry>, budget: usize) -> Vec<ContextEntry> {
        match self.config.priority_mode {
            PriorityMode::DropLowest => {
                // Entries are already sorted by priority descending.
                let mut total = 0usize;
                let mut kept = Vec::new();
                for entry in entries {
                    let tokens = entry.estimated_tokens();
                    if total + tokens <= budget {
                        total += tokens;
                        kept.push(entry);
                    }
                    // else: drop this entry (lowest priority, since we iterate highest first)
                }
                kept
            }
            PriorityMode::TruncateAll => {
                let total_tokens: usize = entries.iter().map(|e| e.estimated_tokens()).sum();
                if total_tokens <= budget {
                    return entries;
                }
                let ratio = budget as f64 / total_tokens as f64;
                for entry in &mut entries {
                    let max_chars = (entry.content.len() as f64 * ratio) as usize;
                    if entry.content.len() > max_chars {
                        entry.content = entry.content[..max_chars].to_string();
                    }
                }
                entries
            }
            PriorityMode::DropOldest => {
                // Sort by creation time descending (newest first).
                entries.sort_by_key(|b| std::cmp::Reverse(b.created_at));
                let mut total = 0usize;
                let mut kept = Vec::new();
                for entry in entries {
                    let tokens = entry.estimated_tokens();
                    if total + tokens <= budget {
                        total += tokens;
                        kept.push(entry);
                    }
                }
                kept
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ContextMiddleware
// ---------------------------------------------------------------------------

/// Middleware that injects contextual information into agent conversations.
pub struct ContextMiddleware {
    manager: Arc<ContextManager>,
}

impl ContextMiddleware {
    /// Create a new context middleware with the given manager.
    pub fn new(manager: Arc<ContextManager>) -> Self {
        Self { manager }
    }

    /// Get a reference to the underlying context manager.
    pub fn manager(&self) -> &ContextManager {
        &self.manager
    }
}

#[async_trait]
impl Middleware for ContextMiddleware {
    fn name(&self) -> &str {
        "context"
    }

    async fn before_model(&self, state: &mut AgentState) -> Result<()> {
        let context = self.manager.get_formatted_context("").await;
        if context.is_empty() {
            return Ok(());
        }

        let context_msg = json!({
            "type": "system",
            "content": context
        });

        if let Some(messages) = state.get_mut("messages").and_then(|v| v.as_array_mut()) {
            let pos = match self.manager.config.context_position {
                ContextPosition::BeforeSystem => 0,
                ContextPosition::AfterSystem => {
                    if messages
                        .first()
                        .and_then(|m| m.get("type"))
                        .and_then(|t| t.as_str())
                        == Some("system")
                    {
                        1
                    } else {
                        0
                    }
                }
                ContextPosition::BeforeUser => messages
                    .iter()
                    .position(|m| {
                        m.get("type")
                            .and_then(|t| t.as_str())
                            .map(|t| t == "human" || t == "user")
                            .unwrap_or(false)
                    })
                    .unwrap_or(messages.len()),
                ContextPosition::AfterUser => messages
                    .iter()
                    .position(|m| {
                        m.get("type")
                            .and_then(|t| t.as_str())
                            .map(|t| t == "human" || t == "user")
                            .unwrap_or(false)
                    })
                    .map(|i| i + 1)
                    .unwrap_or(messages.len()),
            };

            let pos = pos.min(messages.len());
            messages.insert(pos, context_msg);
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Test 1: StaticContext returns fixed text
    #[tokio::test]
    async fn test_static_context_returns_fixed_text() {
        let src = StaticContext::new("instructions", "Always be helpful.", 10);
        let entry = src.get_context("anything").await.unwrap();
        assert_eq!(entry.content, "Always be helpful.");
        assert_eq!(entry.source_name, "instructions");
        assert_eq!(entry.priority, 10);
    }

    // Test 2: UserProfileContext with user data
    #[tokio::test]
    async fn test_user_profile_context_with_data() {
        let mut prefs = HashMap::new();
        prefs.insert("theme".to_string(), "dark".to_string());
        prefs.insert("language".to_string(), "en".to_string());
        let src = UserProfileContext::new("Alice", prefs, 5);

        let entry = src.get_context("").await.unwrap();
        assert!(entry.content.contains("User: Alice"));
        assert!(entry.content.contains("theme: dark"));
        assert!(entry.content.contains("language: en"));
        assert_eq!(
            entry.metadata.get("user_name"),
            Some(&Value::String("Alice".to_string()))
        );
    }

    // Test 3: SessionContext get/set values
    #[tokio::test]
    async fn test_session_context_get_set() {
        let session = SessionContext::new(3);
        assert!(session.get("key").await.is_none());

        session.set("key", json!("value")).await;
        assert_eq!(session.get("key").await, Some(json!("value")));

        session.set("key", json!(42)).await;
        assert_eq!(session.get("key").await, Some(json!(42)));

        session.remove("key").await;
        assert!(session.get("key").await.is_none());
    }

    // Test 4: TimeContext includes current date
    #[tokio::test]
    async fn test_time_context_includes_current_date() {
        let src = TimeContext::new("UTC", 1);
        let entry = src.get_context("").await.unwrap();
        assert!(entry.content.contains("Current time:"));
        assert!(entry.content.contains("Timezone: UTC"));
        assert_eq!(entry.source_name, "time");
    }

    // Test 5: EnvironmentContext reads env vars
    #[tokio::test]
    async fn test_environment_context_reads_env_vars() {
        std::env::set_var("DEEPAGENT_TEST_VAR", "hello_world");
        let src = EnvironmentContext::new(
            vec![
                "DEEPAGENT_TEST_VAR".to_string(),
                "NONEXISTENT_VAR_XYZ".to_string(),
            ],
            2,
        );

        let entry = src.get_context("").await.unwrap();
        assert!(entry.content.contains("DEEPAGENT_TEST_VAR: hello_world"));
        assert!(!entry.content.contains("NONEXISTENT_VAR_XYZ"));
        std::env::remove_var("DEEPAGENT_TEST_VAR");
    }

    // Test 6: ContextManager combines sources
    #[tokio::test]
    async fn test_context_manager_combines_sources() {
        let manager = ContextManager::new(ContextManagerConfig::default());
        manager
            .add_source(Box::new(StaticContext::new("a", "First context", 5)))
            .await;
        manager
            .add_source(Box::new(StaticContext::new("b", "Second context", 10)))
            .await;

        let entries = manager.get_all_context("").await;
        assert_eq!(entries.len(), 2);
        // Higher priority first.
        assert_eq!(entries[0].source_name, "b");
        assert_eq!(entries[1].source_name, "a");
    }

    // Test 7: Priority ordering
    #[tokio::test]
    async fn test_priority_ordering() {
        let manager = ContextManager::new(ContextManagerConfig::default());
        manager
            .add_source(Box::new(StaticContext::new("low", "Low priority", 1)))
            .await;
        manager
            .add_source(Box::new(StaticContext::new("high", "High priority", 100)))
            .await;
        manager
            .add_source(Box::new(StaticContext::new("mid", "Mid priority", 50)))
            .await;

        let entries = manager.get_all_context("").await;
        assert_eq!(entries[0].priority, 100);
        assert_eq!(entries[1].priority, 50);
        assert_eq!(entries[2].priority, 1);
    }

    // Test 8: Token budget enforcement (DropLowest)
    #[tokio::test]
    async fn test_token_budget_drop_lowest() {
        let config = ContextManagerConfig {
            max_context_tokens: Some(10), // Very small budget
            priority_mode: PriorityMode::DropLowest,
            ..Default::default()
        };
        let manager = ContextManager::new(config);
        // "High priority content" is ~5 tokens (21 chars / 4)
        manager
            .add_source(Box::new(StaticContext::new(
                "high",
                "High priority content",
                10,
            )))
            .await;
        // "Low priority content that is much longer and should be dropped" is ~16 tokens
        manager
            .add_source(Box::new(StaticContext::new(
                "low",
                "Low priority content that is much longer and should be dropped",
                1,
            )))
            .await;

        let formatted = manager.get_formatted_context("").await;
        assert!(formatted.contains("High priority"));
        // The low-priority entry should be dropped because combined they exceed budget.
        // High = ~6 tokens, Low = ~16 tokens, budget = 10.
        // High fits (6 <= 10), Low doesn't fit (6+16 > 10).
        assert!(!formatted.contains("should be dropped"));
    }

    // Test 9: TTL expiration
    #[tokio::test]
    async fn test_ttl_expiration() {
        let mut entry = ContextEntry::new("temporary", "src", 5);
        entry.ttl = Some(Duration::from_millis(0)); // Already expired
                                                    // Sleep a tiny bit to ensure elapsed > 0
        tokio::time::sleep(Duration::from_millis(1)).await;
        assert!(entry.is_expired());

        let entry2 = ContextEntry::new("permanent", "src", 5);
        assert!(!entry2.is_expired()); // No TTL, never expires

        let mut entry3 = ContextEntry::new("long-lived", "src", 5);
        entry3.ttl = Some(Duration::from_secs(3600));
        assert!(!entry3.is_expired());
    }

    // Test 10: Context position injection — AfterSystem
    #[tokio::test]
    async fn test_context_position_after_system() {
        let config = ContextManagerConfig {
            context_position: ContextPosition::AfterSystem,
            ..Default::default()
        };
        let manager = Arc::new(ContextManager::new(config));
        manager
            .add_source(Box::new(StaticContext::new("ctx", "Injected context", 5)))
            .await;

        let mw = ContextMiddleware::new(manager);
        let mut state = json!({
            "messages": [
                {"type": "system", "content": "You are a helpful assistant."},
                {"type": "human", "content": "Hello!"}
            ]
        });

        mw.before_model(&mut state).await.unwrap();
        let messages = state["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        // Context should be at index 1 (after system message).
        assert_eq!(messages[1]["content"], "Injected context");
    }

    // Test 11: Add/remove sources
    #[tokio::test]
    async fn test_add_remove_sources() {
        let manager = ContextManager::new(ContextManagerConfig::default());
        manager
            .add_source(Box::new(StaticContext::new("a", "Content A", 5)))
            .await;
        manager
            .add_source(Box::new(StaticContext::new("b", "Content B", 10)))
            .await;

        assert_eq!(manager.get_all_context("").await.len(), 2);

        let removed = manager.remove_source("a").await;
        assert!(removed);
        assert_eq!(manager.get_all_context("").await.len(), 1);
        assert_eq!(manager.get_all_context("").await[0].source_name, "b");

        let not_found = manager.remove_source("nonexistent").await;
        assert!(!not_found);
    }

    // Test 12: Formatted context output
    #[tokio::test]
    async fn test_formatted_context_output() {
        let config = ContextManagerConfig {
            separator: " | ".to_string(),
            ..Default::default()
        };
        let manager = ContextManager::new(config);
        manager
            .add_source(Box::new(StaticContext::new("a", "Alpha", 10)))
            .await;
        manager
            .add_source(Box::new(StaticContext::new("b", "Beta", 5)))
            .await;

        let formatted = manager.get_formatted_context("").await;
        // Higher priority first.
        assert_eq!(formatted, "Alpha | Beta");
    }

    // Test 13: Empty context sources
    #[tokio::test]
    async fn test_empty_context_sources() {
        let manager = Arc::new(ContextManager::new(ContextManagerConfig::default()));
        let mw = ContextMiddleware::new(manager.clone());

        let mut state = json!({
            "messages": [
                {"type": "human", "content": "Hello!"}
            ]
        });

        mw.before_model(&mut state).await.unwrap();
        // No context sources → no injection, message count unchanged.
        let messages = state["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);

        // Also test get_formatted_context with no sources.
        let formatted = manager.get_formatted_context("").await;
        assert!(formatted.is_empty());
    }

    // Test 14: Config defaults
    #[test]
    fn test_config_defaults() {
        let config = ContextManagerConfig::default();
        assert!(config.max_context_tokens.is_none());
        assert_eq!(config.context_position, ContextPosition::AfterSystem);
        assert_eq!(config.separator, "\n---\n");
        assert_eq!(config.priority_mode, PriorityMode::DropLowest);
    }

    // Test 15: Context position BeforeSystem
    #[tokio::test]
    async fn test_context_position_before_system() {
        let config = ContextManagerConfig {
            context_position: ContextPosition::BeforeSystem,
            ..Default::default()
        };
        let manager = Arc::new(ContextManager::new(config));
        manager
            .add_source(Box::new(StaticContext::new("ctx", "Before system", 5)))
            .await;

        let mw = ContextMiddleware::new(manager);
        let mut state = json!({
            "messages": [
                {"type": "system", "content": "System prompt."},
                {"type": "human", "content": "Hello!"}
            ]
        });

        mw.before_model(&mut state).await.unwrap();
        let messages = state["messages"].as_array().unwrap();
        assert_eq!(messages[0]["content"], "Before system");
    }

    // Test 16: Middleware name
    #[test]
    fn test_middleware_name() {
        let manager = Arc::new(ContextManager::new(ContextManagerConfig::default()));
        let mw = ContextMiddleware::new(manager);
        assert_eq!(mw.name(), "context");
    }

    // Test 17: SessionContext produces context entry with data
    #[tokio::test]
    async fn test_session_context_produces_context_entry() {
        let session = SessionContext::new(5);
        session.set("thread_id", json!("abc-123")).await;
        session.set("turn", json!(3)).await;

        let entry = session.get_context("").await.unwrap();
        assert!(entry.content.contains("Session data:"));
        assert!(entry.content.contains("thread_id"));
        assert!(entry.content.contains("turn"));
        assert!(!entry.metadata.is_empty());
    }

    // Test 18: ContextEntry estimated_tokens
    #[test]
    fn test_context_entry_estimated_tokens() {
        let entry = ContextEntry::new("", "src", 1);
        // Empty string: (0 + 3) / 4 = 0
        assert_eq!(entry.estimated_tokens(), 0);

        let entry = ContextEntry::new("Hello, world!", "src", 1);
        // 13 chars: (13 + 3) / 4 = 4
        assert_eq!(entry.estimated_tokens(), 4);
    }
}
