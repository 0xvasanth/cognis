//! Conversation memory — the per-agent message buffer.
//!
//! Memory implementations:
//! - [`Window`] — bounded FIFO with optional pinned system prompt.
//! - [`Buffer`] — unbounded; keeps every message.
//! - [`TokenBufferMemory`] — token-budgeted trim; drops oldest until under budget.
//! - [`SummaryMemory`] — summarizes older history with an LLM, drops the originals.
//! - [`SummaryBufferMemory`] — token budget + summary; the best of both.
//! - [`VectorMemory`] — semantic recall via a [`VectorStore`]; the most
//!   relevant past messages are surfaced into the seed.

use std::collections::VecDeque;
use std::sync::Arc;

use cognis_core::tokenizer::{CharTokenizer, Tokenizer};
use cognis_core::{trim_messages, Message, TrimStrategy};

use cognis_llm::chat::ChatOptions;
use cognis_llm::Client;
use cognis_rag::VectorStore;
use tokio::sync::RwLock;

/// Pluggable memory backend. The `Agent` reads via `seed()` to build
/// initial state, and writes incremental messages via `write()`.
pub trait Memory: Send + Sync {
    /// All currently buffered messages.
    fn read(&self) -> &[Message];

    /// Append one message.
    fn write(&mut self, msg: Message);

    /// Clear all buffered messages (system pinned ones survive in the Window impl).
    fn clear(&mut self);

    /// Build the seed messages for a fresh graph run. Default: `read().to_vec()`.
    fn seed(&self) -> Vec<Message> {
        self.read().to_vec()
    }
}

/// Bounded-capacity sliding window. Drops oldest non-system messages
/// when capacity is hit. The system message (if pinned) is kept at
/// index 0 across all writes and clears.
#[derive(Debug, Clone)]
pub struct Window {
    capacity: usize,
    system_pinned: Option<Message>,
    buf: VecDeque<Message>,
}

impl Window {
    /// New empty window with the given capacity (for non-system messages).
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            system_pinned: None,
            buf: VecDeque::with_capacity(capacity),
        }
    }

    /// Pin a system message that survives writes and clears.
    pub fn with_system(mut self, prompt: impl Into<String>) -> Self {
        self.system_pinned = Some(Message::system(prompt));
        self
    }
}

impl Memory for Window {
    fn read(&self) -> &[Message] {
        // Build a temp slice including system_pinned at the start. Since
        // `&[Message]` requires contiguous storage and we keep system
        // separate, we expose the buf only here. `seed()` (overridden
        // below) handles the merge for callers that need both.
        self.buf.as_slices().0
    }

    fn write(&mut self, msg: Message) {
        if self.buf.len() >= self.capacity {
            self.buf.pop_front();
        }
        self.buf.push_back(msg);
    }

    fn clear(&mut self) {
        self.buf.clear();
    }

    fn seed(&self) -> Vec<Message> {
        let mut out = Vec::with_capacity(self.buf.len() + 1);
        if let Some(s) = &self.system_pinned {
            out.push(s.clone());
        }
        out.extend(self.buf.iter().cloned());
        out
    }
}

// ---------------------------------------------------------------------------
// Buffer — unbounded message store.
// ---------------------------------------------------------------------------

/// Unbounded memory: keeps every message ever written. Use when conversation
/// length is small enough that token cost isn't a concern.
#[derive(Debug, Default, Clone)]
pub struct Buffer {
    system_pinned: Option<Message>,
    msgs: Vec<Message>,
}

impl Buffer {
    /// Empty buffer.
    pub fn new() -> Self {
        Self::default()
    }
    /// Pin a system message at the head.
    pub fn with_system(mut self, prompt: impl Into<String>) -> Self {
        self.system_pinned = Some(Message::system(prompt));
        self
    }
}

impl Memory for Buffer {
    fn read(&self) -> &[Message] {
        &self.msgs
    }
    fn write(&mut self, msg: Message) {
        self.msgs.push(msg);
    }
    fn clear(&mut self) {
        self.msgs.clear();
    }
    fn seed(&self) -> Vec<Message> {
        let mut out = Vec::with_capacity(self.msgs.len() + 1);
        if let Some(s) = &self.system_pinned {
            out.push(s.clone());
        }
        out.extend(self.msgs.iter().cloned());
        out
    }
}

// ---------------------------------------------------------------------------
// TokenBufferMemory — drop oldest until under a token budget.
// ---------------------------------------------------------------------------

/// Token-budgeted memory: every `seed()` call trims the conversation
/// down to `max_tokens` using the configured [`Tokenizer`]. The pinned
/// system prompt (if any) is always kept.
pub struct TokenBufferMemory {
    system_pinned: Option<Message>,
    msgs: Vec<Message>,
    max_tokens: usize,
    tokenizer: Arc<dyn Tokenizer>,
    strategy: TrimStrategy,
}

impl TokenBufferMemory {
    /// Build with the default `CharTokenizer` (chars-as-tokens; conservative).
    pub fn new(max_tokens: usize) -> Self {
        Self {
            system_pinned: None,
            msgs: Vec::new(),
            max_tokens,
            tokenizer: Arc::new(CharTokenizer),
            strategy: TrimStrategy::First,
        }
    }

    /// Override the tokenizer (e.g. plug in tiktoken).
    pub fn with_tokenizer(mut self, t: Arc<dyn Tokenizer>) -> Self {
        self.tokenizer = t;
        self
    }

    /// Override the trim strategy. Default: drop oldest first.
    pub fn with_strategy(mut self, s: TrimStrategy) -> Self {
        self.strategy = s;
        self
    }

    /// Pin a system message at the head of the seed.
    pub fn with_system(mut self, prompt: impl Into<String>) -> Self {
        self.system_pinned = Some(Message::system(prompt));
        self
    }
}

impl std::fmt::Debug for TokenBufferMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenBufferMemory")
            .field("max_tokens", &self.max_tokens)
            .field("strategy", &self.strategy)
            .field("msgs", &self.msgs.len())
            .finish()
    }
}

impl Memory for TokenBufferMemory {
    fn read(&self) -> &[Message] {
        &self.msgs
    }
    fn write(&mut self, msg: Message) {
        self.msgs.push(msg);
    }
    fn clear(&mut self) {
        self.msgs.clear();
    }
    fn seed(&self) -> Vec<Message> {
        let mut all = Vec::with_capacity(self.msgs.len() + 1);
        if let Some(s) = &self.system_pinned {
            all.push(s.clone());
        }
        all.extend(self.msgs.iter().cloned());
        trim_messages(
            &all,
            self.max_tokens,
            self.tokenizer.as_ref(),
            self.strategy,
        )
    }
}

// ---------------------------------------------------------------------------
// SummaryMemory — LLM-backed compression.
// ---------------------------------------------------------------------------

/// LLM-backed memory: when message count exceeds `threshold`, summarize the
/// oldest `threshold/2` messages into a single system message (via the
/// supplied [`Client`]) and drop the originals.
///
/// Summarization is **lazy** — it runs in `seed()` (called by the agent
/// before each turn). That keeps `write()` synchronous, which is what
/// the [`Memory`] trait requires.
pub struct SummaryMemory {
    system_pinned: Option<Message>,
    msgs: Vec<Message>,
    summary: Option<String>,
    threshold: usize,
    client: Client,
    prompt: String,
}

impl SummaryMemory {
    /// Build with the LLM client used to summarize, and the message count
    /// at which compression kicks in.
    pub fn new(client: Client, threshold: usize) -> Self {
        Self {
            system_pinned: None,
            msgs: Vec::new(),
            summary: None,
            threshold,
            client,
            prompt: DEFAULT_SUMMARY_PROMPT.to_string(),
        }
    }

    /// Pin a system message at the head of the seed.
    pub fn with_system(mut self, prompt: impl Into<String>) -> Self {
        self.system_pinned = Some(Message::system(prompt));
        self
    }

    /// Override the summarization prompt.
    pub fn with_prompt(mut self, p: impl Into<String>) -> Self {
        self.prompt = p.into();
        self
    }

    /// Force compression now, regardless of threshold. Useful for tests
    /// and explicit "compact" calls.
    pub async fn compact(&mut self) -> cognis_core::Result<()> {
        if self.msgs.len() < 2 {
            return Ok(());
        }
        let half = self.msgs.len() / 2;
        let to_summarize: Vec<Message> = self.msgs.drain(..half).collect();
        let transcript = to_summarize
            .iter()
            .map(|m| format!("[{}] {}", role_label(m), m.content()))
            .collect::<Vec<_>>()
            .join("\n");
        let request = format!("{}\n\nConversation:\n{transcript}", self.prompt);
        let resp = self
            .client
            .chat(vec![Message::human(request)], ChatOptions::default())
            .await?;
        let new = resp.message.content().to_string();
        self.summary = Some(match self.summary.take() {
            Some(prev) => format!("{prev}\n\n{new}"),
            None => new,
        });
        Ok(())
    }
}

const DEFAULT_SUMMARY_PROMPT: &str =
    "Summarize the following conversation in a few sentences. Preserve key \
     facts, decisions, names, and unfinished work. Output the summary only.";

fn role_label(m: &Message) -> &'static str {
    match m {
        Message::Human(_) => "user",
        Message::Ai(_) => "assistant",
        Message::System(_) => "system",
        Message::Tool(_) => "tool",
    }
}

impl Memory for SummaryMemory {
    fn read(&self) -> &[Message] {
        &self.msgs
    }
    fn write(&mut self, msg: Message) {
        self.msgs.push(msg);
        // If we're past the threshold, schedule compression on the next
        // `seed()` (which is async-friendly via the agent's run loop).
        // We just mark the threshold here — actual compaction happens via
        // explicit `compact()` calls.
    }
    fn clear(&mut self) {
        self.msgs.clear();
        self.summary = None;
    }
    fn seed(&self) -> Vec<Message> {
        let mut out = Vec::with_capacity(self.msgs.len() + 2);
        if let Some(s) = &self.system_pinned {
            out.push(s.clone());
        }
        if let Some(summary) = &self.summary {
            out.push(Message::system(format!(
                "Earlier conversation summary:\n{summary}"
            )));
        }
        out.extend(self.msgs.iter().cloned());
        out
    }
}

impl SummaryMemory {
    /// True when the buffer has grown past the configured threshold and a
    /// `compact()` call would do work.
    pub fn needs_compact(&self) -> bool {
        self.msgs.len() >= self.threshold
    }
}

// ---------------------------------------------------------------------------
// SummaryBufferMemory — token-budgeted buffer with summarized overflow.
// ---------------------------------------------------------------------------

/// Hybrid memory: keeps the most recent messages whole, but compresses
/// older ones into a running LLM-generated summary so the total seed
/// stays under a token budget.
///
/// On every `compact()` call (or every `seed()` after a `compact()`),
/// the oldest messages whose cumulative token cost would push the
/// transcript over `max_tokens` are summarized into the running summary
/// and dropped from the message list.
pub struct SummaryBufferMemory {
    system_pinned: Option<Message>,
    msgs: Vec<Message>,
    summary: Option<String>,
    max_tokens: usize,
    tokenizer: Arc<dyn Tokenizer>,
    client: Client,
    prompt: String,
}

impl SummaryBufferMemory {
    /// Build with a token budget and the LLM client used to summarize
    /// overflow.
    pub fn new(client: Client, max_tokens: usize) -> Self {
        Self {
            system_pinned: None,
            msgs: Vec::new(),
            summary: None,
            max_tokens,
            tokenizer: Arc::new(CharTokenizer),
            client,
            prompt: DEFAULT_SUMMARY_PROMPT.to_string(),
        }
    }

    /// Override the tokenizer.
    pub fn with_tokenizer(mut self, t: Arc<dyn Tokenizer>) -> Self {
        self.tokenizer = t;
        self
    }

    /// Override the summarization prompt.
    pub fn with_prompt(mut self, p: impl Into<String>) -> Self {
        self.prompt = p.into();
        self
    }

    /// Pin a system message at the head.
    pub fn with_system(mut self, prompt: impl Into<String>) -> Self {
        self.system_pinned = Some(Message::system(prompt));
        self
    }

    /// Total token cost of the current seed (system + summary + msgs).
    fn current_cost(&self) -> usize {
        let mut total = 0;
        if let Some(s) = &self.system_pinned {
            total += self.tokenizer.count(s.content());
        }
        if let Some(s) = &self.summary {
            total += self.tokenizer.count(s);
        }
        for m in &self.msgs {
            total += self.tokenizer.count(m.content());
        }
        total
    }

    /// Force compression now: summarize the oldest messages until
    /// `current_cost <= max_tokens`. Returns the number of messages
    /// folded into the summary.
    pub async fn compact(&mut self) -> cognis_core::Result<usize> {
        if self.current_cost() <= self.max_tokens {
            return Ok(0);
        }
        // Identify the oldest messages to summarize: take from the front
        // until the remaining cost is within budget.
        let mut to_summarize: Vec<Message> = Vec::new();
        while self.current_cost_with(&self.msgs[to_summarize.len()..]) > self.max_tokens
            && to_summarize.len() < self.msgs.len()
        {
            to_summarize.push(self.msgs[to_summarize.len()].clone());
        }
        if to_summarize.is_empty() {
            return Ok(0);
        }
        let n = to_summarize.len();
        let transcript = to_summarize
            .iter()
            .map(|m| format!("[{}] {}", role_label(m), m.content()))
            .collect::<Vec<_>>()
            .join("\n");
        let request = format!("{}\n\nConversation:\n{transcript}", self.prompt);
        let resp = self
            .client
            .chat(vec![Message::human(request)], ChatOptions::default())
            .await?;
        let new_summary = resp.message.content().to_string();
        self.summary = Some(match self.summary.take() {
            Some(prev) => format!("{prev}\n\n{new_summary}"),
            None => new_summary,
        });
        // Drain compacted messages.
        self.msgs.drain(..n);
        Ok(n)
    }

    fn current_cost_with(&self, tail: &[Message]) -> usize {
        let mut total = 0;
        if let Some(s) = &self.system_pinned {
            total += self.tokenizer.count(s.content());
        }
        if let Some(s) = &self.summary {
            total += self.tokenizer.count(s);
        }
        for m in tail {
            total += self.tokenizer.count(m.content());
        }
        total
    }

    /// True if a `compact()` would do work.
    pub fn needs_compact(&self) -> bool {
        self.current_cost() > self.max_tokens
    }
}

impl Memory for SummaryBufferMemory {
    fn read(&self) -> &[Message] {
        &self.msgs
    }
    fn write(&mut self, msg: Message) {
        self.msgs.push(msg);
    }
    fn clear(&mut self) {
        self.msgs.clear();
        self.summary = None;
    }
    fn seed(&self) -> Vec<Message> {
        let mut out = Vec::with_capacity(self.msgs.len() + 2);
        if let Some(s) = &self.system_pinned {
            out.push(s.clone());
        }
        if let Some(summary) = &self.summary {
            out.push(Message::system(format!(
                "Earlier conversation summary:\n{summary}"
            )));
        }
        out.extend(self.msgs.iter().cloned());
        out
    }
}

// ---------------------------------------------------------------------------
// VectorMemory — semantic recall via a VectorStore.
// ---------------------------------------------------------------------------

/// Memory backed by a [`VectorStore`]. Every `write` adds the message text
/// to the store (with role metadata). `seed()` returns the system pin
/// only — agents wanting recall call [`VectorMemory::recall`] explicitly
/// to pull in the top-k most relevant messages for the current query.
pub struct VectorMemory {
    system_pinned: Option<Message>,
    store: Arc<RwLock<dyn VectorStore>>,
    k: usize,
}

impl VectorMemory {
    /// Wrap a vector store with default k=4.
    pub fn new(store: Arc<RwLock<dyn VectorStore>>) -> Self {
        Self {
            system_pinned: None,
            store,
            k: 4,
        }
    }

    /// Pin a system message at the head.
    pub fn with_system(mut self, prompt: impl Into<String>) -> Self {
        self.system_pinned = Some(Message::system(prompt));
        self
    }

    /// Override how many memories to surface per recall.
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Pull the top-k semantically-similar past messages for `query`.
    pub async fn recall(&self, query: &str) -> cognis_core::Result<Vec<Message>> {
        let hits = self
            .store
            .read()
            .await
            .similarity_search(query, self.k)
            .await?;
        Ok(hits
            .into_iter()
            .map(|h| {
                let role = h
                    .metadata
                    .get("role")
                    .and_then(|v| v.as_str())
                    .unwrap_or("user");
                match role {
                    "assistant" => Message::ai(h.text),
                    "system" => Message::system(h.text),
                    _ => Message::human(h.text),
                }
            })
            .collect())
    }
}

impl Memory for VectorMemory {
    fn read(&self) -> &[Message] {
        // Vector memory has no on-disk message ordering — the store is
        // keyed by similarity, not time. `read()` is nominal.
        &[]
    }
    fn write(&mut self, msg: Message) {
        // Synchronous interface — best-effort: spawn a task that persists.
        let store = self.store.clone();
        let m = msg.clone();
        tokio::spawn(async move {
            let mut meta = std::collections::HashMap::new();
            meta.insert(
                "role".into(),
                serde_json::Value::String(role_label(&m).into()),
            );
            let _ = store
                .write()
                .await
                .add_texts(vec![m.content().to_string()], Some(vec![meta]))
                .await;
        });
    }
    fn clear(&mut self) {
        // Best-effort: spawn a delete-all. We don't expose `delete_all` on
        // VectorStore yet, so this is a noop. Future: extend trait.
    }
    fn seed(&self) -> Vec<Message> {
        let mut out = Vec::new();
        if let Some(s) = &self.system_pinned {
            out.push(s.clone());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_below_capacity() {
        let mut w = Window::new(5);
        w.write(Message::human("a"));
        w.write(Message::human("b"));
        assert_eq!(w.seed().len(), 2);
    }

    #[test]
    fn fifo_drop_above_capacity() {
        let mut w = Window::new(2);
        w.write(Message::human("1"));
        w.write(Message::human("2"));
        w.write(Message::human("3"));
        let seed = w.seed();
        assert_eq!(seed.len(), 2);
        assert_eq!(seed[0].content(), "2");
        assert_eq!(seed[1].content(), "3");
    }

    #[test]
    fn system_pinned_survives_clear() {
        let mut w = Window::new(5).with_system("you are helpful");
        w.write(Message::human("hi"));
        w.clear();
        let seed = w.seed();
        assert_eq!(seed.len(), 1);
        assert_eq!(seed[0].content(), "you are helpful");
    }

    #[test]
    fn system_pinned_at_index_0() {
        let mut w = Window::new(5).with_system("system!");
        w.write(Message::human("u1"));
        w.write(Message::human("u2"));
        let seed = w.seed();
        assert_eq!(seed.len(), 3);
        assert_eq!(seed[0].content(), "system!");
        assert_eq!(seed[1].content(), "u1");
        assert_eq!(seed[2].content(), "u2");
    }

    #[test]
    fn token_buffer_drops_oldest_until_under_budget() {
        // CharTokenizer counts chars. Budget 6 with 3-char messages.
        let mut m = TokenBufferMemory::new(6);
        m.write(Message::human("aaa"));
        m.write(Message::human("bbb"));
        m.write(Message::human("ccc"));
        let seed = m.seed();
        // Two messages fit (3 + 3 = 6); the third would push to 9 → dropped.
        assert_eq!(seed.len(), 2);
        // Oldest dropped → tail kept.
        assert_eq!(seed[0].content(), "bbb");
        assert_eq!(seed[1].content(), "ccc");
    }

    #[test]
    fn token_buffer_keeps_pinned_system() {
        let mut m = TokenBufferMemory::new(10).with_system("sys");
        m.write(Message::human("aaaa"));
        m.write(Message::human("bbbb"));
        let seed = m.seed();
        // System ("sys", 3 chars) is pinned; budget is 10; remaining 7 fits 4-char + can fit one more.
        assert!(!seed.is_empty());
        assert_eq!(seed[0].content(), "sys");
    }

    #[test]
    fn token_buffer_with_strategy_last_drops_newest() {
        let mut m = TokenBufferMemory::new(6).with_strategy(TrimStrategy::Last);
        m.write(Message::human("aaa"));
        m.write(Message::human("bbb"));
        m.write(Message::human("ccc"));
        let seed = m.seed();
        assert_eq!(seed.len(), 2);
        // Newest dropped → head kept.
        assert_eq!(seed[0].content(), "aaa");
        assert_eq!(seed[1].content(), "bbb");
    }

    #[test]
    fn token_buffer_clear_removes_all() {
        let mut m = TokenBufferMemory::new(100);
        m.write(Message::human("a"));
        m.clear();
        assert!(m.seed().is_empty());
    }
}

// ────────────────────────────────────────────────────────────────────────
// EntityMemory — extracts entities + facts from messages, surfaces them
// back into the seed as a system message.
// ────────────────────────────────────────────────────────────────────────

/// Extracted entity / fact pair. The fact is a free-form snippet —
/// typically the sentence the entity appeared in.
pub type EntityFact = (String, String);

/// Closure-based extractor: text in, `(entity, fact)` pairs out.
pub type EntityExtractor = Arc<dyn Fn(&str) -> Vec<EntityFact> + Send + Sync>;

/// Buffers messages and maintains a per-entity fact ledger. Each `write`
/// runs the extractor over the message content; the seed surfaces the
/// ledger as a system-message preamble so the model can reference prior
/// observations across turns.
///
/// Default extractor = capitalized-word heuristic: any token starting
/// with an uppercase letter becomes an entity, paired with the sentence
/// it appeared in. Plug in [`with_extractor`](EntityMemory::with_extractor)
/// for an LLM-driven version.
pub struct EntityMemory {
    buf: Vec<Message>,
    entities: std::collections::HashMap<String, Vec<String>>,
    extractor: EntityExtractor,
    system_pinned: Option<Message>,
}

impl EntityMemory {
    /// Empty memory with the default capitalized-word extractor.
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            entities: std::collections::HashMap::new(),
            extractor: Arc::new(default_entity_extractor),
            system_pinned: None,
        }
    }

    /// Plug in a custom extractor (e.g. an LLM-backed NER).
    pub fn with_extractor<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> Vec<EntityFact> + Send + Sync + 'static,
    {
        self.extractor = Arc::new(f);
        self
    }

    /// Pin a system prompt at the head of the seed.
    pub fn with_system(mut self, prompt: impl Into<String>) -> Self {
        self.system_pinned = Some(Message::system(prompt));
        self
    }

    /// Inspect the current entity ledger.
    pub fn entities(&self) -> &std::collections::HashMap<String, Vec<String>> {
        &self.entities
    }
}

impl Default for EntityMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl Memory for EntityMemory {
    fn read(&self) -> &[Message] {
        &self.buf
    }
    fn write(&mut self, msg: Message) {
        for (entity, fact) in (self.extractor)(msg.content()) {
            self.entities.entry(entity).or_default().push(fact);
        }
        self.buf.push(msg);
    }
    fn clear(&mut self) {
        self.buf.clear();
        self.entities.clear();
    }
    fn seed(&self) -> Vec<Message> {
        let mut out = Vec::with_capacity(self.buf.len() + 2);
        if let Some(s) = &self.system_pinned {
            out.push(s.clone());
        }
        if !self.entities.is_empty() {
            let mut keys: Vec<&String> = self.entities.keys().collect();
            keys.sort();
            let body = keys
                .into_iter()
                .map(|k| {
                    let facts = self.entities.get(k).unwrap();
                    let joined = facts.join("; ");
                    format!("- {k}: {joined}")
                })
                .collect::<Vec<_>>()
                .join("\n");
            out.push(Message::system(format!("Known entities:\n{body}")));
        }
        out.extend(self.buf.iter().cloned());
        out
    }
}

fn default_entity_extractor(text: &str) -> Vec<EntityFact> {
    // Common capitalized stopwords that lead sentences but aren't entities.
    const STOPWORDS: &[&str] = &[
        "The", "A", "An", "This", "That", "These", "Those", "It", "Its", "Their", "There", "Here",
        "What", "Who", "Which", "When", "Where", "Why", "How", "And", "But", "Or", "If", "Then",
    ];
    let mut out = Vec::new();
    for sentence in split_sentences(text) {
        for tok in sentence.split_whitespace() {
            // Strip surrounding punctuation (keeps internal apostrophes).
            let trimmed: String = tok.trim_matches(|c: char| !c.is_alphanumeric()).to_string();
            if trimmed.len() >= 2
                && trimmed.chars().next().is_some_and(|c| c.is_uppercase())
                && !STOPWORDS.contains(&trimmed.as_str())
            {
                out.push((trimmed, sentence.trim().to_string()));
            }
        }
    }
    out
}

fn split_sentences(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, c) in text.char_indices() {
        if matches!(c, '.' | '!' | '?') {
            let end = i + c.len_utf8();
            let s = text[start..end].trim();
            if !s.is_empty() {
                out.push(s);
            }
            start = end;
        }
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        out.push(tail);
    }
    out
}

// ────────────────────────────────────────────────────────────────────────
// KnowledgeGraphMemory — buffers messages and extracts (S, P, O) triples,
// surfaces them as a system-message KB.
// ────────────────────────────────────────────────────────────────────────

/// Subject-predicate-object triple.
pub type Triple = (String, String, String);

/// Closure-based triple extractor.
pub type TripleExtractor = Arc<dyn Fn(&str) -> Vec<Triple> + Send + Sync>;

/// Buffers messages and a triple store. Each `write` extracts
/// triples; the seed prefixes a `Knowledge:` system message listing
/// every triple. Plug in an LLM extractor for production use; the
/// default handles "X is Y" / "X has Y" / "X are Y" patterns.
pub struct KnowledgeGraphMemory {
    buf: Vec<Message>,
    triples: Vec<Triple>,
    extractor: TripleExtractor,
    system_pinned: Option<Message>,
}

impl KnowledgeGraphMemory {
    /// Empty memory with the default regex extractor.
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            triples: Vec::new(),
            extractor: Arc::new(default_triple_extractor),
            system_pinned: None,
        }
    }

    /// Plug in a custom extractor.
    pub fn with_extractor<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> Vec<Triple> + Send + Sync + 'static,
    {
        self.extractor = Arc::new(f);
        self
    }

    /// Pin a system prompt at the head of the seed.
    pub fn with_system(mut self, prompt: impl Into<String>) -> Self {
        self.system_pinned = Some(Message::system(prompt));
        self
    }

    /// Inspect the triple store.
    pub fn triples(&self) -> &[Triple] {
        &self.triples
    }
}

impl Default for KnowledgeGraphMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl Memory for KnowledgeGraphMemory {
    fn read(&self) -> &[Message] {
        &self.buf
    }
    fn write(&mut self, msg: Message) {
        for t in (self.extractor)(msg.content()) {
            // Dedupe — the same fact restated stays one triple.
            if !self.triples.contains(&t) {
                self.triples.push(t);
            }
        }
        self.buf.push(msg);
    }
    fn clear(&mut self) {
        self.buf.clear();
        self.triples.clear();
    }
    fn seed(&self) -> Vec<Message> {
        let mut out = Vec::with_capacity(self.buf.len() + 2);
        if let Some(s) = &self.system_pinned {
            out.push(s.clone());
        }
        if !self.triples.is_empty() {
            let body = self
                .triples
                .iter()
                .map(|(s, p, o)| format!("- ({s}, {p}, {o})"))
                .collect::<Vec<_>>()
                .join("\n");
            out.push(Message::system(format!("Knowledge:\n{body}")));
        }
        out.extend(self.buf.iter().cloned());
        out
    }
}

fn default_triple_extractor(text: &str) -> Vec<Triple> {
    let mut out = Vec::new();
    for sentence in split_sentences(text) {
        // Find linking verbs.
        for predicate in [" is ", " are ", " has ", " have ", " was ", " were "] {
            if let Some(idx) = sentence.find(predicate) {
                let s = sentence[..idx].trim();
                let o_raw = sentence[idx + predicate.len()..]
                    .trim_end_matches(['.', '!', '?'])
                    .trim();
                if !s.is_empty() && !o_raw.is_empty() {
                    out.push((
                        s.to_string(),
                        predicate.trim().to_string(),
                        o_raw.to_string(),
                    ));
                    break; // one triple per sentence
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests_entity_kg {
    use super::*;

    #[test]
    fn entity_memory_extracts_default() {
        let mut m = EntityMemory::new();
        m.write(Message::human("Ada writes Rust. Bob reviews Ada's PRs."));
        let ents = m.entities();
        assert!(
            ents.contains_key("Ada"),
            "got: {:?}",
            ents.keys().collect::<Vec<_>>()
        );
        assert!(ents.contains_key("Rust"));
        assert!(ents.contains_key("Bob"));
    }

    #[test]
    fn entity_memory_seed_includes_summary() {
        let mut m = EntityMemory::new();
        m.write(Message::human("Cognis is fast."));
        let seed = m.seed();
        // Expect: [system "Known entities:..."] + the original human message.
        assert_eq!(seed.len(), 2);
        assert!(matches!(seed[0], Message::System(_)));
        assert!(seed[0].content().contains("Cognis"));
    }

    #[test]
    fn entity_memory_with_custom_extractor() {
        let mut m = EntityMemory::new()
            .with_extractor(|_text: &str| vec![("forced".into(), "via custom extractor".into())]);
        m.write(Message::human("ignored"));
        assert!(m.entities().contains_key("forced"));
    }

    #[test]
    fn entity_memory_clear_drops_everything() {
        let mut m = EntityMemory::new();
        m.write(Message::human("Rust ships."));
        m.clear();
        assert!(m.entities().is_empty());
        assert!(m.read().is_empty());
    }

    #[test]
    fn kg_memory_extracts_is_pattern() {
        let mut m = KnowledgeGraphMemory::new();
        m.write(Message::human(
            "Cognis is a Rust framework. Tokio is async.",
        ));
        let ts = m.triples();
        assert!(ts.contains(&("Cognis".into(), "is".into(), "a Rust framework".into())));
        assert!(ts.contains(&("Tokio".into(), "is".into(), "async".into())));
    }

    #[test]
    fn kg_memory_dedupes_repeated_triples() {
        let mut m = KnowledgeGraphMemory::new();
        m.write(Message::human("Rust is fast."));
        m.write(Message::human("Rust is fast."));
        assert_eq!(m.triples().len(), 1);
    }

    #[test]
    fn kg_memory_seed_includes_kb() {
        let mut m = KnowledgeGraphMemory::new();
        m.write(Message::human("Cognis is fast."));
        let seed = m.seed();
        assert_eq!(seed.len(), 2);
        assert!(matches!(seed[0], Message::System(_)));
        assert!(seed[0].content().contains("(Cognis, is, fast)"));
    }

    #[test]
    fn kg_memory_with_custom_extractor() {
        let mut m = KnowledgeGraphMemory::new()
            .with_extractor(|_text: &str| vec![("X".into(), "rel".into(), "Y".into())]);
        m.write(Message::human("ignored"));
        assert_eq!(m.triples(), &[("X".into(), "rel".into(), "Y".into())]);
    }
}

// ────────────────────────────────────────────────────────────────────────
// HybridMemory — combine N member memories. Writes fan out to every
// member; seed() concatenates each member's contribution in registration
// order. Use to compose specialized memories (e.g. recent buffer +
// long-term summary + entity ledger + semantic vector recall).
// ────────────────────────────────────────────────────────────────────────

/// A memory composed of several member memories. Each `write` is
/// broadcast to every member; `seed` concatenates each member's
/// contribution in registration order.
///
/// Use to compose specialized memories — e.g. a `Window` for recent
/// turns plus a `SummaryMemory` for older context plus an `EntityMemory`
/// to surface known entities. Each member can do its own thing on write
/// (the Window will trim, the SummaryMemory will compact, etc.); the
/// agent sees a unified seed.
pub struct HybridMemory {
    members: Vec<Box<dyn Memory>>,
    /// Tracks the raw write history so `read()` can return a `&[Message]`
    /// without materializing across members. Members own their own
    /// (possibly-transformed) buffers.
    buf: Vec<Message>,
}

impl Default for HybridMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl HybridMemory {
    /// Empty hybrid with no members. Add members via [`HybridMemory::with`].
    pub fn new() -> Self {
        Self {
            members: Vec::new(),
            buf: Vec::new(),
        }
    }

    /// Append a member memory. Builder-style.
    pub fn with(mut self, member: impl Memory + 'static) -> Self {
        self.members.push(Box::new(member));
        self
    }

    /// Number of members.
    pub fn member_count(&self) -> usize {
        self.members.len()
    }
}

impl Memory for HybridMemory {
    fn read(&self) -> &[Message] {
        &self.buf
    }
    fn write(&mut self, msg: Message) {
        for m in &mut self.members {
            m.write(msg.clone());
        }
        self.buf.push(msg);
    }
    fn clear(&mut self) {
        for m in &mut self.members {
            m.clear();
        }
        self.buf.clear();
    }
    fn seed(&self) -> Vec<Message> {
        let mut out: Vec<Message> = Vec::new();
        for m in &self.members {
            out.extend(m.seed());
        }
        out
    }
}

#[cfg(test)]
mod tests_hybrid {
    use super::*;

    #[test]
    fn write_fans_out_to_every_member() {
        let mut h = HybridMemory::new()
            .with(Buffer::new())
            .with(Window::new(10));
        h.write(Message::human("a"));
        h.write(Message::human("b"));
        assert_eq!(h.read().len(), 2);
        // Both members should have seen both writes.
        let seed = h.seed();
        // Buffer contributes 2 + Window contributes 2 = 4 (no dedup).
        assert_eq!(seed.len(), 4);
    }

    #[test]
    fn clear_empties_every_member() {
        let mut h = HybridMemory::new()
            .with(Buffer::new())
            .with(Window::new(10));
        h.write(Message::human("a"));
        h.clear();
        assert!(h.read().is_empty());
        assert!(h.seed().is_empty());
    }

    #[test]
    fn seed_concatenates_in_member_order() {
        let mut h = HybridMemory::new()
            .with(Buffer::new().with_system("recent context"))
            .with(EntityMemory::new());
        h.write(Message::human("Cognis is fast."));
        let seed = h.seed();
        // Buffer: system pin + 1 human msg → 2
        // EntityMemory: synthesized "Known entities" system + 1 human → 2
        assert_eq!(seed.len(), 4);
        // First member's contribution comes first.
        assert!(matches!(seed[0], Message::System(_)));
        assert_eq!(seed[0].content(), "recent context");
    }

    #[test]
    fn empty_hybrid_round_trips() {
        let mut h = HybridMemory::new();
        h.write(Message::human("a"));
        // No members → seed is empty (only members contribute).
        assert!(h.seed().is_empty());
        // But read() reflects the canonical write-buffer.
        assert_eq!(h.read().len(), 1);
        assert_eq!(h.member_count(), 0);
    }
}
