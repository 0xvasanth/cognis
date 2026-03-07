//! Entity memory that tracks entities mentioned in conversation.
//!
//! [`EntityMemory`] extracts named entities from conversation text using
//! regex-based heuristics and maintains a store of entity descriptions.
//! This allows chains to include relevant entity context in prompts.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;
use tokio::sync::RwLock;

use rustchain_core::error::Result;
use rustchain_core::messages::{get_buffer_string, Message};

use super::BaseMemory;

/// A tracked entity with description and usage metadata.
#[derive(Debug, Clone)]
pub struct Entity {
    /// The entity name (e.g., "Alice", "OpenAI").
    pub name: String,
    /// A description of the entity, updated as new information appears.
    pub description: String,
    /// The message index at which the entity was last seen.
    pub last_seen: usize,
    /// How many times the entity has been mentioned.
    pub mentions: usize,
    /// Arbitrary metadata associated with the entity.
    pub metadata: HashMap<String, Value>,
}

impl Entity {
    /// Create a new entity with the given name and initial description.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            last_seen: 0,
            mentions: 1,
            metadata: HashMap::new(),
        }
    }
}

/// Trait for entity persistence backends.
pub trait EntityStore: Send + Sync {
    /// Retrieve an entity by name.
    fn get(&self, name: &str) -> Option<&Entity>;

    /// Insert or update an entity.
    fn set(&mut self, entity: Entity);

    /// Delete an entity by name.
    fn delete(&mut self, name: &str);

    /// List all stored entities.
    fn list(&self) -> Vec<&Entity>;

    /// Remove all entities.
    fn clear(&mut self);
}

/// In-memory [`EntityStore`] backed by a `HashMap`.
#[derive(Debug, Default)]
pub struct InMemoryEntityStore {
    entities: HashMap<String, Entity>,
}

impl InMemoryEntityStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

impl EntityStore for InMemoryEntityStore {
    fn get(&self, name: &str) -> Option<&Entity> {
        self.entities.get(name)
    }

    fn set(&mut self, entity: Entity) {
        self.entities.insert(entity.name.clone(), entity);
    }

    fn delete(&mut self, name: &str) {
        self.entities.remove(name);
    }

    fn list(&self) -> Vec<&Entity> {
        self.entities.values().collect()
    }

    fn clear(&mut self) {
        self.entities.clear();
    }
}

/// Memory that extracts and tracks entities mentioned in conversation.
///
/// Uses regex-based extraction to identify capitalised words and phrases
/// (likely proper nouns / entity names) and stores them in an [`EntityStore`].
/// When loading memory variables, the entity summaries relevant to the
/// current input are included alongside the conversation history.
pub struct EntityMemory {
    inner: Arc<RwLock<EntityMemoryInner>>,
    memory_key: String,
    entity_key: String,
}

struct EntityMemoryInner {
    store: Box<dyn EntityStore>,
    messages: Vec<Message>,
    message_index: usize,
}

impl EntityMemory {
    /// Create a new entity memory with the default in-memory store.
    pub fn new() -> Self {
        Self::with_store(Box::new(InMemoryEntityStore::new()))
    }

    /// Create a new entity memory with a custom entity store.
    pub fn with_store(store: Box<dyn EntityStore>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(EntityMemoryInner {
                store,
                messages: Vec::new(),
                message_index: 0,
            })),
            memory_key: "history".to_string(),
            entity_key: "entities".to_string(),
        }
    }

    /// Set the memory key used for conversation history in chain context.
    pub fn with_memory_key(mut self, key: impl Into<String>) -> Self {
        self.memory_key = key.into();
        self
    }

    /// Set the key used for entity context in chain variables.
    pub fn with_entity_key(mut self, key: impl Into<String>) -> Self {
        self.entity_key = key.into();
        self
    }

    /// Extract likely entity names from text using regex heuristics.
    ///
    /// Matches capitalised words and multi-word capitalised phrases that
    /// are likely proper nouns. Single-letter words and common sentence
    /// starters are filtered out.
    pub fn extract_entities(text: &str) -> Vec<String> {
        // Match sequences of capitalized words (e.g., "John Smith", "New York City")
        let re = Regex::new(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)").unwrap();
        let mut entities = Vec::new();
        let mut seen = std::collections::HashSet::new();

        // Common words that start sentences but are not entities
        let stop_words: std::collections::HashSet<&str> = [
            "The", "This", "That", "These", "Those", "Here", "There", "Where", "When", "What",
            "Which", "Who", "How", "Why", "Yes", "No", "Not", "But", "And", "Or", "If", "So",
            "Just", "Also", "However", "Indeed", "Nice", "Great",
        ]
        .iter()
        .copied()
        .collect();

        for cap in re.captures_iter(text) {
            let name = cap[1].to_string();
            // Skip very short names
            if name.len() <= 1 {
                continue;
            }
            // Skip common stop words (only for single-word matches)
            if !name.contains(' ') && stop_words.contains(name.as_str()) {
                continue;
            }

            let lower = name.to_lowercase();
            if !seen.contains(&lower) {
                seen.insert(lower);
                entities.push(name);
            }
        }

        entities
    }

    /// Update entity records from the given text at the specified message index.
    pub async fn update_entities(&self, text: &str, message_index: usize) {
        let entities = Self::extract_entities(text);
        let mut inner = self.inner.write().await;

        for name in entities {
            if let Some(existing) = inner.store.get(&name) {
                let mut updated = existing.clone();
                updated.last_seen = message_index;
                updated.mentions += 1;
                inner.store.set(updated);
            } else {
                let mut entity = Entity::new(&name, "Mentioned in conversation.".to_string());
                entity.last_seen = message_index;
                inner.store.set(entity);
            }
        }
    }

    /// Format a summary string for the given entity names.
    pub async fn get_entity_summary(&self, entity_names: &[String]) -> String {
        let inner = self.inner.read().await;
        let mut parts = Vec::new();

        for name in entity_names {
            if let Some(entity) = inner.store.get(name) {
                parts.push(format!(
                    "{}: {} (mentioned {} time{})",
                    entity.name,
                    entity.description,
                    entity.mentions,
                    if entity.mentions == 1 { "" } else { "s" }
                ));
            }
        }

        if parts.is_empty() {
            "No known entities.".to_string()
        } else {
            parts.join("\n")
        }
    }

    /// Extract entities from the input and return their summaries as context.
    pub async fn get_context(&self, input: &str) -> String {
        let entities = Self::extract_entities(input);
        self.get_entity_summary(&entities).await
    }

    /// Get the number of tracked entities.
    pub async fn entity_count(&self) -> usize {
        let inner = self.inner.read().await;
        inner.store.list().len()
    }

    /// Get all tracked entity names.
    pub async fn entity_names(&self) -> Vec<String> {
        let inner = self.inner.read().await;
        inner.store.list().iter().map(|e| e.name.clone()).collect()
    }
}

impl Default for EntityMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BaseMemory for EntityMemory {
    async fn load_memory_variables(&self) -> Result<HashMap<String, Value>> {
        let inner = self.inner.read().await;
        let mut vars = HashMap::new();

        // Return conversation history as formatted string
        let buffer = get_buffer_string(&inner.messages, "Human", "AI");
        vars.insert(self.memory_key.clone(), Value::String(buffer));

        // Return entity information
        let entity_summaries: Vec<String> = inner
            .store
            .list()
            .iter()
            .map(|e| {
                format!(
                    "{}: {} (mentioned {} time{})",
                    e.name,
                    e.description,
                    e.mentions,
                    if e.mentions == 1 { "" } else { "s" }
                )
            })
            .collect();

        let entity_text = if entity_summaries.is_empty() {
            "No known entities.".to_string()
        } else {
            entity_summaries.join("\n")
        };
        vars.insert(self.entity_key.clone(), Value::String(entity_text));

        Ok(vars)
    }

    async fn save_context(&self, input: &Message, output: &Message) -> Result<()> {
        let message_index = {
            let inner = self.inner.read().await;
            inner.message_index
        };

        // Extract entities from both input and output
        let input_text = input.content().text();
        let output_text = output.content().text();

        self.update_entities(&input_text, message_index).await;
        self.update_entities(&output_text, message_index + 1).await;

        // Store messages
        {
            let mut inner = self.inner.write().await;
            inner.messages.push(input.clone());
            inner.messages.push(output.clone());
            inner.message_index += 2;
        }

        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        let mut inner = self.inner.write().await;
        inner.messages.clear();
        inner.store.clear();
        inner.message_index = 0;
        Ok(())
    }

    fn memory_key(&self) -> &str {
        &self.memory_key
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustchain_core::messages::Message;

    // ─── Entity struct tests ───

    #[test]
    fn test_entity_new() {
        let entity = Entity::new("Alice", "A software engineer");
        assert_eq!(entity.name, "Alice");
        assert_eq!(entity.description, "A software engineer");
        assert_eq!(entity.last_seen, 0);
        assert_eq!(entity.mentions, 1);
        assert!(entity.metadata.is_empty());
    }

    #[test]
    fn test_entity_with_metadata() {
        let mut entity = Entity::new("Bob", "A manager");
        entity
            .metadata
            .insert("role".to_string(), Value::String("CTO".to_string()));
        assert_eq!(entity.metadata.get("role").unwrap(), "CTO");
    }

    // ─── InMemoryEntityStore tests ───

    #[test]
    fn test_store_set_and_get() {
        let mut store = InMemoryEntityStore::new();
        let entity = Entity::new("Alice", "Engineer");
        store.set(entity);
        let retrieved = store.get("Alice").unwrap();
        assert_eq!(retrieved.name, "Alice");
        assert_eq!(retrieved.description, "Engineer");
    }

    #[test]
    fn test_store_get_nonexistent() {
        let store = InMemoryEntityStore::new();
        assert!(store.get("Nobody").is_none());
    }

    #[test]
    fn test_store_delete() {
        let mut store = InMemoryEntityStore::new();
        store.set(Entity::new("Alice", "Engineer"));
        assert!(store.get("Alice").is_some());
        store.delete("Alice");
        assert!(store.get("Alice").is_none());
    }

    #[test]
    fn test_store_list() {
        let mut store = InMemoryEntityStore::new();
        store.set(Entity::new("Alice", "Engineer"));
        store.set(Entity::new("Bob", "Manager"));
        let list = store.list();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_store_clear() {
        let mut store = InMemoryEntityStore::new();
        store.set(Entity::new("Alice", "Engineer"));
        store.set(Entity::new("Bob", "Manager"));
        store.clear();
        assert!(store.list().is_empty());
    }

    #[test]
    fn test_store_update_existing() {
        let mut store = InMemoryEntityStore::new();
        store.set(Entity::new("Alice", "Engineer"));
        let mut updated = Entity::new("Alice", "Senior Engineer");
        updated.mentions = 5;
        store.set(updated);
        let e = store.get("Alice").unwrap();
        assert_eq!(e.description, "Senior Engineer");
        assert_eq!(e.mentions, 5);
    }

    // ─── Entity extraction tests ───

    #[test]
    fn test_extract_single_entity() {
        let entities = EntityMemory::extract_entities("I talked to Alice about the project.");
        assert!(entities.contains(&"Alice".to_string()));
    }

    #[test]
    fn test_extract_multi_word_entity() {
        let entities = EntityMemory::extract_entities("I work at New York City headquarters.");
        assert!(entities.contains(&"New York City".to_string()));
    }

    #[test]
    fn test_extract_multiple_entities() {
        let entities = EntityMemory::extract_entities("Alice met Bob at the conference.");
        assert!(entities.contains(&"Alice".to_string()));
        assert!(entities.contains(&"Bob".to_string()));
    }

    #[test]
    fn test_extract_no_entities() {
        let entities = EntityMemory::extract_entities("hello world, no entities here.");
        assert!(entities.is_empty());
    }

    #[test]
    fn test_extract_deduplication() {
        let entities = EntityMemory::extract_entities("Alice went to the store. Alice came back.");
        let alice_count = entities.iter().filter(|e| *e == "Alice").count();
        assert_eq!(alice_count, 1);
    }

    // ─── EntityMemory integration tests ───

    #[tokio::test]
    async fn test_entity_memory_save_and_load() {
        let mem = EntityMemory::new();

        let human = Message::human("I talked to Alice about Rust programming.");
        let ai = Message::ai("Alice is an expert in Rust.");
        mem.save_context(&human, &ai).await.unwrap();

        let vars = mem.load_memory_variables().await.unwrap();
        assert!(vars.contains_key("history"));
        assert!(vars.contains_key("entities"));

        let entities_text = vars.get("entities").unwrap().as_str().unwrap();
        assert!(entities_text.contains("Alice"));
        assert!(entities_text.contains("Rust"));
    }

    #[tokio::test]
    async fn test_entity_memory_mention_counting() {
        let mem = EntityMemory::new();

        mem.save_context(
            &Message::human("Alice is great."),
            &Message::ai("Indeed, Alice is wonderful."),
        )
        .await
        .unwrap();

        // Alice should have 2 mentions (once in input, once in output)
        let vars = mem.load_memory_variables().await.unwrap();
        let entities_text = vars.get("entities").unwrap().as_str().unwrap();
        assert!(entities_text.contains("2 times"));
    }

    #[tokio::test]
    async fn test_entity_memory_get_context() {
        let mem = EntityMemory::new();

        mem.save_context(
            &Message::human("Bob works at Google."),
            &Message::ai("Yes, Bob is a Google engineer."),
        )
        .await
        .unwrap();

        let context = mem.get_context("Tell me about Bob.").await;
        assert!(context.contains("Bob"));
    }

    #[tokio::test]
    async fn test_entity_memory_clear() {
        let mem = EntityMemory::new();

        mem.save_context(
            &Message::human("Alice likes Rust."),
            &Message::ai("Great choice!"),
        )
        .await
        .unwrap();

        mem.clear().await.unwrap();

        assert_eq!(mem.entity_count().await, 0);
        let vars = mem.load_memory_variables().await.unwrap();
        let history = vars.get("history").unwrap().as_str().unwrap();
        assert!(history.is_empty());
    }

    #[tokio::test]
    async fn test_entity_memory_custom_keys() {
        let mem = EntityMemory::new()
            .with_memory_key("chat")
            .with_entity_key("known_entities");

        mem.save_context(
            &Message::human("Alice said hello."),
            &Message::ai("Hi Alice!"),
        )
        .await
        .unwrap();

        let vars = mem.load_memory_variables().await.unwrap();
        assert!(vars.contains_key("chat"));
        assert!(vars.contains_key("known_entities"));
        assert!(!vars.contains_key("history"));
        assert!(!vars.contains_key("entities"));
    }

    #[tokio::test]
    async fn test_entity_memory_entity_count() {
        let mem = EntityMemory::new();
        assert_eq!(mem.entity_count().await, 0);

        mem.save_context(
            &Message::human("Alice and Bob met."),
            &Message::ai("Nice meeting!"),
        )
        .await
        .unwrap();

        assert!(mem.entity_count().await >= 2);
    }

    #[tokio::test]
    async fn test_entity_memory_no_entities_context() {
        let mem = EntityMemory::new();
        let context = mem.get_context("hello world").await;
        assert_eq!(context, "No known entities.");
    }
}
