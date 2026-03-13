//! Knowledge graph memory that extracts and stores knowledge triples from conversation.
//!
//! [`KnowledgeGraphMemory`] extracts subject-predicate-object triples from conversation
//! text and maintains a knowledge graph. This allows chains to include relevant
//! relational knowledge in prompts.
//!
//! - [`KnowledgeTriple`] — a subject-predicate-object triple with confidence and source metadata
//! - [`KnowledgeGraph`] — an in-memory graph storing and querying triples
//! - [`KnowledgeGraphMemory`] — a [`BaseMemory`] implementation that automatically extracts triples

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::RwLock;

use cognis_core::error::Result;
use cognis_core::messages::{get_buffer_string, Message};

use super::BaseMemory;

/// A knowledge triple representing a relationship between two entities.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KnowledgeTriple {
    /// The subject of the triple (e.g., "Alice").
    pub subject: String,
    /// The predicate/relationship (e.g., "works at").
    pub predicate: String,
    /// The object of the triple (e.g., "Google").
    pub object: String,
    /// Confidence score for this triple (0.0 to 1.0).
    pub confidence: f64,
    /// Source message the triple was extracted from.
    pub source: Option<String>,
}

impl KnowledgeTriple {
    /// Create a new knowledge triple with default confidence of 1.0.
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
            confidence: 1.0,
            source: None,
        }
    }

    /// Set the confidence score for this triple.
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    /// Set the source message for this triple.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Convert the triple to a natural language sentence.
    pub fn to_natural_language(&self) -> String {
        format!("{} {} {}", self.subject, self.predicate, self.object)
    }
}

/// A graph structure that stores knowledge triples.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KnowledgeGraph {
    /// All triples in the graph.
    triples: Vec<KnowledgeTriple>,
}

impl KnowledgeGraph {
    /// Create an empty knowledge graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a triple to the graph.
    pub fn add_triple(&mut self, triple: KnowledgeTriple) {
        self.triples.push(triple);
    }

    /// Get all triples where the given entity appears as subject or object.
    pub fn get_triples_for_entity(&self, entity: &str) -> Vec<&KnowledgeTriple> {
        let entity_lower = entity.to_lowercase();
        self.triples
            .iter()
            .filter(|t| {
                t.subject.to_lowercase() == entity_lower || t.object.to_lowercase() == entity_lower
            })
            .collect()
    }

    /// Get all entities connected to the given entity through any relationship.
    pub fn get_related_entities(&self, entity: &str) -> Vec<String> {
        let entity_lower = entity.to_lowercase();
        let mut related = std::collections::HashSet::new();

        for triple in &self.triples {
            if triple.subject.to_lowercase() == entity_lower {
                related.insert(triple.object.clone());
            }
            if triple.object.to_lowercase() == entity_lower {
                related.insert(triple.subject.clone());
            }
        }

        related.into_iter().collect()
    }

    /// Search triples by fuzzy matching across all fields (subject, predicate, object).
    pub fn search_triples(&self, query: &str) -> Vec<&KnowledgeTriple> {
        let query_lower = query.to_lowercase();
        self.triples
            .iter()
            .filter(|t| {
                t.subject.to_lowercase().contains(&query_lower)
                    || t.predicate.to_lowercase().contains(&query_lower)
                    || t.object.to_lowercase().contains(&query_lower)
            })
            .collect()
    }

    /// Remove all triples where the given entity appears as subject or object.
    pub fn remove_triples_for_entity(&mut self, entity: &str) {
        let entity_lower = entity.to_lowercase();
        self.triples.retain(|t| {
            t.subject.to_lowercase() != entity_lower && t.object.to_lowercase() != entity_lower
        });
    }

    /// Merge another knowledge graph into this one, deduplicating triples.
    pub fn merge(&mut self, other: &KnowledgeGraph) {
        for triple in &other.triples {
            let is_duplicate = self.triples.iter().any(|t| {
                t.subject.to_lowercase() == triple.subject.to_lowercase()
                    && t.predicate.to_lowercase() == triple.predicate.to_lowercase()
                    && t.object.to_lowercase() == triple.object.to_lowercase()
            });
            if !is_duplicate {
                self.triples.push(triple.clone());
            }
        }
    }

    /// Convert all triples to natural language sentences.
    pub fn to_natural_language(&self) -> String {
        if self.triples.is_empty() {
            return String::new();
        }
        self.triples
            .iter()
            .map(|t| t.to_natural_language())
            .collect::<Vec<_>>()
            .join(". ")
    }

    /// Return the number of triples in the graph.
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Check if the graph has no triples.
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Remove all triples from the graph.
    pub fn clear(&mut self) {
        self.triples.clear();
    }

    /// Get a reference to all triples.
    pub fn triples(&self) -> &[KnowledgeTriple] {
        &self.triples
    }
}

/// Trait for extracting knowledge triples from text.
pub trait TripleExtractor: Send + Sync {
    /// Extract knowledge triples from the given text.
    fn extract_triples(&self, text: &str) -> Vec<KnowledgeTriple>;
}

/// A pattern used by [`RegexTripleExtractor`] to match triples.
#[derive(Debug, Clone)]
pub struct ExtractionPattern {
    /// The compiled regex pattern.
    pub regex: Regex,
    /// The predicate to assign when this pattern matches.
    pub predicate: String,
}

/// Rule-based triple extractor using configurable regex patterns.
///
/// Ships with default patterns for common relationships like "X is Y",
/// "X has Y", "X works at Y", "X created Y", etc.
pub struct RegexTripleExtractor {
    patterns: Vec<ExtractionPattern>,
}

impl RegexTripleExtractor {
    /// Create a new extractor with default patterns.
    pub fn new() -> Self {
        Self {
            patterns: Self::default_patterns(),
        }
    }

    /// Create an extractor with only the specified patterns.
    pub fn with_patterns(patterns: Vec<ExtractionPattern>) -> Self {
        Self { patterns }
    }

    /// Add a custom extraction pattern.
    pub fn add_pattern(&mut self, regex: Regex, predicate: impl Into<String>) {
        self.patterns.push(ExtractionPattern {
            regex,
            predicate: predicate.into(),
        });
    }

    /// Return the default set of extraction patterns.
    fn default_patterns() -> Vec<ExtractionPattern> {
        let patterns = vec![
            // "X is a/an Y" or "X is Y"
            (
                r"(?i)\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s+is\s+(?:a|an)\s+(.+?)(?:\.|,|;|$)",
                "is a",
            ),
            (
                r"(?i)\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s+is\s+(.+?)(?:\.|,|;|$)",
                "is",
            ),
            // "X has Y"
            (
                r"(?i)\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s+has\s+(.+?)(?:\.|,|;|$)",
                "has",
            ),
            // "X works at Y"
            (
                r"(?i)\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s+works?\s+at\s+(.+?)(?:\.|,|;|$)",
                "works at",
            ),
            // "X lives in Y"
            (
                r"(?i)\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s+lives?\s+in\s+(.+?)(?:\.|,|;|$)",
                "lives in",
            ),
            // "X created Y"
            (
                r"(?i)\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s+created\s+(.+?)(?:\.|,|;|$)",
                "created",
            ),
            // "X loves Y"
            (
                r"(?i)\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s+loves?\s+(.+?)(?:\.|,|;|$)",
                "loves",
            ),
            // "X knows Y"
            (
                r"(?i)\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s+knows?\s+(.+?)(?:\.|,|;|$)",
                "knows",
            ),
            // "X built Y"
            (
                r"(?i)\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s+built\s+(.+?)(?:\.|,|;|$)",
                "built",
            ),
            // "X manages Y"
            (
                r"(?i)\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s+manages?\s+(.+?)(?:\.|,|;|$)",
                "manages",
            ),
            // "X leads Y"
            (
                r"(?i)\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s+leads?\s+(.+?)(?:\.|,|;|$)",
                "leads",
            ),
            // "X owns Y"
            (
                r"(?i)\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s+owns?\s+(.+?)(?:\.|,|;|$)",
                "owns",
            ),
            // "X teaches Y"
            (
                r"(?i)\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s+teaches?\s+(.+?)(?:\.|,|;|$)",
                "teaches",
            ),
            // "X studies Y"
            (
                r"(?i)\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s+studies\s+(.+?)(?:\.|,|;|$)",
                "studies",
            ),
        ];

        patterns
            .into_iter()
            .map(|(pat, pred)| ExtractionPattern {
                regex: Regex::new(pat).expect("invalid default pattern"),
                predicate: pred.to_string(),
            })
            .collect()
    }

    /// Return the number of configured patterns.
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }
}

impl Default for RegexTripleExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl TripleExtractor for RegexTripleExtractor {
    fn extract_triples(&self, text: &str) -> Vec<KnowledgeTriple> {
        let mut triples = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for pattern in &self.patterns {
            for caps in pattern.regex.captures_iter(text) {
                let subject = caps.get(1).map(|m| m.as_str().trim().to_string());
                let object = caps.get(2).map(|m| m.as_str().trim().to_string());

                if let (Some(subj), Some(obj)) = (subject, object) {
                    if subj.is_empty() || obj.is_empty() {
                        continue;
                    }
                    let key = (
                        subj.to_lowercase(),
                        pattern.predicate.to_lowercase(),
                        obj.to_lowercase(),
                    );
                    if seen.contains(&key) {
                        continue;
                    }
                    seen.insert(key);
                    triples.push(
                        KnowledgeTriple::new(&subj, &pattern.predicate, &obj)
                            .with_source(text.to_string()),
                    );
                }
            }
        }

        triples
    }
}

/// Memory that extracts knowledge triples from conversation and stores them
/// in a knowledge graph.
///
/// Uses a [`TripleExtractor`] to parse text and maintains a [`KnowledgeGraph`]
/// of relationships. When loading memory variables, relevant knowledge
/// is included alongside conversation history.
pub struct KnowledgeGraphMemory {
    inner: Arc<RwLock<KnowledgeGraphMemoryInner>>,
    memory_key: String,
    knowledge_key: String,
}

struct KnowledgeGraphMemoryInner {
    graph: KnowledgeGraph,
    extractor: Box<dyn TripleExtractor>,
    messages: Vec<Message>,
}

/// Builder for constructing a [`KnowledgeGraphMemory`] with custom configuration.
pub struct KnowledgeGraphMemoryBuilder {
    extractor: Option<Box<dyn TripleExtractor>>,
    memory_key: String,
    knowledge_key: String,
    initial_triples: Vec<KnowledgeTriple>,
}

impl KnowledgeGraphMemoryBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            extractor: None,
            memory_key: "history".to_string(),
            knowledge_key: "knowledge".to_string(),
            initial_triples: Vec::new(),
        }
    }

    /// Set a custom triple extractor.
    pub fn extractor(mut self, extractor: Box<dyn TripleExtractor>) -> Self {
        self.extractor = Some(extractor);
        self
    }

    /// Set the memory key for conversation history.
    pub fn memory_key(mut self, key: impl Into<String>) -> Self {
        self.memory_key = key.into();
        self
    }

    /// Set the key for knowledge context.
    pub fn knowledge_key(mut self, key: impl Into<String>) -> Self {
        self.knowledge_key = key.into();
        self
    }

    /// Add initial triples to the knowledge graph.
    pub fn initial_triples(mut self, triples: Vec<KnowledgeTriple>) -> Self {
        self.initial_triples = triples;
        self
    }

    /// Build the [`KnowledgeGraphMemory`].
    pub fn build(self) -> KnowledgeGraphMemory {
        let extractor = self
            .extractor
            .unwrap_or_else(|| Box::new(RegexTripleExtractor::new()));

        let mut graph = KnowledgeGraph::new();
        for triple in self.initial_triples {
            graph.add_triple(triple);
        }

        KnowledgeGraphMemory {
            inner: Arc::new(RwLock::new(KnowledgeGraphMemoryInner {
                graph,
                extractor,
                messages: Vec::new(),
            })),
            memory_key: self.memory_key,
            knowledge_key: self.knowledge_key,
        }
    }
}

impl Default for KnowledgeGraphMemoryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl KnowledgeGraphMemory {
    /// Create a new knowledge graph memory with the default regex extractor.
    pub fn new() -> Self {
        KnowledgeGraphMemoryBuilder::new().build()
    }

    /// Create a builder for configuring the memory.
    pub fn builder() -> KnowledgeGraphMemoryBuilder {
        KnowledgeGraphMemoryBuilder::new()
    }

    /// Get formatted knowledge relevant to the given entities.
    pub async fn get_knowledge_for(&self, entities: &[String]) -> String {
        let inner = self.inner.read().await;
        let mut relevant_triples = Vec::new();

        for entity in entities {
            for triple in inner.graph.get_triples_for_entity(entity) {
                if !relevant_triples
                    .iter()
                    .any(|t: &&KnowledgeTriple| std::ptr::eq(*t, triple))
                {
                    relevant_triples.push(triple);
                }
            }
        }

        if relevant_triples.is_empty() {
            "No relevant knowledge.".to_string()
        } else {
            relevant_triples
                .iter()
                .map(|t| t.to_natural_language())
                .collect::<Vec<_>>()
                .join("\n")
        }
    }

    /// Get a snapshot of the current knowledge graph.
    pub async fn graph_snapshot(&self) -> KnowledgeGraph {
        let inner = self.inner.read().await;
        inner.graph.clone()
    }

    /// Get the number of triples in the knowledge graph.
    pub async fn triple_count(&self) -> usize {
        let inner = self.inner.read().await;
        inner.graph.len()
    }
}

impl Default for KnowledgeGraphMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BaseMemory for KnowledgeGraphMemory {
    async fn load_memory_variables(&self) -> Result<HashMap<String, Value>> {
        let inner = self.inner.read().await;
        let mut vars = HashMap::new();

        // Return conversation history
        let buffer = get_buffer_string(&inner.messages, "Human", "AI");
        vars.insert(self.memory_key.clone(), Value::String(buffer));

        // Return knowledge graph as natural language
        let knowledge = if inner.graph.is_empty() {
            "No knowledge extracted yet.".to_string()
        } else {
            inner.graph.to_natural_language()
        };
        vars.insert(self.knowledge_key.clone(), Value::String(knowledge));

        Ok(vars)
    }

    async fn save_context(&self, input: &Message, output: &Message) -> Result<()> {
        let input_text = input.content().text();
        let output_text = output.content().text();

        {
            let mut inner = self.inner.write().await;

            // Extract triples from both messages
            let input_triples = inner.extractor.extract_triples(&input_text);
            let output_triples = inner.extractor.extract_triples(&output_text);

            for triple in input_triples {
                inner.graph.add_triple(triple);
            }
            for triple in output_triples {
                inner.graph.add_triple(triple);
            }

            // Store messages
            inner.messages.push(input.clone());
            inner.messages.push(output.clone());
        }

        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        let mut inner = self.inner.write().await;
        inner.messages.clear();
        inner.graph.clear();
        Ok(())
    }

    fn memory_key(&self) -> &str {
        &self.memory_key
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cognis_core::messages::Message;

    // ─── KnowledgeTriple tests ───

    #[test]
    fn test_triple_new() {
        let triple = KnowledgeTriple::new("Alice", "works at", "Google");
        assert_eq!(triple.subject, "Alice");
        assert_eq!(triple.predicate, "works at");
        assert_eq!(triple.object, "Google");
        assert_eq!(triple.confidence, 1.0);
        assert!(triple.source.is_none());
    }

    #[test]
    fn test_triple_with_confidence() {
        let triple = KnowledgeTriple::new("Alice", "knows", "Bob").with_confidence(0.8);
        assert_eq!(triple.confidence, 0.8);
    }

    #[test]
    fn test_triple_with_source() {
        let triple =
            KnowledgeTriple::new("Alice", "is", "engineer").with_source("Alice is an engineer.");
        assert_eq!(triple.source.unwrap(), "Alice is an engineer.");
    }

    #[test]
    fn test_triple_to_natural_language() {
        let triple = KnowledgeTriple::new("Alice", "works at", "Google");
        assert_eq!(triple.to_natural_language(), "Alice works at Google");
    }

    #[test]
    fn test_triple_serialization() {
        let triple = KnowledgeTriple::new("Alice", "is", "engineer").with_confidence(0.9);
        let json = serde_json::to_string(&triple).unwrap();
        let deserialized: KnowledgeTriple = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.subject, "Alice");
        assert_eq!(deserialized.confidence, 0.9);
    }

    // ─── KnowledgeGraph tests ───

    #[test]
    fn test_graph_new_is_empty() {
        let graph = KnowledgeGraph::new();
        assert!(graph.is_empty());
        assert_eq!(graph.len(), 0);
    }

    #[test]
    fn test_graph_add_triple() {
        let mut graph = KnowledgeGraph::new();
        graph.add_triple(KnowledgeTriple::new("Alice", "works at", "Google"));
        assert_eq!(graph.len(), 1);
        assert!(!graph.is_empty());
    }

    #[test]
    fn test_graph_get_triples_for_entity() {
        let mut graph = KnowledgeGraph::new();
        graph.add_triple(KnowledgeTriple::new("Alice", "works at", "Google"));
        graph.add_triple(KnowledgeTriple::new("Bob", "works at", "Meta"));
        graph.add_triple(KnowledgeTriple::new("Charlie", "knows", "Alice"));

        let alice_triples = graph.get_triples_for_entity("Alice");
        assert_eq!(alice_triples.len(), 2); // "Alice works at Google" + "Charlie knows Alice"
    }

    #[test]
    fn test_graph_get_triples_case_insensitive() {
        let mut graph = KnowledgeGraph::new();
        graph.add_triple(KnowledgeTriple::new("Alice", "works at", "Google"));

        let triples = graph.get_triples_for_entity("alice");
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn test_graph_get_related_entities() {
        let mut graph = KnowledgeGraph::new();
        graph.add_triple(KnowledgeTriple::new("Alice", "works at", "Google"));
        graph.add_triple(KnowledgeTriple::new("Alice", "knows", "Bob"));
        graph.add_triple(KnowledgeTriple::new("Charlie", "manages", "Alice"));

        let related = graph.get_related_entities("Alice");
        assert_eq!(related.len(), 3);
        assert!(related.contains(&"Google".to_string()));
        assert!(related.contains(&"Bob".to_string()));
        assert!(related.contains(&"Charlie".to_string()));
    }

    #[test]
    fn test_graph_search_triples() {
        let mut graph = KnowledgeGraph::new();
        graph.add_triple(KnowledgeTriple::new("Alice", "works at", "Google"));
        graph.add_triple(KnowledgeTriple::new("Bob", "lives in", "New York"));

        let results = graph.search_triples("google");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].subject, "Alice");
    }

    #[test]
    fn test_graph_search_triples_predicate() {
        let mut graph = KnowledgeGraph::new();
        graph.add_triple(KnowledgeTriple::new("Alice", "works at", "Google"));
        graph.add_triple(KnowledgeTriple::new("Bob", "works at", "Meta"));
        graph.add_triple(KnowledgeTriple::new("Charlie", "lives in", "NYC"));

        let results = graph.search_triples("works");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_graph_remove_triples_for_entity() {
        let mut graph = KnowledgeGraph::new();
        graph.add_triple(KnowledgeTriple::new("Alice", "works at", "Google"));
        graph.add_triple(KnowledgeTriple::new("Bob", "works at", "Meta"));
        graph.add_triple(KnowledgeTriple::new("Charlie", "knows", "Alice"));

        graph.remove_triples_for_entity("Alice");
        assert_eq!(graph.len(), 1);
        assert_eq!(graph.triples()[0].subject, "Bob");
    }

    #[test]
    fn test_graph_merge() {
        let mut graph1 = KnowledgeGraph::new();
        graph1.add_triple(KnowledgeTriple::new("Alice", "works at", "Google"));

        let mut graph2 = KnowledgeGraph::new();
        graph2.add_triple(KnowledgeTriple::new("Alice", "works at", "Google")); // duplicate
        graph2.add_triple(KnowledgeTriple::new("Bob", "works at", "Meta"));

        graph1.merge(&graph2);
        assert_eq!(graph1.len(), 2); // deduplication
    }

    #[test]
    fn test_graph_merge_case_insensitive_dedup() {
        let mut graph1 = KnowledgeGraph::new();
        graph1.add_triple(KnowledgeTriple::new("Alice", "works at", "Google"));

        let mut graph2 = KnowledgeGraph::new();
        graph2.add_triple(KnowledgeTriple::new("alice", "Works At", "google"));

        graph1.merge(&graph2);
        assert_eq!(graph1.len(), 1);
    }

    #[test]
    fn test_graph_to_natural_language() {
        let mut graph = KnowledgeGraph::new();
        graph.add_triple(KnowledgeTriple::new("Alice", "works at", "Google"));
        graph.add_triple(KnowledgeTriple::new("Bob", "lives in", "NYC"));

        let nl = graph.to_natural_language();
        assert!(nl.contains("Alice works at Google"));
        assert!(nl.contains("Bob lives in NYC"));
    }

    #[test]
    fn test_graph_to_natural_language_empty() {
        let graph = KnowledgeGraph::new();
        assert!(graph.to_natural_language().is_empty());
    }

    #[test]
    fn test_graph_clear() {
        let mut graph = KnowledgeGraph::new();
        graph.add_triple(KnowledgeTriple::new("Alice", "works at", "Google"));
        graph.clear();
        assert!(graph.is_empty());
    }

    // ─── RegexTripleExtractor tests ───

    #[test]
    fn test_extractor_is_pattern() {
        let extractor = RegexTripleExtractor::new();
        let triples = extractor.extract_triples("Alice is a software engineer.");
        assert!(!triples.is_empty());
        assert_eq!(triples[0].subject, "Alice");
        assert!(triples[0].object.contains("software engineer"));
    }

    #[test]
    fn test_extractor_works_at_pattern() {
        let extractor = RegexTripleExtractor::new();
        let triples = extractor.extract_triples("Bob works at Google.");
        assert!(!triples.is_empty());
        let t = triples.iter().find(|t| t.predicate == "works at").unwrap();
        assert_eq!(t.subject, "Bob");
        assert!(t.object.contains("Google"));
    }

    #[test]
    fn test_extractor_lives_in_pattern() {
        let extractor = RegexTripleExtractor::new();
        let triples = extractor.extract_triples("Charlie lives in New York.");
        assert!(!triples.is_empty());
        let t = triples.iter().find(|t| t.predicate == "lives in").unwrap();
        assert_eq!(t.subject, "Charlie");
        assert!(t.object.contains("New York"));
    }

    #[test]
    fn test_extractor_multiple_triples() {
        let extractor = RegexTripleExtractor::new();
        let triples = extractor.extract_triples("Alice is a developer. Bob works at Google.");
        assert!(triples.len() >= 2);
    }

    #[test]
    fn test_extractor_no_triples() {
        let extractor = RegexTripleExtractor::new();
        let triples = extractor.extract_triples("hello world, nothing to extract here.");
        assert!(triples.is_empty());
    }

    #[test]
    fn test_extractor_source_set() {
        let extractor = RegexTripleExtractor::new();
        let text = "Alice is a programmer.";
        let triples = extractor.extract_triples(text);
        assert!(!triples.is_empty());
        assert_eq!(triples[0].source.as_deref(), Some(text));
    }

    #[test]
    fn test_extractor_custom_pattern() {
        let mut extractor = RegexTripleExtractor::with_patterns(vec![]);
        extractor.add_pattern(
            Regex::new(r"(?i)\b([A-Z][a-zA-Z]*)\s+likes\s+(.+?)(?:\.|$)").unwrap(),
            "likes",
        );
        let triples = extractor.extract_triples("Alice likes chocolate.");
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].predicate, "likes");
    }

    #[test]
    fn test_extractor_deduplication() {
        let extractor = RegexTripleExtractor::new();
        // The "is a" and "is" patterns could both match, but dedup should prevent duplicates
        let triples = extractor.extract_triples("Alice is a developer.");
        let unique_subjects: std::collections::HashSet<_> =
            triples.iter().map(|t| (&t.subject, &t.object)).collect();
        // Each (subject, object) pair should be unique per predicate
        assert_eq!(unique_subjects.len(), triples.len());
    }

    // ─── KnowledgeGraphMemory integration tests ───

    #[tokio::test]
    async fn test_memory_save_and_load() {
        let mem = KnowledgeGraphMemory::new();

        let human = Message::human("Alice works at Google.");
        let ai = Message::ai("That's great! Alice is a talented engineer.");
        mem.save_context(&human, &ai).await.unwrap();

        let vars = mem.load_memory_variables().await.unwrap();
        assert!(vars.contains_key("history"));
        assert!(vars.contains_key("knowledge"));

        let knowledge = vars.get("knowledge").unwrap().as_str().unwrap();
        assert!(knowledge.contains("Alice"));
    }

    #[tokio::test]
    async fn test_memory_get_knowledge_for() {
        let mem = KnowledgeGraphMemory::new();

        mem.save_context(
            &Message::human("Alice works at Google."),
            &Message::ai("Nice!"),
        )
        .await
        .unwrap();

        let knowledge = mem.get_knowledge_for(&["Alice".to_string()]).await;
        assert!(knowledge.contains("Alice"));
        assert!(knowledge.contains("Google") || knowledge.contains("works"));
    }

    #[tokio::test]
    async fn test_memory_get_knowledge_for_unknown_entity() {
        let mem = KnowledgeGraphMemory::new();
        let knowledge = mem.get_knowledge_for(&["Nobody".to_string()]).await;
        assert_eq!(knowledge, "No relevant knowledge.");
    }

    #[tokio::test]
    async fn test_memory_clear() {
        let mem = KnowledgeGraphMemory::new();

        mem.save_context(
            &Message::human("Alice works at Google."),
            &Message::ai("Cool!"),
        )
        .await
        .unwrap();

        mem.clear().await.unwrap();
        assert_eq!(mem.triple_count().await, 0);

        let vars = mem.load_memory_variables().await.unwrap();
        let knowledge = vars.get("knowledge").unwrap().as_str().unwrap();
        assert_eq!(knowledge, "No knowledge extracted yet.");
    }

    #[tokio::test]
    async fn test_memory_builder_custom_keys() {
        let mem = KnowledgeGraphMemory::builder()
            .memory_key("chat")
            .knowledge_key("kg")
            .build();

        mem.save_context(
            &Message::human("Alice is a developer."),
            &Message::ai("Ok!"),
        )
        .await
        .unwrap();

        let vars = mem.load_memory_variables().await.unwrap();
        assert!(vars.contains_key("chat"));
        assert!(vars.contains_key("kg"));
        assert!(!vars.contains_key("history"));
        assert!(!vars.contains_key("knowledge"));
    }

    #[tokio::test]
    async fn test_memory_builder_initial_triples() {
        let mem = KnowledgeGraphMemory::builder()
            .initial_triples(vec![KnowledgeTriple::new("Alice", "works at", "Google")])
            .build();

        assert_eq!(mem.triple_count().await, 1);
        let knowledge = mem.get_knowledge_for(&["Alice".to_string()]).await;
        assert!(knowledge.contains("Alice works at Google"));
    }

    #[tokio::test]
    async fn test_memory_graph_snapshot() {
        let mem = KnowledgeGraphMemory::new();

        mem.save_context(
            &Message::human("Bob lives in Paris."),
            &Message::ai("Paris is beautiful."),
        )
        .await
        .unwrap();

        let snapshot = mem.graph_snapshot().await;
        assert!(!snapshot.is_empty());
    }

    #[tokio::test]
    async fn test_memory_key_default() {
        let mem = KnowledgeGraphMemory::new();
        assert_eq!(mem.memory_key(), "history");
    }
}
