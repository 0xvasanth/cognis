//! Prompt management — versioning, storage, composition, few-shot selection,
//! and simple optimisation utilities.

use std::collections::HashMap;

use serde_json::Value;

use cognis_core::error::{CognisError, Result};

// ---------------------------------------------------------------------------
// PromptVersion
// ---------------------------------------------------------------------------

/// A single immutable snapshot of a prompt at a particular version.
#[derive(Debug, Clone)]
pub struct PromptVersion {
    /// Semantic or sequential version string (e.g. "1", "1.0.0").
    pub version: String,
    /// The full prompt content for this version.
    pub content: String,
    /// ISO-8601 timestamp of when this version was created.
    pub created_at: String,
    /// Optional author who created this version.
    pub author: Option<String>,
    /// Optional changelog entry describing what changed.
    pub changelog: Option<String>,
}

impl PromptVersion {
    /// Serialise this version to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        let mut map = serde_json::Map::new();
        map.insert("version".into(), Value::String(self.version.clone()));
        map.insert("content".into(), Value::String(self.content.clone()));
        map.insert("created_at".into(), Value::String(self.created_at.clone()));
        map.insert(
            "author".into(),
            match &self.author {
                Some(a) => Value::String(a.clone()),
                None => Value::Null,
            },
        );
        map.insert(
            "changelog".into(),
            match &self.changelog {
                Some(c) => Value::String(c.clone()),
                None => Value::Null,
            },
        );
        Value::Object(map)
    }
}

// ---------------------------------------------------------------------------
// VersionedPrompt
// ---------------------------------------------------------------------------

/// A named prompt that tracks multiple versions with rollback support.
#[derive(Debug, Clone)]
pub struct VersionedPrompt {
    /// Human-readable name of this prompt.
    pub name: String,
    /// Ordered list of versions (oldest first).
    versions: Vec<PromptVersion>,
    /// Index into `versions` that is considered "current".
    current_idx: usize,
}

impl VersionedPrompt {
    /// Create a new versioned prompt with initial content (version "1").
    pub fn new(name: String, initial_content: String) -> Self {
        let version = PromptVersion {
            version: "1".to_string(),
            content: initial_content,
            created_at: "2026-03-08T00:00:00Z".to_string(),
            author: None,
            changelog: Some("Initial version".to_string()),
        };
        Self {
            name,
            versions: vec![version],
            current_idx: 0,
        }
    }

    /// Add a new version with the given content, author, and changelog.
    pub fn add_version(
        &mut self,
        content: String,
        author: Option<String>,
        changelog: Option<String>,
    ) {
        let next_num = self.versions.len() + 1;
        let version = PromptVersion {
            version: next_num.to_string(),
            content,
            created_at: "2026-03-08T00:00:00Z".to_string(),
            author,
            changelog,
        };
        self.versions.push(version);
        self.current_idx = self.versions.len() - 1;
    }

    /// Return a reference to the current version.
    pub fn current(&self) -> &PromptVersion {
        &self.versions[self.current_idx]
    }

    /// Look up a version by its version string.
    pub fn get_version(&self, version: &str) -> Option<&PromptVersion> {
        self.versions.iter().find(|v| v.version == version)
    }

    /// Roll back the current pointer to the specified version string.
    pub fn rollback(&mut self, version: &str) -> Result<()> {
        let idx = self
            .versions
            .iter()
            .position(|v| v.version == version)
            .ok_or_else(|| CognisError::Other(format!("Version '{}' not found", version)))?;
        self.current_idx = idx;
        Ok(())
    }

    /// Return the full ordered history of versions.
    pub fn history(&self) -> &[PromptVersion] {
        &self.versions
    }

    /// Return the number of stored versions.
    pub fn version_count(&self) -> usize {
        self.versions.len()
    }
}

// ---------------------------------------------------------------------------
// PromptLibrary
// ---------------------------------------------------------------------------

/// An in-memory library of named [`VersionedPrompt`]s with search support.
#[derive(Debug)]
pub struct PromptLibrary {
    prompts: HashMap<String, VersionedPrompt>,
}

impl PromptLibrary {
    /// Create an empty library.
    pub fn new() -> Self {
        Self {
            prompts: HashMap::new(),
        }
    }

    /// Add a new prompt (creates version 1).
    pub fn add(&mut self, name: &str, content: &str) {
        let vp = VersionedPrompt::new(name.to_string(), content.to_string());
        self.prompts.insert(name.to_string(), vp);
    }

    /// Get an immutable reference to a prompt by name.
    pub fn get(&self, name: &str) -> Option<&VersionedPrompt> {
        self.prompts.get(name)
    }

    /// Get a mutable reference to a prompt by name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut VersionedPrompt> {
        self.prompts.get_mut(name)
    }

    /// Remove a prompt by name, returning `true` if it existed.
    pub fn remove(&mut self, name: &str) -> bool {
        self.prompts.remove(name).is_some()
    }

    /// Search prompts whose name or current content contains `query`
    /// (case-insensitive).
    pub fn search(&self, query: &str) -> Vec<&VersionedPrompt> {
        let q = query.to_lowercase();
        self.prompts
            .values()
            .filter(|vp| {
                vp.name.to_lowercase().contains(&q)
                    || vp.current().content.to_lowercase().contains(&q)
            })
            .collect()
    }

    /// List all prompt names in the library.
    pub fn list(&self) -> Vec<&str> {
        self.prompts.keys().map(|s| s.as_str()).collect()
    }

    /// Return the number of prompts.
    pub fn len(&self) -> usize {
        self.prompts.len()
    }

    /// Return whether the library is empty.
    pub fn is_empty(&self) -> bool {
        self.prompts.is_empty()
    }

    /// Serialise the library to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        let mut map = serde_json::Map::new();
        for (name, vp) in &self.prompts {
            let versions: Vec<Value> = vp.versions.iter().map(|v| v.to_json()).collect();
            map.insert(name.clone(), Value::Array(versions));
        }
        Value::Object(map)
    }
}

impl Default for PromptLibrary {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PromptSectionComposer
// ---------------------------------------------------------------------------

/// Builds a single prompt string from multiple structured sections (system
/// message, context, examples, instructions).
#[derive(Debug, Clone)]
pub struct PromptSectionComposer {
    system: Option<String>,
    context: Vec<(String, String)>,
    examples: Vec<(String, String)>,
    instructions: Option<String>,
}

impl PromptSectionComposer {
    /// Create an empty composer.
    pub fn new() -> Self {
        Self {
            system: None,
            context: Vec::new(),
            examples: Vec::new(),
            instructions: None,
        }
    }

    /// Set the system message section.
    pub fn add_system(&mut self, content: &str) {
        self.system = Some(content.to_string());
    }

    /// Append a key-value context entry.
    pub fn add_context(&mut self, key: &str, value: &str) {
        self.context.push((key.to_string(), value.to_string()));
    }

    /// Append input/output example pairs.
    pub fn add_examples(&mut self, examples: Vec<(String, String)>) {
        self.examples.extend(examples);
    }

    /// Set the instructions section.
    pub fn add_instructions(&mut self, content: &str) {
        self.instructions = Some(content.to_string());
    }

    /// Compose all sections into a single prompt string.
    pub fn compose(&self) -> String {
        let mut parts: Vec<String> = Vec::new();

        if let Some(sys) = &self.system {
            parts.push(format!("[System]\n{}", sys));
        }

        if !self.context.is_empty() {
            let ctx_lines: Vec<String> = self
                .context
                .iter()
                .map(|(k, v)| format!("- {}: {}", k, v))
                .collect();
            parts.push(format!("[Context]\n{}", ctx_lines.join("\n")));
        }

        if !self.examples.is_empty() {
            let ex_lines: Vec<String> = self
                .examples
                .iter()
                .enumerate()
                .map(|(i, (input, output))| {
                    format!(
                        "Example {}:\n  Input: {}\n  Output: {}",
                        i + 1,
                        input,
                        output
                    )
                })
                .collect();
            parts.push(format!("[Examples]\n{}", ex_lines.join("\n")));
        }

        if let Some(instr) = &self.instructions {
            parts.push(format!("[Instructions]\n{}", instr));
        }

        parts.join("\n\n")
    }
}

impl Default for PromptSectionComposer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FewShotExample
// ---------------------------------------------------------------------------

/// A single few-shot example with an input, expected output, and optional
/// explanation.
#[derive(Debug, Clone)]
pub struct FewShotExample {
    /// The example input / question.
    pub input: String,
    /// The expected output / answer.
    pub output: String,
    /// Optional explanation of the reasoning.
    pub explanation: Option<String>,
}

impl FewShotExample {
    /// Serialise this example to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        let mut map = serde_json::Map::new();
        map.insert("input".into(), Value::String(self.input.clone()));
        map.insert("output".into(), Value::String(self.output.clone()));
        map.insert(
            "explanation".into(),
            match &self.explanation {
                Some(e) => Value::String(e.clone()),
                None => Value::Null,
            },
        );
        Value::Object(map)
    }

    /// Format the example as a human-readable string.
    pub fn format(&self) -> String {
        let mut s = format!("Input: {}\nOutput: {}", self.input, self.output);
        if let Some(explanation) = &self.explanation {
            s.push_str(&format!("\nExplanation: {}", explanation));
        }
        s
    }
}

// ---------------------------------------------------------------------------
// FewShotSelector
// ---------------------------------------------------------------------------

/// Selects relevant [`FewShotExample`]s using simple keyword similarity or
/// random sampling.
#[derive(Debug, Clone)]
pub struct FewShotSelector {
    examples: Vec<FewShotExample>,
}

impl FewShotSelector {
    /// Create a selector with the given pool of examples.
    pub fn new(examples: Vec<FewShotExample>) -> Self {
        Self { examples }
    }

    /// Select up to `k` examples whose input shares the most words with
    /// `query` (simple keyword overlap scoring).
    pub fn select_by_similarity<'a>(&'a self, query: &str, k: usize) -> Vec<&'a FewShotExample> {
        let query_words: Vec<String> = query
            .to_lowercase()
            .split_whitespace()
            .map(|w| w.to_string())
            .collect();

        let mut scored: Vec<(usize, &FewShotExample)> = self
            .examples
            .iter()
            .map(|ex| {
                let input_lower = ex.input.to_lowercase();
                let score = query_words
                    .iter()
                    .filter(|w| input_lower.contains(w.as_str()))
                    .count();
                (score, ex)
            })
            .collect();

        // Sort descending by score, then take top-k.
        scored.sort_by(|a, b| b.0.cmp(&a.0));
        scored.into_iter().take(k).map(|(_, ex)| ex).collect()
    }

    /// Select up to `k` examples in a deterministic round-robin fashion
    /// (evenly spaced indices). This avoids pulling in a random number
    /// generator dependency while still providing variety.
    pub fn select_random(&self, k: usize) -> Vec<&FewShotExample> {
        if self.examples.is_empty() || k == 0 {
            return Vec::new();
        }
        let n = self.examples.len();
        let take = k.min(n);
        let step = n as f64 / take as f64;
        (0..take)
            .map(|i| {
                let idx = (i as f64 * step) as usize;
                &self.examples[idx.min(n - 1)]
            })
            .collect()
    }

    /// Return the total number of examples in the pool.
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Return whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }
}

// ---------------------------------------------------------------------------
// PromptOptimizer
// ---------------------------------------------------------------------------

/// Simple, dependency-free prompt optimisation utilities.
pub struct PromptOptimizer;

impl PromptOptimizer {
    /// Collapse runs of whitespace into single spaces and trim.
    pub fn trim_whitespace(prompt: &str) -> String {
        prompt.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// Rough token count estimation (~4 characters per token, which is the
    /// commonly cited average for English text with GPT-style tokenisers).
    pub fn estimate_tokens(prompt: &str) -> usize {
        let chars = prompt.len();
        chars.div_ceil(4)
    }

    /// Truncate a prompt so that its estimated token count does not exceed
    /// `max_tokens`. Truncation happens at a word boundary when possible.
    pub fn truncate_to_tokens(prompt: &str, max_tokens: usize) -> String {
        let max_chars = max_tokens * 4;
        if prompt.len() <= max_chars {
            return prompt.to_string();
        }
        // Try to break at a word boundary.
        let truncated = &prompt[..max_chars];
        if let Some(last_space) = truncated.rfind(' ') {
            format!("{}...", &prompt[..last_space])
        } else {
            format!("{}...", truncated)
        }
    }

    /// Return a summary of the prompt by taking the first `max_chars`
    /// characters, breaking at a word boundary, and appending an ellipsis.
    pub fn summarize_prompt(prompt: &str, max_chars: usize) -> String {
        if prompt.len() <= max_chars {
            return prompt.to_string();
        }
        let slice = &prompt[..max_chars];
        if let Some(last_space) = slice.rfind(' ') {
            format!("{}...", &prompt[..last_space])
        } else {
            format!("{}...", slice)
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- PromptVersion -------------------------------------------------------

    #[test]
    fn test_prompt_version_to_json() {
        let v = PromptVersion {
            version: "1".into(),
            content: "Hello".into(),
            created_at: "2026-01-01".into(),
            author: Some("alice".into()),
            changelog: Some("init".into()),
        };
        let j = v.to_json();
        assert_eq!(j["version"], "1");
        assert_eq!(j["content"], "Hello");
        assert_eq!(j["author"], "alice");
        assert_eq!(j["changelog"], "init");
    }

    #[test]
    fn test_prompt_version_to_json_nulls() {
        let v = PromptVersion {
            version: "2".into(),
            content: "Hi".into(),
            created_at: "2026-01-02".into(),
            author: None,
            changelog: None,
        };
        let j = v.to_json();
        assert!(j["author"].is_null());
        assert!(j["changelog"].is_null());
    }

    // -- VersionedPrompt -----------------------------------------------------

    #[test]
    fn test_versioned_prompt_new() {
        let vp = VersionedPrompt::new("greet".into(), "Hello {name}".into());
        assert_eq!(vp.name, "greet");
        assert_eq!(vp.version_count(), 1);
        assert_eq!(vp.current().version, "1");
        assert_eq!(vp.current().content, "Hello {name}");
    }

    #[test]
    fn test_versioned_prompt_add_version() {
        let mut vp = VersionedPrompt::new("greet".into(), "v1".into());
        vp.add_version("v2".into(), Some("bob".into()), Some("updated".into()));
        assert_eq!(vp.version_count(), 2);
        assert_eq!(vp.current().version, "2");
        assert_eq!(vp.current().content, "v2");
        assert_eq!(vp.current().author.as_deref(), Some("bob"));
    }

    #[test]
    fn test_versioned_prompt_get_version() {
        let mut vp = VersionedPrompt::new("x".into(), "a".into());
        vp.add_version("b".into(), None, None);
        assert!(vp.get_version("1").is_some());
        assert!(vp.get_version("2").is_some());
        assert!(vp.get_version("3").is_none());
    }

    #[test]
    fn test_versioned_prompt_rollback_ok() {
        let mut vp = VersionedPrompt::new("x".into(), "a".into());
        vp.add_version("b".into(), None, None);
        vp.add_version("c".into(), None, None);
        assert_eq!(vp.current().version, "3");
        vp.rollback("1").unwrap();
        assert_eq!(vp.current().version, "1");
        assert_eq!(vp.current().content, "a");
    }

    #[test]
    fn test_versioned_prompt_rollback_err() {
        let mut vp = VersionedPrompt::new("x".into(), "a".into());
        let res = vp.rollback("99");
        assert!(res.is_err());
    }

    #[test]
    fn test_versioned_prompt_history() {
        let mut vp = VersionedPrompt::new("x".into(), "a".into());
        vp.add_version("b".into(), None, None);
        let h = vp.history();
        assert_eq!(h.len(), 2);
        assert_eq!(h[0].content, "a");
        assert_eq!(h[1].content, "b");
    }

    #[test]
    fn test_versioned_prompt_rollback_then_add() {
        let mut vp = VersionedPrompt::new("x".into(), "a".into());
        vp.add_version("b".into(), None, None);
        vp.rollback("1").unwrap();
        assert_eq!(vp.current().content, "a");
        vp.add_version("c".into(), None, None);
        assert_eq!(vp.current().content, "c");
        assert_eq!(vp.version_count(), 3);
    }

    // -- PromptLibrary -------------------------------------------------------

    #[test]
    fn test_library_new_empty() {
        let lib = PromptLibrary::new();
        assert_eq!(lib.len(), 0);
        assert!(lib.is_empty());
    }

    #[test]
    fn test_library_add_and_get() {
        let mut lib = PromptLibrary::new();
        lib.add("greet", "Hello {name}");
        assert_eq!(lib.len(), 1);
        let vp = lib.get("greet").unwrap();
        assert_eq!(vp.current().content, "Hello {name}");
    }

    #[test]
    fn test_library_get_missing() {
        let lib = PromptLibrary::new();
        assert!(lib.get("nope").is_none());
    }

    #[test]
    fn test_library_get_mut() {
        let mut lib = PromptLibrary::new();
        lib.add("greet", "v1");
        {
            let vp = lib.get_mut("greet").unwrap();
            vp.add_version("v2".into(), None, None);
        }
        assert_eq!(lib.get("greet").unwrap().current().content, "v2");
    }

    #[test]
    fn test_library_remove() {
        let mut lib = PromptLibrary::new();
        lib.add("greet", "Hello");
        assert!(lib.remove("greet"));
        assert!(!lib.remove("greet"));
        assert_eq!(lib.len(), 0);
    }

    #[test]
    fn test_library_search_by_name() {
        let mut lib = PromptLibrary::new();
        lib.add("greeting", "Hello");
        lib.add("farewell", "Goodbye");
        let results = lib.search("greet");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "greeting");
    }

    #[test]
    fn test_library_search_by_content() {
        let mut lib = PromptLibrary::new();
        lib.add("p1", "summarize the document");
        lib.add("p2", "translate text");
        let results = lib.search("summarize");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_library_search_case_insensitive() {
        let mut lib = PromptLibrary::new();
        lib.add("MyPrompt", "Content here");
        let results = lib.search("myprompt");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_library_search_no_match() {
        let mut lib = PromptLibrary::new();
        lib.add("p1", "hello");
        let results = lib.search("xyz");
        assert!(results.is_empty());
    }

    #[test]
    fn test_library_list() {
        let mut lib = PromptLibrary::new();
        lib.add("alpha", "a");
        lib.add("beta", "b");
        let mut names = lib.list();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_library_to_json() {
        let mut lib = PromptLibrary::new();
        lib.add("p1", "content");
        let j = lib.to_json();
        assert!(j.is_object());
        assert!(j["p1"].is_array());
    }

    #[test]
    fn test_library_default() {
        let lib = PromptLibrary::default();
        assert!(lib.is_empty());
    }

    // -- PromptComposer ------------------------------------------------------

    #[test]
    fn test_composer_empty() {
        let c = PromptSectionComposer::new();
        assert_eq!(c.compose(), "");
    }

    #[test]
    fn test_composer_system_only() {
        let mut c = PromptSectionComposer::new();
        c.add_system("You are helpful.");
        assert!(c.compose().contains("[System]"));
        assert!(c.compose().contains("You are helpful."));
    }

    #[test]
    fn test_composer_context() {
        let mut c = PromptSectionComposer::new();
        c.add_context("language", "Rust");
        let out = c.compose();
        assert!(out.contains("[Context]"));
        assert!(out.contains("- language: Rust"));
    }

    #[test]
    fn test_composer_examples() {
        let mut c = PromptSectionComposer::new();
        c.add_examples(vec![("hi".into(), "hello".into())]);
        let out = c.compose();
        assert!(out.contains("[Examples]"));
        assert!(out.contains("Input: hi"));
        assert!(out.contains("Output: hello"));
    }

    #[test]
    fn test_composer_instructions() {
        let mut c = PromptSectionComposer::new();
        c.add_instructions("Be concise.");
        let out = c.compose();
        assert!(out.contains("[Instructions]"));
        assert!(out.contains("Be concise."));
    }

    #[test]
    fn test_composer_full() {
        let mut c = PromptSectionComposer::new();
        c.add_system("System msg");
        c.add_context("k", "v");
        c.add_examples(vec![("a".into(), "b".into())]);
        c.add_instructions("Do stuff");
        let out = c.compose();
        assert!(out.contains("[System]"));
        assert!(out.contains("[Context]"));
        assert!(out.contains("[Examples]"));
        assert!(out.contains("[Instructions]"));
    }

    #[test]
    fn test_composer_default() {
        let c = PromptSectionComposer::default();
        assert_eq!(c.compose(), "");
    }

    #[test]
    fn test_composer_multiple_contexts() {
        let mut c = PromptSectionComposer::new();
        c.add_context("a", "1");
        c.add_context("b", "2");
        let out = c.compose();
        assert!(out.contains("- a: 1"));
        assert!(out.contains("- b: 2"));
    }

    // -- FewShotExample ------------------------------------------------------

    #[test]
    fn test_few_shot_example_format() {
        let ex = FewShotExample {
            input: "2+2".into(),
            output: "4".into(),
            explanation: None,
        };
        let f = ex.format();
        assert!(f.contains("Input: 2+2"));
        assert!(f.contains("Output: 4"));
        assert!(!f.contains("Explanation"));
    }

    #[test]
    fn test_few_shot_example_format_with_explanation() {
        let ex = FewShotExample {
            input: "2+2".into(),
            output: "4".into(),
            explanation: Some("addition".into()),
        };
        let f = ex.format();
        assert!(f.contains("Explanation: addition"));
    }

    #[test]
    fn test_few_shot_example_to_json() {
        let ex = FewShotExample {
            input: "q".into(),
            output: "a".into(),
            explanation: Some("reason".into()),
        };
        let j = ex.to_json();
        assert_eq!(j["input"], "q");
        assert_eq!(j["output"], "a");
        assert_eq!(j["explanation"], "reason");
    }

    #[test]
    fn test_few_shot_example_to_json_null_explanation() {
        let ex = FewShotExample {
            input: "q".into(),
            output: "a".into(),
            explanation: None,
        };
        let j = ex.to_json();
        assert!(j["explanation"].is_null());
    }

    // -- FewShotSelector -----------------------------------------------------

    fn sample_examples() -> Vec<FewShotExample> {
        vec![
            FewShotExample {
                input: "translate hello to french".into(),
                output: "bonjour".into(),
                explanation: None,
            },
            FewShotExample {
                input: "translate goodbye to french".into(),
                output: "au revoir".into(),
                explanation: None,
            },
            FewShotExample {
                input: "summarize the article about rust".into(),
                output: "Rust is a systems language.".into(),
                explanation: None,
            },
            FewShotExample {
                input: "write a poem about nature".into(),
                output: "Trees sway gently...".into(),
                explanation: None,
            },
        ]
    }

    #[test]
    fn test_selector_len() {
        let sel = FewShotSelector::new(sample_examples());
        assert_eq!(sel.len(), 4);
        assert!(!sel.is_empty());
    }

    #[test]
    fn test_selector_empty() {
        let sel = FewShotSelector::new(vec![]);
        assert_eq!(sel.len(), 0);
        assert!(sel.is_empty());
    }

    #[test]
    fn test_selector_similarity_translate() {
        let sel = FewShotSelector::new(sample_examples());
        let results = sel.select_by_similarity("translate this to french", 2);
        assert_eq!(results.len(), 2);
        // Both translate examples should be top-scored.
        assert!(results[0].input.contains("translate"));
        assert!(results[1].input.contains("translate"));
    }

    #[test]
    fn test_selector_similarity_k_larger_than_pool() {
        let sel = FewShotSelector::new(sample_examples());
        let results = sel.select_by_similarity("translate", 100);
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_selector_similarity_no_overlap() {
        let sel = FewShotSelector::new(sample_examples());
        let results = sel.select_by_similarity("xyzzy", 2);
        // Still returns 2 examples (with score 0), just no particular ordering.
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_selector_random() {
        let sel = FewShotSelector::new(sample_examples());
        let results = sel.select_random(2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_selector_random_k_zero() {
        let sel = FewShotSelector::new(sample_examples());
        let results = sel.select_random(0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_selector_random_empty_pool() {
        let sel = FewShotSelector::new(vec![]);
        let results = sel.select_random(5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_selector_random_k_exceeds_pool() {
        let sel = FewShotSelector::new(sample_examples());
        let results = sel.select_random(100);
        assert_eq!(results.len(), 4);
    }

    // -- PromptOptimizer -----------------------------------------------------

    #[test]
    fn test_optimizer_trim_whitespace() {
        let result = PromptOptimizer::trim_whitespace("  hello   world  ");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_optimizer_trim_whitespace_newlines() {
        let result = PromptOptimizer::trim_whitespace("hello\n\n  world\t!");
        assert_eq!(result, "hello world !");
    }

    #[test]
    fn test_optimizer_trim_whitespace_empty() {
        assert_eq!(PromptOptimizer::trim_whitespace(""), "");
    }

    #[test]
    fn test_optimizer_estimate_tokens() {
        // 12 chars -> ceil(12/4) = 3
        assert_eq!(PromptOptimizer::estimate_tokens("hello world!"), 3);
    }

    #[test]
    fn test_optimizer_estimate_tokens_empty() {
        assert_eq!(PromptOptimizer::estimate_tokens(""), 0);
    }

    #[test]
    fn test_optimizer_truncate_no_op() {
        let prompt = "short";
        let result = PromptOptimizer::truncate_to_tokens(prompt, 100);
        assert_eq!(result, "short");
    }

    #[test]
    fn test_optimizer_truncate_cuts() {
        let prompt = "this is a longer prompt that should be truncated";
        let result = PromptOptimizer::truncate_to_tokens(prompt, 3); // ~12 chars
        assert!(result.len() < prompt.len());
        assert!(result.ends_with("..."));
    }

    #[test]
    fn test_optimizer_summarize_short() {
        let result = PromptOptimizer::summarize_prompt("hi", 100);
        assert_eq!(result, "hi");
    }

    #[test]
    fn test_optimizer_summarize_long() {
        let prompt = "This is a long prompt that needs to be summarized down";
        let result = PromptOptimizer::summarize_prompt(prompt, 20);
        assert!(result.ends_with("..."));
        assert!(result.len() <= 24); // 20 + "..."
    }

    #[test]
    fn test_optimizer_estimate_tokens_one_char() {
        assert_eq!(PromptOptimizer::estimate_tokens("a"), 1);
    }
}
