//! A local prompt registry for storing, versioning, sharing, and reusing prompt templates.
//!
//! Provides [`PromptRegistry`] for managing named prompt templates with version
//! tracking, category organisation, search, import/export, and a built-in
//! library of common prompts.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use rustchain_core::error::{Result, RustChainError};

// ---------------------------------------------------------------------------
// RegisteredPrompt
// ---------------------------------------------------------------------------

/// A single registered prompt template with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredPrompt {
    /// Unique name of the prompt.
    pub name: String,
    /// Optional human-readable description.
    pub description: Option<String>,
    /// The raw template string with `{variable}` placeholders.
    pub template: String,
    /// Variable names extracted from the template.
    pub input_variables: Vec<String>,
    /// Monotonically increasing version number (starts at 1).
    pub version: u32,
    /// Free-form tags for organisation and search.
    pub tags: Vec<String>,
    /// Category this prompt belongs to (e.g. "chat", "extraction").
    pub category: Option<String>,
    /// Arbitrary metadata.
    pub metadata: HashMap<String, Value>,
    /// ISO-8601 creation timestamp.
    pub created_at: String,
    /// ISO-8601 last-update timestamp.
    pub updated_at: String,
}

// ---------------------------------------------------------------------------
// Internal storage
// ---------------------------------------------------------------------------

/// Per-name storage: all versions plus the index of the "current" version.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PromptVersionStore {
    versions: Vec<RegisteredPrompt>,
    /// Index into `versions` that is considered "current". Normally points to
    /// the last element, but can be changed by [`PromptRegistry::rollback`].
    current_idx: usize,
}

// ---------------------------------------------------------------------------
// PromptRegistry
// ---------------------------------------------------------------------------

/// A thread-safe, in-memory registry of named prompt templates with version
/// tracking, category filtering, search, and import/export support.
#[derive(Clone)]
pub struct PromptRegistry {
    store: Arc<RwLock<HashMap<String, PromptVersionStore>>>,
}

impl PromptRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            store: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a registry pre-loaded with a set of commonly-used prompt
    /// templates.
    pub fn with_defaults() -> Self {
        let reg = Self::new();

        reg.register_full(
            "qa",
            "Answer the following question based on the provided context.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
            Some("Basic question-answering template".into()),
            vec!["qa".into(), "default".into()],
            Some("chat".into()),
            HashMap::new(),
        );

        reg.register_full(
            "summarize",
            "Summarize the following text concisely.\n\nText: {text}\n\nSummary:",
            Some("Text summarization template".into()),
            vec!["summarization".into(), "default".into()],
            Some("summarization".into()),
            HashMap::new(),
        );

        reg.register_full(
            "translate",
            "Translate the following text from {source_language} to {target_language}.\n\nText: {text}\n\nTranslation:",
            Some("Translation template".into()),
            vec!["translation".into(), "default".into()],
            Some("chat".into()),
            HashMap::new(),
        );

        reg.register_full(
            "extract",
            "Extract structured data from the following text. Return the result as JSON.\n\nText: {text}\n\nSchema: {schema}\n\nExtracted data:",
            Some("Structured data extraction template".into()),
            vec!["extraction".into(), "default".into()],
            Some("extraction".into()),
            HashMap::new(),
        );

        reg.register_full(
            "chat",
            "You are a helpful assistant.\n\n{input}",
            Some("Basic chat system prompt".into()),
            vec!["chat".into(), "default".into()],
            Some("chat".into()),
            HashMap::new(),
        );

        reg
    }

    // -- Registration -------------------------------------------------------

    /// Register (or update) a prompt template.
    ///
    /// If a prompt with the same name already exists a new version is created.
    /// The latest registered version becomes the current one.
    pub fn register(&self, name: impl Into<String>, template: impl Into<String>) {
        self.register_full(name, template, None, Vec::new(), None, HashMap::new());
    }

    /// Register a prompt with full metadata.
    pub fn register_full(
        &self,
        name: impl Into<String>,
        template: impl Into<String>,
        description: Option<String>,
        tags: Vec<String>,
        category: Option<String>,
        metadata: HashMap<String, Value>,
    ) {
        let name = name.into();
        let template = template.into();
        let input_variables = extract_variables(&template);
        let now = current_timestamp();

        let mut store = self.store.write().unwrap();
        let entry = store
            .entry(name.clone())
            .or_insert_with(|| PromptVersionStore {
                versions: Vec::new(),
                current_idx: 0,
            });

        let version = entry.versions.last().map_or(1, |p| p.version + 1);
        let created_at = entry
            .versions
            .first()
            .map(|p| p.created_at.clone())
            .unwrap_or_else(|| now.clone());

        entry.versions.push(RegisteredPrompt {
            name,
            description,
            template,
            input_variables,
            version,
            tags,
            category,
            metadata,
            created_at,
            updated_at: now,
        });
        entry.current_idx = entry.versions.len() - 1;
    }

    // -- Retrieval ----------------------------------------------------------

    /// Get the current version of a prompt by name.
    pub fn get(&self, name: &str) -> Option<RegisteredPrompt> {
        let store = self.store.read().unwrap();
        store.get(name).map(|s| s.versions[s.current_idx].clone())
    }

    /// Get a specific version of a prompt.
    pub fn get_version(&self, name: &str, version: u32) -> Option<RegisteredPrompt> {
        let store = self.store.read().unwrap();
        store
            .get(name)
            .and_then(|s| s.versions.iter().find(|p| p.version == version).cloned())
    }

    /// List all registered prompt names.
    pub fn list(&self) -> Vec<String> {
        let store = self.store.read().unwrap();
        store.keys().cloned().collect()
    }

    /// List all version numbers for a given prompt name.
    pub fn list_versions(&self, name: &str) -> Vec<u32> {
        let store = self.store.read().unwrap();
        store
            .get(name)
            .map(|s| s.versions.iter().map(|p| p.version).collect())
            .unwrap_or_default()
    }

    /// Remove a prompt (all versions). Returns `true` if it existed.
    pub fn remove(&self, name: &str) -> bool {
        let mut store = self.store.write().unwrap();
        store.remove(name).is_some()
    }

    // -- Search -------------------------------------------------------------

    /// Simple text search across prompt names and descriptions. Returns
    /// matching prompt names (case-insensitive).
    pub fn search(&self, query: &str) -> Vec<String> {
        let q = query.to_lowercase();
        let store = self.store.read().unwrap();
        store
            .iter()
            .filter(|(name, vs)| {
                let current = &vs.versions[vs.current_idx];
                name.to_lowercase().contains(&q)
                    || current
                        .description
                        .as_deref()
                        .is_some_and(|d| d.to_lowercase().contains(&q))
                    || current.tags.iter().any(|t| t.to_lowercase().contains(&q))
            })
            .map(|(name, _)| name.clone())
            .collect()
    }

    // -- Category -----------------------------------------------------------

    /// List all prompt names that belong to the given category.
    pub fn list_by_category(&self, category: &str) -> Vec<String> {
        let cat = category.to_lowercase();
        let store = self.store.read().unwrap();
        store
            .iter()
            .filter(|(_, vs)| {
                let current = &vs.versions[vs.current_idx];
                current
                    .category
                    .as_deref()
                    .is_some_and(|c| c.to_lowercase() == cat)
            })
            .map(|(name, _)| name.clone())
            .collect()
    }

    // -- Version management -------------------------------------------------

    /// Roll back to a previous version, making it the current one.
    ///
    /// Returns `Ok(())` on success, or an error if the name or version is not
    /// found.
    pub fn rollback(&self, name: &str, version: u32) -> Result<()> {
        let mut store = self.store.write().unwrap();
        let entry = store
            .get_mut(name)
            .ok_or_else(|| RustChainError::Other(format!("Prompt '{}' not found", name)))?;
        let idx = entry
            .versions
            .iter()
            .position(|p| p.version == version)
            .ok_or_else(|| {
                RustChainError::Other(format!(
                    "Version {} not found for prompt '{}'",
                    version, name
                ))
            })?;
        entry.current_idx = idx;
        Ok(())
    }
}

impl Default for PromptRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PromptExporter
// ---------------------------------------------------------------------------

/// Utilities for exporting and importing a [`PromptRegistry`] as JSON (or
/// optionally YAML).
pub struct PromptExporter;

impl PromptExporter {
    /// Export all prompts (all versions) as a JSON string.
    pub fn export_json(registry: &PromptRegistry) -> Result<String> {
        let store = registry.store.read().unwrap();
        let map: HashMap<String, Vec<RegisteredPrompt>> = store
            .iter()
            .map(|(k, v)| (k.clone(), v.versions.clone()))
            .collect();
        serde_json::to_string_pretty(&map).map_err(|e| RustChainError::Other(e.to_string()))
    }

    /// Import prompts from a JSON string, returning a new registry.
    pub fn import_json(json: &str) -> Result<PromptRegistry> {
        let map: HashMap<String, Vec<RegisteredPrompt>> =
            serde_json::from_str(json).map_err(|e| RustChainError::Other(e.to_string()))?;

        let registry = PromptRegistry::new();
        {
            let mut store = registry.store.write().unwrap();
            for (name, versions) in map {
                let current_idx = versions.len().saturating_sub(1);
                store.insert(
                    name,
                    PromptVersionStore {
                        versions,
                        current_idx,
                    },
                );
            }
        }
        Ok(registry)
    }

    /// Export all prompts as YAML (requires the `yaml` feature).
    #[cfg(feature = "yaml")]
    pub fn export_yaml(registry: &PromptRegistry) -> Result<String> {
        let store = registry.store.read().unwrap();
        let map: HashMap<String, Vec<RegisteredPrompt>> = store
            .iter()
            .map(|(k, v)| (k.clone(), v.versions.clone()))
            .collect();
        serde_yaml::to_string(&map).map_err(|e| RustChainError::Other(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// PromptComposer
// ---------------------------------------------------------------------------

/// Utilities for combining prompt templates.
pub struct PromptComposer;

impl PromptComposer {
    /// Concatenate raw template strings with a separator.
    pub fn compose(prompts: &[&str], separator: &str) -> String {
        prompts.join(separator)
    }

    /// Look up prompts by name in the registry and concatenate their templates.
    pub fn chain_prompts(names: &[&str], registry: &PromptRegistry) -> Result<String> {
        let mut parts = Vec::with_capacity(names.len());
        for name in names {
            let prompt = registry.get(name).ok_or_else(|| {
                RustChainError::Other(format!("Prompt '{}' not found in registry", name))
            })?;
            parts.push(prompt.template);
        }
        Ok(parts.join("\n\n"))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract `{variable}` placeholders from a template string, skipping escaped
/// `{{` sequences.
fn extract_variables(template: &str) -> Vec<String> {
    let mut vars = Vec::new();
    let mut chars = template.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '{' {
            if chars.peek() == Some(&'{') {
                chars.next();
                continue;
            }
            let mut name = String::new();
            for inner in chars.by_ref() {
                if inner == '}' {
                    break;
                }
                name.push(inner);
            }
            if !name.is_empty() && !vars.contains(&name) {
                vars.push(name);
            }
        } else if ch == '}' && chars.peek() == Some(&'}') {
            chars.next();
        }
    }
    vars
}

/// Return an ISO-8601 style timestamp string. Uses a simple UTC-style format
/// without pulling in the `chrono` crate.
fn current_timestamp() -> String {
    // Use std SystemTime for a basic timestamp.
    use std::time::{SystemTime, UNIX_EPOCH};
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    // Format as a simple epoch-based string. Good enough for ordering/display.
    format!("{}", secs)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_retrieve() {
        let reg = PromptRegistry::new();
        reg.register("greet", "Hello {name}!");
        let p = reg.get("greet").unwrap();
        assert_eq!(p.name, "greet");
        assert_eq!(p.template, "Hello {name}!");
        assert_eq!(p.version, 1);
        assert_eq!(p.input_variables, vec!["name".to_string()]);
    }

    #[test]
    fn test_version_tracking() {
        let reg = PromptRegistry::new();
        reg.register("t", "v1 {x}");
        reg.register("t", "v2 {x} {y}");
        let latest = reg.get("t").unwrap();
        assert_eq!(latest.version, 2);
        assert_eq!(latest.template, "v2 {x} {y}");
        assert_eq!(
            latest.input_variables,
            vec!["x".to_string(), "y".to_string()]
        );
    }

    #[test]
    fn test_list_all_prompts() {
        let reg = PromptRegistry::new();
        reg.register("a", "tmpl a");
        reg.register("b", "tmpl b");
        let mut names = reg.list();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn test_list_versions() {
        let reg = PromptRegistry::new();
        reg.register("t", "v1");
        reg.register("t", "v2");
        reg.register("t", "v3");
        assert_eq!(reg.list_versions("t"), vec![1, 2, 3]);
        assert!(reg.list_versions("missing").is_empty());
    }

    #[test]
    fn test_remove_prompt() {
        let reg = PromptRegistry::new();
        reg.register("t", "hello");
        assert!(reg.get("t").is_some());
        assert!(reg.remove("t"));
        assert!(reg.get("t").is_none());
        // Removing again returns false
        assert!(!reg.remove("t"));
    }

    #[test]
    fn test_search_by_name() {
        let reg = PromptRegistry::new();
        reg.register_full(
            "greeting",
            "Hello {name}!",
            Some("A friendly greeting".into()),
            vec!["social".into()],
            None,
            HashMap::new(),
        );
        reg.register("farewell", "Goodbye!");

        let results = reg.search("greet");
        assert!(results.contains(&"greeting".to_string()));
        assert!(!results.contains(&"farewell".to_string()));

        // Search in description
        let results2 = reg.search("friendly");
        assert!(results2.contains(&"greeting".to_string()));

        // Search in tags
        let results3 = reg.search("social");
        assert!(results3.contains(&"greeting".to_string()));
    }

    #[test]
    fn test_get_specific_version() {
        let reg = PromptRegistry::new();
        reg.register("t", "v1");
        reg.register("t", "v2");
        reg.register("t", "v3");
        let v2 = reg.get_version("t", 2).unwrap();
        assert_eq!(v2.template, "v2");
        assert_eq!(v2.version, 2);
        assert!(reg.get_version("t", 99).is_none());
    }

    #[test]
    fn test_rollback_to_previous_version() {
        let reg = PromptRegistry::new();
        reg.register("t", "version-1");
        reg.register("t", "version-2");
        reg.register("t", "version-3");

        // Current should be v3
        assert_eq!(reg.get("t").unwrap().version, 3);

        // Roll back to v1
        reg.rollback("t", 1).unwrap();
        let current = reg.get("t").unwrap();
        assert_eq!(current.version, 1);
        assert_eq!(current.template, "version-1");

        // Error on missing name
        assert!(reg.rollback("missing", 1).is_err());
        // Error on missing version
        assert!(reg.rollback("t", 99).is_err());
    }

    #[test]
    fn test_category_filtering() {
        let reg = PromptRegistry::new();
        reg.register_full(
            "a",
            "tmpl a",
            None,
            vec![],
            Some("chat".into()),
            HashMap::new(),
        );
        reg.register_full(
            "b",
            "tmpl b",
            None,
            vec![],
            Some("extraction".into()),
            HashMap::new(),
        );
        reg.register_full(
            "c",
            "tmpl c",
            None,
            vec![],
            Some("chat".into()),
            HashMap::new(),
        );

        let mut chat = reg.list_by_category("chat");
        chat.sort();
        assert_eq!(chat, vec!["a", "c"]);

        let extraction = reg.list_by_category("extraction");
        assert_eq!(extraction, vec!["b"]);

        assert!(reg.list_by_category("nonexistent").is_empty());
    }

    #[test]
    fn test_export_import_json_roundtrip() {
        let reg = PromptRegistry::new();
        reg.register_full(
            "greet",
            "Hello {name}!",
            Some("A greeting".into()),
            vec!["social".into()],
            Some("chat".into()),
            HashMap::new(),
        );
        reg.register("greet", "Hi {name}!"); // v2

        let json = PromptExporter::export_json(&reg).unwrap();
        let reg2 = PromptExporter::import_json(&json).unwrap();

        assert_eq!(reg2.list_versions("greet"), vec![1, 2]);
        let p = reg2.get("greet").unwrap();
        assert_eq!(p.template, "Hi {name}!");
        assert_eq!(p.version, 2);
    }

    #[test]
    fn test_default_prompts_loaded() {
        let reg = PromptRegistry::with_defaults();
        let mut names = reg.list();
        names.sort();
        assert!(names.contains(&"qa".to_string()));
        assert!(names.contains(&"summarize".to_string()));
        assert!(names.contains(&"translate".to_string()));
        assert!(names.contains(&"extract".to_string()));
        assert!(names.contains(&"chat".to_string()));

        let qa = reg.get("qa").unwrap();
        assert!(qa.template.contains("{context}"));
        assert!(qa.template.contains("{question}"));
        assert!(qa.description.is_some());
    }

    #[test]
    fn test_prompt_composer() {
        let result = PromptComposer::compose(&["Hello {name}!", "How are you?"], "\n");
        assert_eq!(result, "Hello {name}!\nHow are you?");

        let reg = PromptRegistry::new();
        reg.register("a", "Part A");
        reg.register("b", "Part B");

        let chained = PromptComposer::chain_prompts(&["a", "b"], &reg).unwrap();
        assert_eq!(chained, "Part A\n\nPart B");

        // Missing prompt returns error
        assert!(PromptComposer::chain_prompts(&["a", "missing"], &reg).is_err());
    }

    #[test]
    fn test_thread_safety() {
        use std::thread;

        let reg = PromptRegistry::new();
        let mut handles = vec![];

        for i in 0..10 {
            let r = reg.clone();
            handles.push(thread::spawn(move || {
                let name = format!("prompt_{}", i);
                r.register(&name, format!("Template {}", i));
                assert!(r.get(&name).is_some());
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(reg.list().len(), 10);
    }

    #[test]
    fn test_metadata_and_tags() {
        let mut meta = HashMap::new();
        meta.insert("author".to_string(), Value::String("alice".to_string()));
        meta.insert("priority".to_string(), serde_json::json!(1));

        let reg = PromptRegistry::new();
        reg.register_full(
            "tagged",
            "Hello {name}!",
            Some("With metadata".into()),
            vec!["greeting".into(), "test".into()],
            Some("chat".into()),
            meta.clone(),
        );

        let p = reg.get("tagged").unwrap();
        assert_eq!(p.tags, vec!["greeting", "test"]);
        assert_eq!(
            p.metadata.get("author").unwrap(),
            &Value::String("alice".to_string())
        );
        assert_eq!(p.metadata.get("priority").unwrap(), &serde_json::json!(1));
        assert_eq!(p.category, Some("chat".to_string()));
        assert!(!p.created_at.is_empty());
        assert!(!p.updated_at.is_empty());
    }
}
