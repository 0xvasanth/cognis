//! A versioned prompt registry for managing and rendering prompt templates.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

use serde_json::Value;

use cognis_core::error::{Result, CognisError};

/// A single versioned prompt entry.
#[derive(Debug, Clone)]
pub struct PromptEntry {
    /// Unique name of the template.
    pub name: String,
    /// The raw template string with `{variable}` placeholders.
    pub template: String,
    /// Monotonically increasing version number.
    pub version: u32,
    /// Optional human-readable description.
    pub description: Option<String>,
    /// Variable names extracted from the template.
    pub variables: Vec<String>,
    /// Arbitrary metadata.
    pub metadata: HashMap<String, Value>,
}

/// A thread-safe, versioned prompt template registry.
///
/// Templates are stored by name and each registration creates a new version,
/// allowing callers to retrieve the latest or a specific historical version.
pub struct PromptHub {
    templates: Arc<RwLock<HashMap<String, Vec<PromptEntry>>>>,
}

impl PromptHub {
    /// Create an empty `PromptHub`.
    pub fn new() -> Self {
        Self {
            templates: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a template under the given name.
    ///
    /// Variables are auto-extracted from `{var}` placeholders in the template.
    /// The version is auto-incremented from the previous version (starting at 1).
    pub fn register(
        &self,
        name: impl Into<String>,
        template: impl Into<String>,
        description: Option<String>,
    ) {
        let name = name.into();
        let template = template.into();
        let variables = extract_variables(&template);

        let mut store = self.templates.write().unwrap();
        let versions = store.entry(name.clone()).or_default();
        let version = versions.last().map_or(1, |e| e.version + 1);

        versions.push(PromptEntry {
            name,
            template,
            version,
            description,
            variables,
            metadata: HashMap::new(),
        });
    }

    /// Get the latest version of a template by name.
    pub fn get(&self, name: &str) -> Option<PromptEntry> {
        let store = self.templates.read().unwrap();
        store.get(name).and_then(|v| v.last().cloned())
    }

    /// Get a specific version of a template.
    pub fn get_version(&self, name: &str, version: u32) -> Option<PromptEntry> {
        let store = self.templates.read().unwrap();
        store
            .get(name)
            .and_then(|v| v.iter().find(|e| e.version == version).cloned())
    }

    /// Render the latest version of a template with the given variables.
    pub fn format(&self, name: &str, variables: &HashMap<String, String>) -> Result<String> {
        let entry = self
            .get(name)
            .ok_or_else(|| CognisError::Other(format!("Template '{}' not found", name)))?;
        format_template_str(&entry.template, variables)
    }

    /// List all registered template names.
    pub fn list(&self) -> Vec<String> {
        let store = self.templates.read().unwrap();
        store.keys().cloned().collect()
    }

    /// List all version numbers for a given template name.
    pub fn list_versions(&self, name: &str) -> Vec<u32> {
        let store = self.templates.read().unwrap();
        store
            .get(name)
            .map(|v| v.iter().map(|e| e.version).collect())
            .unwrap_or_default()
    }

    /// Delete all versions of a template.
    pub fn delete(&self, name: &str) {
        let mut store = self.templates.write().unwrap();
        store.remove(name);
    }

    /// Load templates from a directory.
    ///
    /// Each `.txt` or `.prompt` file is loaded as a template whose name is the
    /// file stem. If the first line starts with `#`, it is used as the
    /// description.
    pub async fn load_from_directory(path: &Path) -> Result<PromptHub> {
        let hub = PromptHub::new();
        let mut entries = tokio::fs::read_dir(path).await?;

        while let Some(entry) = entries.next_entry().await? {
            let file_path = entry.path();
            let ext = file_path.extension().and_then(|e| e.to_str()).unwrap_or("");

            if ext != "txt" && ext != "prompt" {
                continue;
            }

            let name = file_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();

            if name.is_empty() {
                continue;
            }

            let content = tokio::fs::read_to_string(&file_path).await?;
            let (description, template) = parse_file_content(&content);
            hub.register(name, template, description);
        }

        Ok(hub)
    }

    /// Save the latest version of every template to a directory.
    ///
    /// Each template is written to `<name>.prompt`. If a description exists it
    /// is written as a `# description` first line.
    pub async fn save_to_directory(&self, path: &Path) -> Result<()> {
        tokio::fs::create_dir_all(path).await?;

        let entries: Vec<PromptEntry> = {
            let store = self.templates.read().unwrap();
            store.values().filter_map(|v| v.last().cloned()).collect()
        };

        for entry in entries {
            let file_path = path.join(format!("{}.prompt", entry.name));
            let mut content = String::new();
            if let Some(ref desc) = entry.description {
                content.push_str(&format!("# {}\n", desc));
            }
            content.push_str(&entry.template);
            tokio::fs::write(&file_path, content).await?;
        }

        Ok(())
    }
}

impl Default for PromptHub {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract `{variable}` placeholders from a template string.
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

/// Render a template string by replacing `{var}` placeholders.
fn format_template_str(template: &str, variables: &HashMap<String, String>) -> Result<String> {
    let mut result = String::with_capacity(template.len());
    let mut chars = template.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '{' {
            if chars.peek() == Some(&'{') {
                chars.next();
                result.push('{');
                continue;
            }
            let mut name = String::new();
            for inner in chars.by_ref() {
                if inner == '}' {
                    break;
                }
                name.push(inner);
            }
            let value = variables.get(&name).ok_or_else(|| {
                CognisError::Other(format!(
                    "Missing variable '{}'. Available: {:?}",
                    name,
                    variables.keys().collect::<Vec<_>>()
                ))
            })?;
            result.push_str(value);
        } else if ch == '}' {
            if chars.peek() == Some(&'}') {
                chars.next();
                result.push('}');
            } else {
                result.push('}');
            }
        } else {
            result.push(ch);
        }
    }
    Ok(result)
}

/// Parse file content, extracting description from first `#` line.
fn parse_file_content(content: &str) -> (Option<String>, String) {
    if let Some(rest) = content.strip_prefix('#') {
        if let Some(newline_pos) = rest.find('\n') {
            let desc = rest[..newline_pos].trim().to_string();
            let template = rest[newline_pos + 1..].to_string();
            (Some(desc), template)
        } else {
            // Entire file is a description line with no template body
            (Some(rest.trim().to_string()), String::new())
        }
    } else {
        (None, content.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_register_and_retrieve() {
        let hub = PromptHub::new();
        hub.register("greet", "Hello {name}!", Some("A greeting".into()));
        let entry = hub.get("greet").unwrap();
        assert_eq!(entry.name, "greet");
        assert_eq!(entry.template, "Hello {name}!");
        assert_eq!(entry.version, 1);
        assert_eq!(entry.description, Some("A greeting".into()));
        assert_eq!(entry.variables, vec!["name".to_string()]);
    }

    #[test]
    fn test_version_auto_increment() {
        let hub = PromptHub::new();
        hub.register("t", "v1 {x}", None);
        hub.register("t", "v2 {x} {y}", None);
        let latest = hub.get("t").unwrap();
        assert_eq!(latest.version, 2);
        assert_eq!(latest.template, "v2 {x} {y}");
    }

    #[test]
    fn test_get_specific_version() {
        let hub = PromptHub::new();
        hub.register("t", "v1", None);
        hub.register("t", "v2", None);
        hub.register("t", "v3", None);
        let v2 = hub.get_version("t", 2).unwrap();
        assert_eq!(v2.template, "v2");
        assert_eq!(v2.version, 2);
    }

    #[test]
    fn test_format_with_variables() {
        let hub = PromptHub::new();
        hub.register("greet", "Hello {name}, welcome to {place}!", None);
        let mut vars = HashMap::new();
        vars.insert("name".into(), "Alice".into());
        vars.insert("place".into(), "Rust".into());
        let result = hub.format("greet", &vars).unwrap();
        assert_eq!(result, "Hello Alice, welcome to Rust!");
    }

    #[test]
    fn test_list_templates() {
        let hub = PromptHub::new();
        hub.register("a", "tmpl a", None);
        hub.register("b", "tmpl b", None);
        let mut names = hub.list();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn test_list_versions() {
        let hub = PromptHub::new();
        hub.register("t", "v1", None);
        hub.register("t", "v2", None);
        assert_eq!(hub.list_versions("t"), vec![1, 2]);
        assert_eq!(hub.list_versions("nonexistent"), Vec::<u32>::new());
    }

    #[test]
    fn test_delete_template() {
        let hub = PromptHub::new();
        hub.register("t", "hello", None);
        assert!(hub.get("t").is_some());
        hub.delete("t");
        assert!(hub.get("t").is_none());
    }

    #[tokio::test]
    async fn test_load_from_directory() {
        let dir = TempDir::new().unwrap();

        // Write a .prompt file with description
        std::fs::write(
            dir.path().join("greeting.prompt"),
            "# A friendly greeting\nHello {name}!",
        )
        .unwrap();

        // Write a .txt file without description
        std::fs::write(dir.path().join("farewell.txt"), "Goodbye {name}!").unwrap();

        // Write a non-prompt file that should be ignored
        std::fs::write(dir.path().join("notes.md"), "ignore me").unwrap();

        let hub = PromptHub::load_from_directory(dir.path()).await.unwrap();
        let mut names = hub.list();
        names.sort();
        assert_eq!(names, vec!["farewell", "greeting"]);

        let greet = hub.get("greeting").unwrap();
        assert_eq!(greet.description, Some("A friendly greeting".into()));
        assert_eq!(greet.template, "Hello {name}!");

        let farewell = hub.get("farewell").unwrap();
        assert!(farewell.description.is_none());
        assert_eq!(farewell.template, "Goodbye {name}!");
    }

    #[tokio::test]
    async fn test_save_to_directory() {
        let hub = PromptHub::new();
        hub.register("greet", "Hello {name}!", Some("A greeting".into()));
        hub.register("bye", "Goodbye!", None);

        let dir = TempDir::new().unwrap();
        hub.save_to_directory(dir.path()).await.unwrap();

        let greet_content = std::fs::read_to_string(dir.path().join("greet.prompt")).unwrap();
        assert!(greet_content.starts_with("# A greeting\n"));
        assert!(greet_content.contains("Hello {name}!"));

        let bye_content = std::fs::read_to_string(dir.path().join("bye.prompt")).unwrap();
        assert_eq!(bye_content, "Goodbye!");
    }
}
