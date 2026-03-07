//! Pre-configured agent presets for common use cases.
//!
//! This module provides ready-to-use agent configurations via [`AgentPreset`],
//! a [`PresetRegistry`] for managing and discovering presets, and a
//! [`PresetCustomizer`] for tweaking presets before use.
//!
//! # Built-in Presets
//!
//! - [`coding_assistant`] -- code generation, review, and filesystem tools
//! - [`research_assistant`] -- web search and summarization with higher token limits
//! - [`data_analyst`] -- structured output and data analysis tooling
//! - [`writing_assistant`] -- creative writing with lower temperature
//! - [`task_planner`] -- planning and sub-agent delegation
//! - [`customer_support`] -- memory-enabled FAQ-focused agent
//! - [`code_reviewer`] -- filesystem tools with review-focused prompting
//!
//! # Example
//!
//! ```rust
//! use deepagents::presets::{PresetRegistry, PresetCustomizer};
//!
//! let registry = PresetRegistry::default();
//! let preset = registry.get("coding-assistant").unwrap();
//!
//! let customized = PresetCustomizer::from_preset(preset)
//!     .with_model("openai", "gpt-4")
//!     .with_temperature(0.5)
//!     .build();
//! ```

use std::collections::HashMap;

use crate::config::{AgentConfig, AgentConfigBuilder};

// ---------------------------------------------------------------------------
// AgentPreset
// ---------------------------------------------------------------------------

/// A pre-configured agent template for a specific use case.
#[derive(Debug, Clone)]
pub struct AgentPreset {
    /// Short identifier for this preset (e.g. `"coding-assistant"`).
    pub name: String,
    /// Human-readable description of what this preset is optimized for.
    pub description: String,
    /// The underlying agent configuration.
    pub config: AgentConfig,
    /// System prompt tailored for this preset's use case.
    pub system_prompt: String,
    /// Names of tools that are recommended for this preset.
    pub suggested_tools: Vec<String>,
    /// Arbitrary tags for categorization and filtering.
    pub tags: HashMap<String, String>,
}

impl AgentPreset {
    /// Create a new preset with the given name and description.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            config: AgentConfig::default(),
            system_prompt: String::new(),
            suggested_tools: Vec::new(),
            tags: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// PresetRegistry
// ---------------------------------------------------------------------------

/// A registry of named agent presets with tag-based filtering.
#[derive(Debug, Clone)]
pub struct PresetRegistry {
    presets: Vec<AgentPreset>,
}

impl PresetRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            presets: Vec::new(),
        }
    }

    /// Register a preset. If a preset with the same name exists, it is replaced.
    pub fn register(&mut self, preset: AgentPreset) {
        self.presets.retain(|p| p.name != preset.name);
        self.presets.push(preset);
    }

    /// Look up a preset by name.
    pub fn get(&self, name: &str) -> Option<&AgentPreset> {
        self.presets.iter().find(|p| p.name == name)
    }

    /// Return all registered presets.
    pub fn list(&self) -> Vec<&AgentPreset> {
        self.presets.iter().collect()
    }

    /// Return presets that have a tag with the given key (any value).
    pub fn list_by_tag(&self, tag: &str) -> Vec<&AgentPreset> {
        self.presets
            .iter()
            .filter(|p| p.tags.contains_key(tag))
            .collect()
    }
}

impl Default for PresetRegistry {
    /// Create a registry pre-populated with all built-in presets.
    fn default() -> Self {
        let mut registry = Self::new();
        registry.register(coding_assistant());
        registry.register(research_assistant());
        registry.register(data_analyst());
        registry.register(writing_assistant());
        registry.register(task_planner());
        registry.register(customer_support());
        registry.register(code_reviewer());
        registry
    }
}

// ---------------------------------------------------------------------------
// PresetCustomizer
// ---------------------------------------------------------------------------

/// Builder for modifying an existing preset before use.
#[derive(Debug, Clone)]
pub struct PresetCustomizer {
    preset: AgentPreset,
}

impl PresetCustomizer {
    /// Start customizing from an existing preset (cloned).
    pub fn from_preset(preset: &AgentPreset) -> Self {
        Self {
            preset: preset.clone(),
        }
    }

    /// Override the model provider and model name.
    pub fn with_model(mut self, provider: &str, model: &str) -> Self {
        self.preset.config.model.provider = provider.to_string();
        self.preset.config.model.model_name = model.to_string();
        self
    }

    /// Override the sampling temperature.
    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.preset.config.model.temperature = Some(temp);
        self
    }

    /// Override the suggested and configured tools.
    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.preset.suggested_tools = tools.clone();
        self.preset.config.tools = tools;
        self
    }

    /// Override the system prompt.
    pub fn with_system_prompt(mut self, prompt: &str) -> Self {
        self.preset.system_prompt = prompt.to_string();
        self.preset.config.system_prompt = Some(prompt.to_string());
        self
    }

    /// Consume the customizer and return the modified preset.
    pub fn build(self) -> AgentPreset {
        self.preset
    }
}

// ---------------------------------------------------------------------------
// Built-in preset factory functions
// ---------------------------------------------------------------------------

/// Preset optimized for code generation, review, and filesystem operations.
pub fn coding_assistant() -> AgentPreset {
    let config = AgentConfigBuilder::new()
        .name("coding-assistant")
        .description("Code generation and review agent")
        .system_prompt(
            "You are an expert software engineer. Write clean, well-documented, \
             and tested code. Use the filesystem tools to read and write files. \
             Follow best practices for the language being used.",
        )
        .temperature(0.2)
        .max_tokens(4096)
        .enable_filesystem(true)
        .enable_memory(true)
        .enable_summarization(false)
        .enable_planning(false)
        .tag("category", "development")
        .tag("tools", "filesystem")
        .build();

    let mut tags = HashMap::new();
    tags.insert("category".to_string(), "development".to_string());
    tags.insert("tools".to_string(), "filesystem".to_string());

    AgentPreset {
        name: "coding-assistant".to_string(),
        description: "Optimized for code generation, review, and filesystem operations.".to_string(),
        system_prompt: config.system_prompt.clone().unwrap_or_default(),
        suggested_tools: vec![
            "read_file".to_string(),
            "write_file".to_string(),
            "list_directory".to_string(),
            "grep".to_string(),
            "glob".to_string(),
        ],
        config,
        tags,
    }
}

/// Preset optimized for web research, summarization, and higher token limits.
pub fn research_assistant() -> AgentPreset {
    let config = AgentConfigBuilder::new()
        .name("research-assistant")
        .description("Web research and summarization agent")
        .system_prompt(
            "You are a meticulous research assistant. Search the web for information, \
             synthesize findings from multiple sources, and provide well-structured \
             summaries with citations. Always verify claims across sources.",
        )
        .temperature(0.3)
        .max_tokens(8192)
        .enable_filesystem(false)
        .enable_memory(true)
        .enable_summarization(true)
        .enable_planning(false)
        .max_context_tokens(32000)
        .tag("category", "research")
        .tag("tools", "web")
        .build();

    let mut tags = HashMap::new();
    tags.insert("category".to_string(), "research".to_string());
    tags.insert("tools".to_string(), "web".to_string());

    AgentPreset {
        name: "research-assistant".to_string(),
        description: "Web search and summarization with higher token limits.".to_string(),
        system_prompt: config.system_prompt.clone().unwrap_or_default(),
        suggested_tools: vec![
            "web_search".to_string(),
            "web_fetch".to_string(),
            "summarize".to_string(),
        ],
        config,
        tags,
    }
}

/// Preset optimized for structured data analysis.
pub fn data_analyst() -> AgentPreset {
    let config = AgentConfigBuilder::new()
        .name("data-analyst")
        .description("Structured data analysis agent")
        .system_prompt(
            "You are a data analyst. Analyze datasets, generate statistical summaries, \
             and produce structured output. Use code execution tools to run analysis \
             scripts. Present findings clearly with tables and charts when appropriate.",
        )
        .temperature(0.1)
        .max_tokens(4096)
        .enable_filesystem(true)
        .enable_memory(false)
        .enable_summarization(false)
        .enable_planning(true)
        .tag("category", "data")
        .tag("output", "structured")
        .build();

    let mut tags = HashMap::new();
    tags.insert("category".to_string(), "data".to_string());
    tags.insert("output".to_string(), "structured".to_string());

    AgentPreset {
        name: "data-analyst".to_string(),
        description: "Structured output and data analysis with code execution.".to_string(),
        system_prompt: config.system_prompt.clone().unwrap_or_default(),
        suggested_tools: vec![
            "python_repl".to_string(),
            "read_file".to_string(),
            "write_file".to_string(),
        ],
        config,
        tags,
    }
}

/// Preset optimized for creative and technical writing.
pub fn writing_assistant() -> AgentPreset {
    let config = AgentConfigBuilder::new()
        .name("writing-assistant")
        .description("Creative and technical writing agent")
        .system_prompt(
            "You are an expert writer. Produce clear, engaging, and well-structured \
             prose. Adapt your tone and style to the requested format — whether it is \
             a blog post, documentation, email, or creative fiction. Pay attention to \
             grammar, flow, and readability.",
        )
        .temperature(0.7)
        .max_tokens(4096)
        .enable_filesystem(false)
        .enable_memory(false)
        .enable_summarization(false)
        .enable_planning(false)
        .tag("category", "writing")
        .build();

    let mut tags = HashMap::new();
    tags.insert("category".to_string(), "writing".to_string());

    AgentPreset {
        name: "writing-assistant".to_string(),
        description: "Creative writing with lower temperature and no tools needed.".to_string(),
        system_prompt: config.system_prompt.clone().unwrap_or_default(),
        suggested_tools: Vec::new(),
        config,
        tags,
    }
}

/// Preset optimized for task decomposition, planning, and sub-agent delegation.
pub fn task_planner() -> AgentPreset {
    let config = AgentConfigBuilder::new()
        .name("task-planner")
        .description("Task planning and delegation agent")
        .system_prompt(
            "You are a task planner. Break complex tasks into smaller, actionable steps. \
             Create detailed execution plans, delegate sub-tasks to specialized agents, \
             and track progress. Ensure all dependencies between steps are identified.",
        )
        .temperature(0.3)
        .max_tokens(4096)
        .enable_filesystem(false)
        .enable_memory(true)
        .enable_summarization(true)
        .enable_planning(true)
        .tag("category", "planning")
        .tag("tools", "subagent")
        .build();

    let mut tags = HashMap::new();
    tags.insert("category".to_string(), "planning".to_string());
    tags.insert("tools".to_string(), "subagent".to_string());

    AgentPreset {
        name: "task-planner".to_string(),
        description: "Planning middleware with sub-agent delegation.".to_string(),
        system_prompt: config.system_prompt.clone().unwrap_or_default(),
        suggested_tools: vec![
            "create_plan".to_string(),
            "spawn_subagent".to_string(),
        ],
        config,
        tags,
    }
}

/// Preset optimized for customer support with memory and concise responses.
pub fn customer_support() -> AgentPreset {
    let config = AgentConfigBuilder::new()
        .name("customer-support")
        .description("Customer support agent with memory")
        .system_prompt(
            "You are a helpful customer support agent. Answer questions concisely and \
             accurately based on available knowledge. Remember previous interactions \
             with the customer. If you do not know the answer, say so and offer to \
             escalate. Always be polite and professional.",
        )
        .temperature(0.2)
        .max_tokens(2048)
        .enable_filesystem(false)
        .enable_memory(true)
        .enable_summarization(false)
        .enable_planning(false)
        .tag("category", "support")
        .tag("memory", "enabled")
        .build();

    let mut tags = HashMap::new();
    tags.insert("category".to_string(), "support".to_string());
    tags.insert("memory".to_string(), "enabled".to_string());

    AgentPreset {
        name: "customer-support".to_string(),
        description: "Memory-enabled agent for FAQ-focused customer support.".to_string(),
        system_prompt: config.system_prompt.clone().unwrap_or_default(),
        suggested_tools: vec![
            "knowledge_base_search".to_string(),
        ],
        config,
        tags,
    }
}

/// Preset optimized for code review with filesystem tools and structured output.
pub fn code_reviewer() -> AgentPreset {
    let config = AgentConfigBuilder::new()
        .name("code-reviewer")
        .description("Code review agent with structured feedback")
        .system_prompt(
            "You are an expert code reviewer. Read the provided code carefully and \
             provide structured feedback covering: correctness, performance, security, \
             readability, and adherence to best practices. Categorize each finding by \
             severity (critical, warning, suggestion). Always explain why something \
             is an issue and suggest a concrete fix.",
        )
        .temperature(0.1)
        .max_tokens(4096)
        .enable_filesystem(true)
        .enable_memory(false)
        .enable_summarization(false)
        .enable_planning(false)
        .tag("category", "development")
        .tag("output", "structured")
        .tag("tools", "filesystem")
        .build();

    let mut tags = HashMap::new();
    tags.insert("category".to_string(), "development".to_string());
    tags.insert("output".to_string(), "structured".to_string());
    tags.insert("tools".to_string(), "filesystem".to_string());

    AgentPreset {
        name: "code-reviewer".to_string(),
        description: "Filesystem tools with review-focused structured output.".to_string(),
        system_prompt: config.system_prompt.clone().unwrap_or_default(),
        suggested_tools: vec![
            "read_file".to_string(),
            "list_directory".to_string(),
            "grep".to_string(),
            "glob".to_string(),
        ],
        config,
        tags,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- AgentPreset construction tests --

    #[test]
    fn test_agent_preset_new() {
        let preset = AgentPreset::new("test", "A test preset");
        assert_eq!(preset.name, "test");
        assert_eq!(preset.description, "A test preset");
        assert!(preset.system_prompt.is_empty());
        assert!(preset.suggested_tools.is_empty());
        assert!(preset.tags.is_empty());
    }

    #[test]
    fn test_coding_assistant_preset() {
        let preset = coding_assistant();
        assert_eq!(preset.name, "coding-assistant");
        assert!(!preset.description.is_empty());
        assert!(!preset.system_prompt.is_empty());
        assert!(preset.suggested_tools.contains(&"read_file".to_string()));
        assert!(preset.suggested_tools.contains(&"write_file".to_string()));
        assert_eq!(preset.config.model.temperature, Some(0.2));
        assert!(preset.config.middleware.enable_filesystem);
        assert!(preset.tags.contains_key("category"));
        assert_eq!(preset.tags.get("category"), Some(&"development".to_string()));
    }

    #[test]
    fn test_research_assistant_preset() {
        let preset = research_assistant();
        assert_eq!(preset.name, "research-assistant");
        assert!(preset.config.middleware.enable_summarization);
        assert_eq!(preset.config.model.max_tokens, Some(8192));
        assert_eq!(preset.config.middleware.max_context_tokens, Some(32000));
        assert!(preset.suggested_tools.contains(&"web_search".to_string()));
        assert_eq!(preset.tags.get("category"), Some(&"research".to_string()));
    }

    #[test]
    fn test_data_analyst_preset() {
        let preset = data_analyst();
        assert_eq!(preset.name, "data-analyst");
        assert_eq!(preset.config.model.temperature, Some(0.1));
        assert!(preset.config.middleware.enable_planning);
        assert!(!preset.config.middleware.enable_memory);
        assert!(preset.suggested_tools.contains(&"python_repl".to_string()));
        assert_eq!(preset.tags.get("output"), Some(&"structured".to_string()));
    }

    #[test]
    fn test_writing_assistant_preset() {
        let preset = writing_assistant();
        assert_eq!(preset.name, "writing-assistant");
        assert_eq!(preset.config.model.temperature, Some(0.7));
        assert!(!preset.config.middleware.enable_filesystem);
        assert!(preset.suggested_tools.is_empty());
        assert_eq!(preset.tags.get("category"), Some(&"writing".to_string()));
    }

    #[test]
    fn test_task_planner_preset() {
        let preset = task_planner();
        assert_eq!(preset.name, "task-planner");
        assert!(preset.config.middleware.enable_planning);
        assert!(preset.config.middleware.enable_summarization);
        assert!(preset.suggested_tools.contains(&"spawn_subagent".to_string()));
        assert_eq!(preset.tags.get("tools"), Some(&"subagent".to_string()));
    }

    #[test]
    fn test_customer_support_preset() {
        let preset = customer_support();
        assert_eq!(preset.name, "customer-support");
        assert!(preset.config.middleware.enable_memory);
        assert_eq!(preset.config.model.temperature, Some(0.2));
        assert_eq!(preset.config.model.max_tokens, Some(2048));
        assert_eq!(preset.tags.get("memory"), Some(&"enabled".to_string()));
    }

    #[test]
    fn test_code_reviewer_preset() {
        let preset = code_reviewer();
        assert_eq!(preset.name, "code-reviewer");
        assert_eq!(preset.config.model.temperature, Some(0.1));
        assert!(preset.config.middleware.enable_filesystem);
        assert!(preset.suggested_tools.contains(&"grep".to_string()));
        assert!(preset.tags.contains_key("output"));
    }

    // -- PresetRegistry tests --

    #[test]
    fn test_registry_new_is_empty() {
        let registry = PresetRegistry::new();
        assert!(registry.list().is_empty());
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut registry = PresetRegistry::new();
        registry.register(coding_assistant());
        let preset = registry.get("coding-assistant");
        assert!(preset.is_some());
        assert_eq!(preset.unwrap().name, "coding-assistant");
    }

    #[test]
    fn test_registry_get_nonexistent() {
        let registry = PresetRegistry::new();
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_default_has_all_builtins() {
        let registry = PresetRegistry::default();
        let names: Vec<&str> = registry.list().iter().map(|p| p.name.as_str()).collect();
        assert!(names.contains(&"coding-assistant"));
        assert!(names.contains(&"research-assistant"));
        assert!(names.contains(&"data-analyst"));
        assert!(names.contains(&"writing-assistant"));
        assert!(names.contains(&"task-planner"));
        assert!(names.contains(&"customer-support"));
        assert!(names.contains(&"code-reviewer"));
        assert_eq!(registry.list().len(), 7);
    }

    #[test]
    fn test_registry_list_by_tag() {
        let registry = PresetRegistry::default();
        let dev_presets = registry.list_by_tag("category");
        // All built-in presets have a "category" tag
        assert_eq!(dev_presets.len(), 7);
    }

    #[test]
    fn test_registry_list_by_tag_specific() {
        let registry = PresetRegistry::default();
        let memory_presets = registry.list_by_tag("memory");
        assert_eq!(memory_presets.len(), 1);
        assert_eq!(memory_presets[0].name, "customer-support");
    }

    #[test]
    fn test_registry_list_by_tag_no_match() {
        let registry = PresetRegistry::default();
        let presets = registry.list_by_tag("nonexistent-tag");
        assert!(presets.is_empty());
    }

    #[test]
    fn test_registry_register_replaces_duplicate() {
        let mut registry = PresetRegistry::new();
        let mut preset1 = coding_assistant();
        preset1.description = "first".to_string();
        registry.register(preset1);

        let mut preset2 = coding_assistant();
        preset2.description = "second".to_string();
        registry.register(preset2);

        assert_eq!(registry.list().len(), 1);
        assert_eq!(registry.get("coding-assistant").unwrap().description, "second");
    }

    // -- PresetCustomizer tests --

    #[test]
    fn test_customizer_from_preset_clones() {
        let preset = coding_assistant();
        let customizer = PresetCustomizer::from_preset(&preset);
        let customized = customizer.build();
        assert_eq!(customized.name, preset.name);
        assert_eq!(customized.description, preset.description);
    }

    #[test]
    fn test_customizer_with_model() {
        let preset = coding_assistant();
        let customized = PresetCustomizer::from_preset(&preset)
            .with_model("openai", "gpt-4")
            .build();
        assert_eq!(customized.config.model.provider, "openai");
        assert_eq!(customized.config.model.model_name, "gpt-4");
    }

    #[test]
    fn test_customizer_with_temperature() {
        let preset = coding_assistant();
        let customized = PresetCustomizer::from_preset(&preset)
            .with_temperature(0.9)
            .build();
        assert_eq!(customized.config.model.temperature, Some(0.9));
    }

    #[test]
    fn test_customizer_with_tools() {
        let preset = coding_assistant();
        let tools = vec!["custom_tool".to_string()];
        let customized = PresetCustomizer::from_preset(&preset)
            .with_tools(tools.clone())
            .build();
        assert_eq!(customized.suggested_tools, tools);
        assert_eq!(customized.config.tools, tools);
    }

    #[test]
    fn test_customizer_with_system_prompt() {
        let preset = coding_assistant();
        let customized = PresetCustomizer::from_preset(&preset)
            .with_system_prompt("Be concise.")
            .build();
        assert_eq!(customized.system_prompt, "Be concise.");
        assert_eq!(customized.config.system_prompt, Some("Be concise.".to_string()));
    }

    #[test]
    fn test_customizer_chained() {
        let preset = writing_assistant();
        let customized = PresetCustomizer::from_preset(&preset)
            .with_model("anthropic", "claude-opus-4-20250514")
            .with_temperature(0.5)
            .with_system_prompt("Write poetry.")
            .with_tools(vec!["rhyme_finder".to_string()])
            .build();

        assert_eq!(customized.config.model.provider, "anthropic");
        assert_eq!(customized.config.model.model_name, "claude-opus-4-20250514");
        assert_eq!(customized.config.model.temperature, Some(0.5));
        assert_eq!(customized.system_prompt, "Write poetry.");
        assert_eq!(customized.suggested_tools, vec!["rhyme_finder"]);
    }

    #[test]
    fn test_customizer_does_not_mutate_original() {
        let preset = coding_assistant();
        let original_temp = preset.config.model.temperature;
        let _ = PresetCustomizer::from_preset(&preset)
            .with_temperature(0.99)
            .build();
        assert_eq!(preset.config.model.temperature, original_temp);
    }

    // -- Preset system prompt / config consistency --

    #[test]
    fn test_all_presets_have_system_prompt_in_config() {
        let presets = vec![
            coding_assistant(),
            research_assistant(),
            data_analyst(),
            writing_assistant(),
            task_planner(),
            customer_support(),
            code_reviewer(),
        ];
        for preset in &presets {
            assert!(
                preset.config.system_prompt.is_some(),
                "Preset '{}' should have a system prompt in its config",
                preset.name,
            );
            assert_eq!(
                preset.system_prompt,
                preset.config.system_prompt.clone().unwrap(),
                "Preset '{}' system_prompt field should match config.system_prompt",
                preset.name,
            );
        }
    }

    #[test]
    fn test_all_presets_have_nonempty_description() {
        let registry = PresetRegistry::default();
        for preset in registry.list() {
            assert!(
                !preset.description.is_empty(),
                "Preset '{}' should have a non-empty description",
                preset.name,
            );
        }
    }

    #[test]
    fn test_preset_config_name_matches_preset_name() {
        let registry = PresetRegistry::default();
        for preset in registry.list() {
            assert_eq!(
                preset.name, preset.config.name,
                "Preset name '{}' should match config name '{}'",
                preset.name, preset.config.name,
            );
        }
    }
}
