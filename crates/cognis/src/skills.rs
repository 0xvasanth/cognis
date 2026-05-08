//! Skills — coherent bundles of (system-prompt fragment, tools,
//! activation predicate) that the agent installs at build time.
//!
//! A `Skill` is more focused than an `AgentPlugin`: a plugin mutates an
//! `AgentBuilder` arbitrarily, while a skill makes a specific contract
//! — "I provide these tools, this prompt fragment, and I know how to
//! decide whether I'm relevant for a given input."
//!
//! Customization:
//! - Implement [`Skill`] for a fully custom skill type.
//! - Use [`SkillBuilder`] for closure-assembled skills.
//! - [`SkillSelector`] is the matcher trait — default `AllSkills` always
//!   activates everything; users plug in keyword / classifier / LLM-based
//!   selectors.

use std::sync::Arc;

use cognis_llm::tools::Tool;

/// One installed skill.
pub trait Skill: Send + Sync {
    /// Skill identifier (e.g. `"web-search"`, `"sql"`).
    fn id(&self) -> &str;

    /// Human-readable description (used by [`SkillSelector`]
    /// implementations and surfaced to LLMs that route based on skill
    /// metadata).
    fn description(&self) -> &str;

    /// Tools provided by this skill.
    fn tools(&self) -> Vec<Arc<dyn Tool>>;

    /// Optional system-prompt fragment appended when the skill is active.
    fn system_prompt_fragment(&self) -> Option<String> {
        None
    }
}

// ---------------------------------------------------------------------------
// SkillBuilder — assemble a skill from a closure.
// ---------------------------------------------------------------------------

type ToolFactory = Arc<dyn Fn() -> Vec<Arc<dyn Tool>> + Send + Sync>;

/// Closure-assembled skill.
pub struct BuiltSkill {
    id: String,
    description: String,
    system_prompt: Option<String>,
    tools_fn: ToolFactory,
}

impl Skill for BuiltSkill {
    fn id(&self) -> &str {
        &self.id
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn tools(&self) -> Vec<Arc<dyn Tool>> {
        (self.tools_fn)()
    }
    fn system_prompt_fragment(&self) -> Option<String> {
        self.system_prompt.clone()
    }
}

/// Fluent builder for [`BuiltSkill`].
pub struct SkillBuilder {
    id: String,
    description: String,
    system_prompt: Option<String>,
    tools_fn: Option<ToolFactory>,
}

impl SkillBuilder {
    /// New skill with `id` and `description`.
    pub fn new(id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            system_prompt: None,
            tools_fn: None,
        }
    }

    /// Set the system-prompt fragment.
    pub fn with_system_prompt(mut self, fragment: impl Into<String>) -> Self {
        self.system_prompt = Some(fragment.into());
        self
    }

    /// Provide a tool factory (called each time the skill activates).
    pub fn with_tools<F>(mut self, factory: F) -> Self
    where
        F: Fn() -> Vec<Arc<dyn Tool>> + Send + Sync + 'static,
    {
        self.tools_fn = Some(Arc::new(factory));
        self
    }

    /// Provide a static tool list.
    pub fn with_static_tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        let cloned = tools;
        self.tools_fn = Some(Arc::new(move || cloned.clone()));
        self
    }

    /// Build.
    pub fn build(self) -> BuiltSkill {
        BuiltSkill {
            id: self.id,
            description: self.description,
            system_prompt: self.system_prompt,
            tools_fn: self.tools_fn.unwrap_or_else(|| Arc::new(Vec::new)),
        }
    }
}

// ---------------------------------------------------------------------------
// SkillSelector — pluggable activation policy.
// ---------------------------------------------------------------------------

/// Decides which skills activate for a given input. Returns indices
/// into the supplied skills slice — using indices avoids variance issues
/// that arise with `&'a Arc<dyn Skill>` return types and keeps the
/// trait object-safe.
///
/// The default [`AllSkills`] activates every registered skill; users
/// plug in keyword / classifier / LLM-based policies for selective
/// activation.
pub trait SkillSelector: Send + Sync {
    /// Indices of skills (within `skills`) to activate.
    fn select(&self, input: &str, skills: &[Arc<dyn Skill>]) -> Vec<usize>;
}

/// Activate every registered skill regardless of input.
#[derive(Debug, Default, Clone, Copy)]
pub struct AllSkills;

impl SkillSelector for AllSkills {
    fn select(&self, _input: &str, skills: &[Arc<dyn Skill>]) -> Vec<usize> {
        (0..skills.len()).collect()
    }
}

/// Activate skills whose id or description contains any of the
/// configured keywords (case-insensitive).
pub struct KeywordSelector {
    keywords: Vec<String>,
}

impl KeywordSelector {
    /// Build from a list of keywords.
    pub fn new<I, S>(keywords: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            keywords: keywords
                .into_iter()
                .map(|s| s.into().to_lowercase())
                .collect(),
        }
    }
}

impl SkillSelector for KeywordSelector {
    fn select(&self, input: &str, skills: &[Arc<dyn Skill>]) -> Vec<usize> {
        let lower = input.to_lowercase();
        if !self.keywords.iter().any(|k| lower.contains(k)) {
            return Vec::new();
        }
        (0..skills.len()).collect()
    }
}

/// Closure-based selector.
impl<F> SkillSelector for F
where
    F: Fn(&str, &[Arc<dyn Skill>]) -> Vec<usize> + Send + Sync,
{
    fn select(&self, input: &str, skills: &[Arc<dyn Skill>]) -> Vec<usize> {
        (self)(input, skills)
    }
}

// ---------------------------------------------------------------------------
// SkillRegistry
// ---------------------------------------------------------------------------

/// In-process skill catalog. The agent builder consults this at build
/// time to materialize tools + prompt fragments.
#[derive(Default, Clone)]
pub struct SkillRegistry {
    skills: Vec<Arc<dyn Skill>>,
}

impl SkillRegistry {
    /// Empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a skill.
    pub fn register(mut self, skill: impl Skill + 'static) -> Self {
        self.skills.push(Arc::new(skill));
        self
    }

    /// Append a pre-boxed skill (for users keeping their own `Arc`s).
    pub fn register_arc(mut self, skill: Arc<dyn Skill>) -> Self {
        self.skills.push(skill);
        self
    }

    /// All registered skills (read-only).
    pub fn skills(&self) -> &[Arc<dyn Skill>] {
        &self.skills
    }

    /// Filter via a [`SkillSelector`] for the given input. Returns the
    /// active skills as borrowed `Arc` references.
    pub fn active_for<S: SkillSelector + ?Sized>(
        &self,
        selector: &S,
        input: &str,
    ) -> Vec<&Arc<dyn Skill>> {
        let idxs = selector.select(input, &self.skills);
        idxs.into_iter()
            .filter_map(|i| self.skills.get(i))
            .collect()
    }

    /// Materialize all tools across active skills.
    pub fn collect_tools<'a, I>(active: I) -> Vec<Arc<dyn Tool>>
    where
        I: IntoIterator<Item = &'a Arc<dyn Skill>>,
    {
        let mut out = Vec::new();
        for s in active {
            out.extend(s.tools());
        }
        out
    }

    /// Concatenate the system-prompt fragments of `active` skills,
    /// separated by `\n\n`. Empty if no fragments.
    pub fn collect_prompt<'a, I>(active: I) -> String
    where
        I: IntoIterator<Item = &'a Arc<dyn Skill>>,
    {
        let mut parts: Vec<String> = Vec::new();
        for s in active {
            if let Some(p) = s.system_prompt_fragment() {
                parts.push(p);
            }
        }
        parts.join("\n\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use cognis_core::Result;
    use cognis_llm::tools::{Tool, ToolInput, ToolOutput};

    /// Trivial tool used in tests.
    struct NoopTool(&'static str);

    #[async_trait]
    impl Tool for NoopTool {
        fn name(&self) -> &str {
            self.0
        }
        fn description(&self) -> &str {
            "noop"
        }
        fn args_schema(&self) -> Option<serde_json::Value> {
            None
        }
        async fn _run(&self, _: ToolInput) -> Result<ToolOutput> {
            Ok(ToolOutput::Empty)
        }
    }

    #[test]
    fn skill_builder_defaults() {
        let s: BuiltSkill = SkillBuilder::new("web", "browse the web")
            .with_system_prompt("you may browse the web")
            .with_static_tools(vec![Arc::new(NoopTool("search"))])
            .build();
        assert_eq!(s.id(), "web");
        assert_eq!(s.tools().len(), 1);
        assert_eq!(
            s.system_prompt_fragment().as_deref(),
            Some("you may browse the web")
        );
    }

    #[test]
    fn registry_collects_active_tools_and_prompts() {
        let reg = SkillRegistry::new()
            .register(
                SkillBuilder::new("a", "skill a")
                    .with_system_prompt("PA")
                    .with_static_tools(vec![Arc::new(NoopTool("ta"))])
                    .build(),
            )
            .register(
                SkillBuilder::new("b", "skill b")
                    .with_system_prompt("PB")
                    .with_static_tools(vec![Arc::new(NoopTool("tb"))])
                    .build(),
            );
        let active = reg.active_for(&AllSkills, "anything");
        assert_eq!(active.len(), 2);
        let tools = SkillRegistry::collect_tools(active.iter().copied());
        assert_eq!(tools.len(), 2);
        let prompt = SkillRegistry::collect_prompt(active.iter().copied());
        assert!(prompt.contains("PA"));
        assert!(prompt.contains("PB"));
    }

    #[test]
    fn keyword_selector_activates_only_on_match() {
        let reg = SkillRegistry::new().register(
            SkillBuilder::new("sql", "database queries")
                .with_static_tools(vec![Arc::new(NoopTool("query"))])
                .build(),
        );
        let sel = KeywordSelector::new(["sql", "database"]);
        assert_eq!(reg.active_for(&sel, "what is rust").len(), 0);
        assert_eq!(reg.active_for(&sel, "run a SQL query").len(), 1);
    }

    #[test]
    fn closure_selector_works() {
        let reg = SkillRegistry::new().register(
            SkillBuilder::new("a", "always-on")
                .with_static_tools(vec![Arc::new(NoopTool("t"))])
                .build(),
        );
        // Match only when input contains the digit '1'.
        let sel = |input: &str, skills: &[Arc<dyn Skill>]| -> Vec<usize> {
            if input.contains('1') {
                (0..skills.len()).collect()
            } else {
                Vec::new()
            }
        };
        assert!(reg.active_for(&sel, "no digit here").is_empty());
        assert_eq!(reg.active_for(&sel, "step 1 done").len(), 1);
    }
}
