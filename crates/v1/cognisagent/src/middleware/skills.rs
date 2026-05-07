//! Skills middleware — loads custom skill definitions and injects them into agent context.

use std::path::Path;

use async_trait::async_trait;
use serde_json::json;

use crate::middleware::{AgentState, Middleware, Result};

/// A skill definition that can be injected into the agent's context.
#[derive(Debug, Clone)]
pub struct Skill {
    /// The name of the skill.
    pub name: String,
    /// A short description of the skill.
    pub description: String,
    /// The full instructions/prompt for the skill.
    pub instructions: String,
    /// An optional trigger pattern (e.g. "/commit") that activates this skill.
    pub trigger: Option<String>,
}

/// Middleware that loads custom skill definitions and injects them into the
/// agent's context before each model call.
pub struct SkillsMiddleware {
    /// Loaded skill definitions.
    skills: Vec<Skill>,
}

impl SkillsMiddleware {
    /// Create a new `SkillsMiddleware` with an empty skills list.
    pub fn new() -> Self {
        Self { skills: Vec::new() }
    }

    /// Add a skill definition. Returns `&mut Self` for chaining.
    pub fn add_skill(&mut self, skill: Skill) -> &mut Self {
        self.skills.push(skill);
        self
    }

    /// Load skills from a directory of `.md` files.
    ///
    /// Each markdown file is parsed as:
    /// - File name (without extension) = skill name
    /// - First line (stripped of `# ` prefix) = description
    /// - Remaining lines = instructions
    pub fn load_from_dir(dir: &Path) -> std::result::Result<Self, std::io::Error> {
        let mut middleware = Self::new();

        let entries = std::fs::read_dir(dir)?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|e| e.to_str()) != Some("md") {
                continue;
            }

            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();

            let content = std::fs::read_to_string(&path)?;
            let mut lines = content.lines();

            let first_line = lines.next().unwrap_or("");
            let description = first_line
                .strip_prefix("# ")
                .unwrap_or(first_line)
                .to_string();
            let instructions = lines
                .collect::<Vec<_>>()
                .join("\n")
                .trim_start()
                .to_string();

            middleware.add_skill(Skill {
                name,
                description,
                instructions,
                trigger: None,
            });
        }

        Ok(middleware)
    }

    /// Return a reference to the loaded skills.
    pub fn skills(&self) -> &[Skill] {
        &self.skills
    }
}

impl Default for SkillsMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Middleware for SkillsMiddleware {
    fn name(&self) -> &str {
        "skills"
    }

    /// Before the model is called, inject a system message listing available skills.
    /// If a skill's trigger matches content in the last user message, inject that
    /// skill's full instructions as an additional system message.
    async fn before_model(&self, state: &mut AgentState) -> Result<()> {
        if self.skills.is_empty() {
            return Ok(());
        }

        let messages = match state.get_mut("messages").and_then(|v| v.as_array_mut()) {
            Some(m) => m,
            None => return Ok(()),
        };

        // Build skill listing.
        let listing: Vec<String> = self
            .skills
            .iter()
            .map(|s| {
                let trigger_info = s
                    .trigger
                    .as_ref()
                    .map(|t| format!(" (trigger: {t})"))
                    .unwrap_or_default();
                format!("- **{}**: {}{}", s.name, s.description, trigger_info)
            })
            .collect();

        let listing_msg = json!({
            "type": "system",
            "content": format!("## Available Skills\n{}", listing.join("\n"))
        });

        // Determine the last user message content for trigger matching.
        let last_user_content = messages
            .iter()
            .rev()
            .find(|m| m.get("type").and_then(|t| t.as_str()) == Some("human"))
            .and_then(|m| m.get("content").and_then(|c| c.as_str()))
            .unwrap_or("")
            .to_string();

        // Check for triggered skills and collect their instructions.
        let mut triggered_msgs: Vec<serde_json::Value> = Vec::new();
        for skill in &self.skills {
            if let Some(trigger) = &skill.trigger {
                if last_user_content.contains(trigger.as_str()) {
                    triggered_msgs.push(json!({
                        "type": "system",
                        "content": format!(
                            "## Skill Instructions: {}\n{}",
                            skill.name, skill.instructions
                        )
                    }));
                }
            }
        }

        // Insert the listing (and any triggered instructions) after the first system message.
        let insert_pos = if messages
            .first()
            .and_then(|m| m.get("type"))
            .and_then(|t| t.as_str())
            == Some("system")
        {
            1
        } else {
            0
        };

        // Insert in order: listing first, then triggered skill instructions.
        let mut offset = 0;
        messages.insert(insert_pos + offset, listing_msg);
        offset += 1;
        for msg in triggered_msgs {
            messages.insert(insert_pos + offset, msg);
            offset += 1;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    #[test]
    fn test_load_from_dir() {
        let dir = TempDir::new().unwrap();

        std::fs::write(
            dir.path().join("commit.md"),
            "# Git commit helper\nRun git commit with a good message.\nAlways sign commits.",
        )
        .unwrap();

        std::fs::write(
            dir.path().join("review.md"),
            "# Code review\nReview the code carefully.\nCheck for bugs.",
        )
        .unwrap();

        // Non-md file should be ignored.
        std::fs::write(dir.path().join("notes.txt"), "some notes").unwrap();

        let mw = SkillsMiddleware::load_from_dir(dir.path()).unwrap();
        assert_eq!(mw.skills().len(), 2);

        let names: Vec<&str> = mw.skills().iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"commit"));
        assert!(names.contains(&"review"));

        let commit_skill = mw.skills().iter().find(|s| s.name == "commit").unwrap();
        assert_eq!(commit_skill.description, "Git commit helper");
        assert!(commit_skill.instructions.contains("Always sign commits"));
    }

    #[tokio::test]
    async fn test_before_model_injects_listing() {
        let mut mw = SkillsMiddleware::new();
        mw.add_skill(Skill {
            name: "test_skill".to_string(),
            description: "A test skill".to_string(),
            instructions: "Do testing things.".to_string(),
            trigger: None,
        });

        let mut state = json!({
            "messages": [
                { "type": "human", "content": "Hello" }
            ]
        });

        mw.before_model(&mut state).await.unwrap();

        let messages = state["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);

        let injected = &messages[0];
        assert_eq!(injected["type"], "system");
        let content = injected["content"].as_str().unwrap();
        assert!(content.contains("Available Skills"));
        assert!(content.contains("test_skill"));
        assert!(content.contains("A test skill"));
    }

    #[tokio::test]
    async fn test_trigger_injects_full_instructions() {
        let mut mw = SkillsMiddleware::new();
        mw.add_skill(Skill {
            name: "commit".to_string(),
            description: "Git commit helper".to_string(),
            instructions: "Always write clear commit messages.".to_string(),
            trigger: Some("/commit".to_string()),
        });
        mw.add_skill(Skill {
            name: "review".to_string(),
            description: "Code review".to_string(),
            instructions: "Review all changes carefully.".to_string(),
            trigger: Some("/review".to_string()),
        });

        let mut state = json!({
            "messages": [
                { "type": "system", "content": "You are an assistant." },
                { "type": "human", "content": "Please /commit my changes" }
            ]
        });

        mw.before_model(&mut state).await.unwrap();

        let messages = state["messages"].as_array().unwrap();
        // Original 2 + listing + 1 triggered skill = 4
        assert_eq!(messages.len(), 4);

        // The listing should be at index 1 (after original system msg).
        assert!(messages[1]["content"]
            .as_str()
            .unwrap()
            .contains("Available Skills"));

        // The triggered skill instructions at index 2.
        let triggered = messages[2]["content"].as_str().unwrap();
        assert!(triggered.contains("Skill Instructions: commit"));
        assert!(triggered.contains("Always write clear commit messages"));

        // The /review skill should NOT be triggered.
        let all_content: String = messages
            .iter()
            .filter_map(|m| m["content"].as_str())
            .collect();
        assert!(!all_content.contains("Skill Instructions: review"));
    }
}
