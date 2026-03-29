//! Todo list middleware for tracking agent task progress.
//!
//! Injects the current todo list into the system prompt before each model
//! call, giving the agent awareness of pending, in-progress, and completed
//! tasks.

use async_trait::async_trait;
use serde_json::{json, Value};

use super::Middleware;
use crate::agent::DeepAgentError;

/// Middleware that maintains and displays a todo list during agent execution.
///
/// Reads todos from `state["todos"]` (a JSON array of objects with `content`
/// and `status` fields) and prepends a system message listing them before
/// each model call.
///
/// Status values: `"pending"`, `"in_progress"`, `"completed"`.
pub struct TodoListMiddleware {
    /// Key in state where todos are stored.
    state_key: String,
}

impl TodoListMiddleware {
    /// Create a new middleware using the default `"todos"` state key.
    pub fn new() -> Self {
        Self {
            state_key: "todos".to_string(),
        }
    }

    /// Create with a custom state key.
    pub fn with_key(key: impl Into<String>) -> Self {
        Self {
            state_key: key.into(),
        }
    }

    /// Format a list of todo items into a readable string with status icons.
    pub fn format_todos(todos: &[Value]) -> String {
        let mut lines = Vec::new();
        for (i, todo) in todos.iter().enumerate() {
            let content = todo
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("(untitled)");
            let status = todo
                .get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("pending");
            let icon = match status {
                "completed" => "[x]",
                "in_progress" => "[~]",
                _ => "[ ]",
            };
            lines.push(format!("{}. {} {}", i + 1, icon, content));
        }
        lines.join("\n")
    }
}

impl Default for TodoListMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Middleware for TodoListMiddleware {
    fn name(&self) -> &str {
        "todo_list"
    }

    async fn before_model(&self, state: &mut Value) -> std::result::Result<(), DeepAgentError> {
        let todos = match state.get(&self.state_key).and_then(|v| v.as_array()) {
            Some(arr) if !arr.is_empty() => arr.clone(),
            _ => return Ok(()),
        };

        let formatted = Self::format_todos(&todos);
        let system_content = format!(
            "Current task list:\n{}\n\nUpdate task status as you work through them.",
            formatted
        );

        if let Some(messages) = state.get_mut("messages").and_then(|v| v.as_array_mut()) {
            let todo_msg = json!({
                "type": "system",
                "content": system_content
            });

            // Replace an existing todo-list system message if present,
            // otherwise insert a new one. This prevents accumulation
            // across multiple model calls. Check both type and content.
            let is_todo_system = messages.first().is_some_and(|m| {
                m.get("type").and_then(|t| t.as_str()) == Some("system")
                    && m.get("content")
                        .and_then(|c| c.as_str())
                        .is_some_and(|c| c.starts_with("Current task list:"))
            });

            if is_todo_system {
                messages[0] = todo_msg;
            } else {
                messages.insert(0, todo_msg);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_todo_middleware_injects_system_prompt() {
        let middleware = TodoListMiddleware::new();
        let mut state = json!({
            "messages": [
                {"type": "human", "content": "Plan a project"}
            ],
            "todos": [
                {"content": "Design API", "status": "pending"},
                {"content": "Write tests", "status": "completed"}
            ]
        });

        middleware.before_model(&mut state).await.unwrap();

        let messages = state["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["type"], "system");
        let content = messages[0]["content"].as_str().unwrap();
        assert!(content.contains("Design API"));
        assert!(content.contains("Write tests"));
        assert!(content.contains("[ ]"));
        assert!(content.contains("[x]"));
    }

    #[tokio::test]
    async fn test_todo_middleware_no_todos_no_injection() {
        let middleware = TodoListMiddleware::new();
        let mut state = json!({
            "messages": [
                {"type": "human", "content": "Hello"}
            ]
        });

        middleware.before_model(&mut state).await.unwrap();

        let messages = state["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
    }

    #[test]
    fn test_format_todos() {
        let todos = vec![
            json!({"content": "Task A", "status": "completed"}),
            json!({"content": "Task B", "status": "in_progress"}),
            json!({"content": "Task C", "status": "pending"}),
        ];
        let formatted = TodoListMiddleware::format_todos(&todos);
        assert!(formatted.contains("[x] Task A"));
        assert!(formatted.contains("[~] Task B"));
        assert!(formatted.contains("[ ] Task C"));
    }

    #[tokio::test]
    async fn test_todo_middleware_custom_key() {
        let middleware = TodoListMiddleware::with_key("tasks");
        let mut state = json!({
            "messages": [{"type": "human", "content": "Go"}],
            "tasks": [{"content": "Do thing", "status": "pending"}]
        });

        middleware.before_model(&mut state).await.unwrap();

        let messages = state["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert!(messages[0]["content"]
            .as_str()
            .unwrap()
            .contains("Do thing"));
    }

    #[tokio::test]
    async fn test_todo_middleware_empty_todos_no_injection() {
        let middleware = TodoListMiddleware::new();
        let mut state = json!({
            "messages": [{"type": "human", "content": "Hi"}],
            "todos": []
        });

        middleware.before_model(&mut state).await.unwrap();
        assert_eq!(state["messages"].as_array().unwrap().len(), 1);
    }
}
