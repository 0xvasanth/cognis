//! Todo/planning middleware — task tracking for agent workflows.
//!
//! Provides a todo list that the agent can maintain during execution,
//! with a system prompt injection and parallel call detection.

use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use rustchain_core::error::Result;
use rustchain_core::messages::Message;

use super::types::{AgentMiddleware, AgentState, AsyncModelHandler, ModelCallResult, ModelRequest};

/// Status of a todo item.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum TodoStatus {
    /// The task has not been started.
    #[default]
    Pending,
    /// The task is currently being worked on.
    InProgress,
    /// The task has been completed.
    Completed,
}

/// A single todo item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Todo {
    /// Description of the task.
    pub content: String,
    /// Current status of the task.
    pub status: TodoStatus,
}

impl Todo {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            status: TodoStatus::Pending,
        }
    }

    pub fn with_status(mut self, status: TodoStatus) -> Self {
        self.status = status;
        self
    }

    /// Mark this todo as in progress.
    pub fn start(&mut self) {
        self.status = TodoStatus::InProgress;
    }

    /// Mark this todo as completed.
    pub fn complete(&mut self) {
        self.status = TodoStatus::Completed;
    }

    /// Check if this todo is completed.
    pub fn is_completed(&self) -> bool {
        self.status == TodoStatus::Completed
    }
}

/// Middleware that maintains a todo list and injects planning context.
///
/// Injects a system prompt with the current todo list before model calls,
/// and checks the model's response for parallel tool calls that might
/// indicate task planning behavior.
pub struct TodoListMiddleware {
    /// System prompt template for injecting the todo list.
    /// Use `{todos}` as a placeholder for the formatted todo list.
    pub system_prompt: String,
    /// The key in AgentState.extra where the todo list is stored.
    pub state_key: String,
}

impl TodoListMiddleware {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    pub fn with_state_key(mut self, key: impl Into<String>) -> Self {
        self.state_key = key.into();
        self
    }

    /// Get the current todo list from state.
    fn get_todos(&self, state: &AgentState) -> Vec<Todo> {
        state
            .extra
            .get(&self.state_key)
            .and_then(|v| serde_json::from_value::<Vec<Todo>>(v.clone()).ok())
            .unwrap_or_default()
    }

    /// Format the todo list for inclusion in a system prompt.
    fn format_todos(&self, todos: &[Todo]) -> String {
        if todos.is_empty() {
            return "No tasks in the todo list.".into();
        }

        let mut lines = Vec::new();
        for (i, todo) in todos.iter().enumerate() {
            let status_icon = match todo.status {
                TodoStatus::Pending => "[ ]",
                TodoStatus::InProgress => "[~]",
                TodoStatus::Completed => "[x]",
            };
            lines.push(format!("{}. {} {}", i + 1, status_icon, todo.content));
        }
        lines.join("\n")
    }

    /// Build the system prompt with the current todo list injected.
    fn build_system_prompt(&self, todos: &[Todo]) -> String {
        let formatted = self.format_todos(todos);
        self.system_prompt.replace("{todos}", &formatted)
    }

    /// Check the model response for signs of parallel tool calls,
    /// which might indicate the model is planning multiple tasks.
    fn check_parallel_calls(&self, state: &AgentState) -> Option<usize> {
        if let Some(Message::Ai(ai_msg)) = state.messages.last() {
            let call_count = ai_msg.tool_calls.len();
            if call_count > 1 {
                return Some(call_count);
            }
        }
        None
    }
}

impl Default for TodoListMiddleware {
    fn default() -> Self {
        Self {
            system_prompt: "You are working through a task list. Here is the current status of your tasks:\n\n\
                {todos}\n\n\
                Instructions for task management:\n\
                - Work through tasks in order, marking them as in_progress when you start and completed when done.\n\
                - If you discover sub-tasks, add them to the list.\n\
                - If a task is blocked, note the reason and move on to the next available task.\n\
                - Provide a brief status update when completing each task.\n\
                - Use parallel tool calls when tasks are independent and can be done simultaneously."
                .into(),
            state_key: "todo_list".into(),
        }
    }
}

#[async_trait]
impl AgentMiddleware for TodoListMiddleware {
    fn name(&self) -> &str {
        "TodoListMiddleware"
    }

    async fn wrap_model_call(
        &self,
        request: &ModelRequest,
        handler: &AsyncModelHandler,
    ) -> Result<ModelCallResult> {
        // Inject the todo list into the system prompt context
        let todos = self.get_todos(&request.state);
        let todo_prompt = self.build_system_prompt(&todos);

        // Build a new system message that combines any existing system prompt
        // with the todo context
        let new_system_message = if let Some(existing) = &request.system_message {
            let existing_text = existing.content().text();
            Message::system(format!("{}\n\n{}", existing_text, todo_prompt))
        } else {
            Message::system(todo_prompt)
        };

        // Construct a new request with the todo-aware system message
        let new_request = ModelRequest {
            model: request.model.clone(),
            messages: request.messages.clone(),
            system_message: Some(new_system_message),
            tool_choice: request.tool_choice.clone(),
            tools: request.tools.clone(),
            response_format: request.response_format.clone(),
            state: request.state.clone(),
            model_settings: request.model_settings.clone(),
        };

        let response = handler(&new_request).await?;
        Ok(ModelCallResult::Response(response))
    }

    async fn after_model(&self, state: &AgentState) -> Result<Option<HashMap<String, Value>>> {
        let mut updates = HashMap::new();

        // Check for parallel calls that might indicate planning behavior
        if let Some(call_count) = self.check_parallel_calls(state) {
            updates.insert("parallel_tool_calls".into(), serde_json::json!(call_count));
        }

        // Track the current todo completion status
        let todos = self.get_todos(state);
        if !todos.is_empty() {
            let completed = todos.iter().filter(|t| t.is_completed()).count();
            let total = todos.len();
            updates.insert(
                "todo_progress".into(),
                serde_json::json!({
                    "completed": completed,
                    "total": total,
                    "all_done": completed == total
                }),
            );
        }

        if updates.is_empty() {
            Ok(None)
        } else {
            Ok(Some(updates))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_todo_new() {
        let todo = Todo::new("Write tests");
        assert_eq!(todo.content, "Write tests");
        assert_eq!(todo.status, TodoStatus::Pending);
        assert!(!todo.is_completed());
    }

    #[test]
    fn test_todo_lifecycle() {
        let mut todo = Todo::new("Implement feature");
        assert_eq!(todo.status, TodoStatus::Pending);
        todo.start();
        assert_eq!(todo.status, TodoStatus::InProgress);
        todo.complete();
        assert_eq!(todo.status, TodoStatus::Completed);
        assert!(todo.is_completed());
    }

    #[test]
    fn test_todo_status_serde() {
        assert_eq!(
            serde_json::to_string(&TodoStatus::Pending).unwrap(),
            "\"pending\""
        );
        assert_eq!(
            serde_json::to_string(&TodoStatus::InProgress).unwrap(),
            "\"in_progress\""
        );
        assert_eq!(
            serde_json::to_string(&TodoStatus::Completed).unwrap(),
            "\"completed\""
        );
        let s: TodoStatus = serde_json::from_str("\"completed\"").unwrap();
        assert_eq!(s, TodoStatus::Completed);
    }

    #[test]
    fn test_todo_serde() {
        let todo = Todo::new("Test task").with_status(TodoStatus::InProgress);
        let json = serde_json::to_string(&todo).unwrap();
        let parsed: Todo = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.content, "Test task");
        assert_eq!(parsed.status, TodoStatus::InProgress);
    }

    #[test]
    fn test_format_todos_empty() {
        let mw = TodoListMiddleware::new();
        let formatted = mw.format_todos(&[]);
        assert_eq!(formatted, "No tasks in the todo list.");
    }

    #[test]
    fn test_format_todos_with_items() {
        let mw = TodoListMiddleware::new();
        let todos = vec![
            Todo::new("Task 1"),
            Todo::new("Task 2").with_status(TodoStatus::InProgress),
            Todo::new("Task 3").with_status(TodoStatus::Completed),
        ];
        let formatted = mw.format_todos(&todos);
        assert!(formatted.contains("[ ] Task 1"));
        assert!(formatted.contains("[~] Task 2"));
        assert!(formatted.contains("[x] Task 3"));
    }

    #[test]
    fn test_build_system_prompt() {
        let mw = TodoListMiddleware::new();
        let todos = vec![Todo::new("Do something")];
        let prompt = mw.build_system_prompt(&todos);
        assert!(prompt.contains("[ ] Do something"));
        assert!(prompt.contains("working through a task list"));
    }

    #[test]
    fn test_get_todos_from_state() {
        let mw = TodoListMiddleware::new();
        let mut state = AgentState::default();
        let todos = vec![Todo::new("A"), Todo::new("B")];
        state.set_extra("todo_list", serde_json::to_value(&todos).unwrap());
        let retrieved = mw.get_todos(&state);
        assert_eq!(retrieved.len(), 2);
        assert_eq!(retrieved[0].content, "A");
    }

    #[test]
    fn test_get_todos_empty_state() {
        let mw = TodoListMiddleware::new();
        let state = AgentState::default();
        let todos = mw.get_todos(&state);
        assert!(todos.is_empty());
    }

    #[test]
    fn test_middleware_name() {
        let mw = TodoListMiddleware::new();
        assert_eq!(mw.name(), "TodoListMiddleware");
    }

    #[test]
    fn test_middleware_builder() {
        let mw = TodoListMiddleware::new()
            .with_system_prompt("Custom: {todos}")
            .with_state_key("my_todos");
        assert_eq!(mw.system_prompt, "Custom: {todos}");
        assert_eq!(mw.state_key, "my_todos");
    }
}
