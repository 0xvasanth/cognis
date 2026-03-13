//! Integration tests for the cognisagent crate.
//!
//! All tests are self-contained and require no external services.
//! They use `FakeMessagesListChatModel` and mock tools to exercise
//! the agent, middleware, and backend systems end-to-end.
//!
//! Run with: `cargo test -p cognisagent -- integration`

mod integration {
    use std::collections::HashMap;
    use std::sync::Arc;

    use async_trait::async_trait;
    use serde_json::{json, Value};

    use cognisagent::agent::{create_deep_agent, DeepAgentError};
    use cognisagent::backends::filesystem::FilesystemBackend;
    use cognisagent::backends::state::StateBackend;
    use cognisagent::backends::Backend;
    use cognisagent::config::DeepAgentConfig;
    use cognisagent::middleware::memory::MemoryMiddleware;
    use cognisagent::middleware::patch_tool_calls::PatchToolCallsMiddleware;
    use cognisagent::middleware::skills::{Skill, SkillsMiddleware};
    use cognisagent::middleware::summarization::SummarizationMiddleware;
    use cognisagent::middleware::Middleware;

    use cognis_core::language_models::fake::FakeMessagesListChatModel;
    use cognis_core::messages::{AIMessage, Message, ToolCall};
    use cognis_core::tools::base::BaseTool;
    use cognis_core::tools::types::{ToolInput, ToolOutput};

    // -----------------------------------------------------------------------
    // Mock helpers
    // -----------------------------------------------------------------------

    /// A simple mock tool that returns a fixed result.
    struct MockTool {
        name: String,
        result: String,
    }

    impl MockTool {
        fn new(name: &str, result: &str) -> Self {
            Self {
                name: name.to_string(),
                result: result.to_string(),
            }
        }
    }

    #[async_trait]
    impl BaseTool for MockTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "A mock tool for testing"
        }

        async fn _run(&self, _input: ToolInput) -> cognis_core::error::Result<ToolOutput> {
            Ok(ToolOutput::Content(Value::String(self.result.clone())))
        }
    }

    /// A no-op middleware used to test default trait implementations and
    /// middleware chaining.
    struct NoopMiddleware;

    #[async_trait]
    impl Middleware for NoopMiddleware {
        fn name(&self) -> &str {
            "noop"
        }
    }

    // -------------------------------------------------------------------
    // 1. test_create_deep_agent_with_default_config
    // -------------------------------------------------------------------

    #[test]
    fn test_create_deep_agent_with_default_config() {
        let model = Arc::new(FakeMessagesListChatModel::new(vec![Message::Ai(
            AIMessage::new("hello"),
        )]));
        let config = DeepAgentConfig::default();

        let graph = create_deep_agent(model, config);
        assert!(graph.is_ok(), "Agent should compile with default config");

        let graph = graph.unwrap();
        let mut names = graph.node_names();
        names.sort();
        assert_eq!(names, vec!["agent", "tools"]);
    }

    // -------------------------------------------------------------------
    // 2. test_deep_agent_simple_conversation
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_deep_agent_simple_conversation() {
        let model = Arc::new(FakeMessagesListChatModel::new(vec![Message::Ai(
            AIMessage::new("The answer is 42"),
        )]));
        let tool: Arc<dyn BaseTool> = Arc::new(MockTool::new("dummy", "unused"));
        let config = DeepAgentConfig {
            tools: vec![tool],
            ..Default::default()
        };

        let graph = create_deep_agent(model, config).unwrap();
        let input = json!({
            "messages": [{"type": "human", "content": "What is the meaning of life?"}]
        });

        let result = graph.invoke(input).await.unwrap();
        let messages = result["messages"].as_array().unwrap();

        // human + ai response = 2 messages
        assert_eq!(messages.len(), 2);

        let last: Message = serde_json::from_value(messages.last().unwrap().clone()).unwrap();
        assert_eq!(last.content().text(), "The answer is 42");
    }

    // -------------------------------------------------------------------
    // 3. test_deep_agent_with_tool_call
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_deep_agent_with_tool_call() {
        // First response: AI makes a tool call.
        let tc = ToolCall {
            name: "calculator".to_string(),
            args: {
                let mut m = HashMap::new();
                m.insert("expression".to_string(), json!("6*7"));
                m
            },
            id: Some("call_1".to_string()),
        };
        let mut ai_with_tc = AIMessage::new("");
        ai_with_tc.tool_calls = vec![tc];

        // Second response: AI gives the final answer.
        let ai_final = AIMessage::new("The result is 42");

        let model = Arc::new(FakeMessagesListChatModel::new(vec![
            Message::Ai(ai_with_tc),
            Message::Ai(ai_final),
        ]));
        let tool: Arc<dyn BaseTool> = Arc::new(MockTool::new("calculator", "42"));
        let config = DeepAgentConfig {
            tools: vec![tool],
            ..Default::default()
        };

        let graph = create_deep_agent(model, config).unwrap();
        let input = json!({
            "messages": [{"type": "human", "content": "What is 6*7?"}]
        });

        let result = graph.invoke(input).await.unwrap();
        let messages = result["messages"].as_array().unwrap();

        // human + ai(tool_call) + tool_result + ai(final) = 4
        assert_eq!(messages.len(), 4);

        let final_msg: Message = serde_json::from_value(messages[3].clone()).unwrap();
        assert_eq!(final_msg.content().text(), "The result is 42");

        // Verify the tool result message is present.
        let tool_msg: Message = serde_json::from_value(messages[2].clone()).unwrap();
        assert!(matches!(tool_msg, Message::Tool(_)));
        assert!(tool_msg.content().text().contains("42"));
    }

    // -------------------------------------------------------------------
    // 4. test_deep_agent_with_system_prompt
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_deep_agent_with_system_prompt() {
        let model = Arc::new(FakeMessagesListChatModel::new(vec![Message::Ai(
            AIMessage::new("I am a helpful math tutor."),
        )]));
        let tool: Arc<dyn BaseTool> = Arc::new(MockTool::new("dummy", "unused"));
        let config = DeepAgentConfig::default()
            .with_system_prompt("You are a helpful math tutor.")
            .with_tool(tool);

        let graph = create_deep_agent(model, config).unwrap();
        let input = json!({
            "messages": [{"type": "human", "content": "Hello"}]
        });

        let result = graph.invoke(input).await.unwrap();
        let messages = result["messages"].as_array().unwrap();

        // The system prompt is injected during model invocation but not persisted
        // as a separate message in the state -- it is added transiently.
        // We verify the agent runs successfully and returns the AI response.
        let last: Message = serde_json::from_value(messages.last().unwrap().clone()).unwrap();
        assert_eq!(last.content().text(), "I am a helpful math tutor.");
    }

    // -------------------------------------------------------------------
    // 5. test_deep_agent_with_memory_middleware
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_deep_agent_with_memory_middleware() {
        let memory = Arc::new(MemoryMiddleware::new(10));
        memory.remember("user_name", "Alice").await;

        let model = Arc::new(FakeMessagesListChatModel::new(vec![Message::Ai(
            AIMessage::new("Hello Alice!"),
        )]));
        let tool: Arc<dyn BaseTool> = Arc::new(MockTool::new("dummy", "unused"));

        let config = DeepAgentConfig {
            tools: vec![tool],
            middleware: vec![memory.clone()],
            ..Default::default()
        };

        let graph = create_deep_agent(model, config).unwrap();
        let input = json!({
            "messages": [{"type": "human", "content": "What is my name?"}]
        });

        let result = graph.invoke(input).await.unwrap();
        let messages = result["messages"].as_array().unwrap();

        // The memory middleware injects a system message before model call.
        assert!(messages.len() >= 2);

        // Verify memory is still accessible after graph execution.
        assert_eq!(memory.recall("user_name").await, Some("Alice".to_string()));
    }

    // -------------------------------------------------------------------
    // 6. test_deep_agent_with_multiple_middleware
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_deep_agent_with_multiple_middleware() {
        let noop = Arc::new(NoopMiddleware) as Arc<dyn Middleware>;
        let memory = Arc::new(MemoryMiddleware::new(5));
        memory.remember("key", "value").await;

        let model = Arc::new(FakeMessagesListChatModel::new(vec![Message::Ai(
            AIMessage::new("Done"),
        )]));
        let tool: Arc<dyn BaseTool> = Arc::new(MockTool::new("dummy", "unused"));

        let config = DeepAgentConfig {
            tools: vec![tool],
            middleware: vec![noop, memory.clone()],
            ..Default::default()
        };

        let graph = create_deep_agent(model, config).unwrap();
        let input = json!({
            "messages": [{"type": "human", "content": "Hello"}]
        });

        let result = graph.invoke(input).await.unwrap();
        let messages = result["messages"].as_array().unwrap();
        assert!(messages.len() >= 2);

        let last: Message = serde_json::from_value(messages.last().unwrap().clone()).unwrap();
        assert_eq!(last.content().text(), "Done");
    }

    // -------------------------------------------------------------------
    // 7. test_memory_middleware_remember_recall
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_memory_middleware_remember_recall() {
        let mw = MemoryMiddleware::new(10);

        // Store entries.
        mw.remember("user_name", "Alice").await;
        mw.remember("preference", "dark mode").await;
        mw.remember("language", "Rust").await;

        // Recall existing keys.
        assert_eq!(mw.recall("user_name").await, Some("Alice".to_string()));
        assert_eq!(mw.recall("preference").await, Some("dark mode".to_string()));
        assert_eq!(mw.recall("language").await, Some("Rust".to_string()));

        // Recall non-existent key.
        assert_eq!(mw.recall("nonexistent").await, None);

        // Verify keys.
        let keys = mw.keys().await;
        assert_eq!(keys.len(), 3);

        // Overwrite an entry.
        mw.remember("user_name", "Bob").await;
        assert_eq!(mw.recall("user_name").await, Some("Bob".to_string()));

        // Clear all.
        mw.clear().await;
        assert_eq!(mw.keys().await.len(), 0);
        assert_eq!(mw.recall("user_name").await, None);
    }

    // -------------------------------------------------------------------
    // 8. test_patch_tool_calls_name_correction
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_patch_tool_calls_name_correction() {
        let mw = PatchToolCallsMiddleware::new(vec![
            "calculator".to_string(),
            "search".to_string(),
            "read_file".to_string(),
        ]);

        // Misspelled tool name: "calculater" -> "calculator".
        let mut state = json!({
            "messages": [
                { "type": "human", "content": "help" },
                {
                    "type": "ai",
                    "content": "",
                    "tool_calls": [
                        { "name": "calculater", "args": {"expr": "2+2"}, "id": "c1" }
                    ]
                }
            ]
        });

        mw.after_model(&mut state).await.unwrap();

        let tool_calls = state["messages"][1]["tool_calls"].as_array().unwrap();
        assert_eq!(
            tool_calls[0]["name"].as_str().unwrap(),
            "calculator",
            "Misspelled tool name should be corrected"
        );

        // Already correct name should remain unchanged.
        let mut state2 = json!({
            "messages": [
                {
                    "type": "ai",
                    "content": "",
                    "tool_calls": [
                        { "name": "search", "args": {"q": "hello"}, "id": "c2" }
                    ]
                }
            ]
        });

        mw.after_model(&mut state2).await.unwrap();
        let tool_calls2 = state2["messages"][0]["tool_calls"].as_array().unwrap();
        assert_eq!(tool_calls2[0]["name"].as_str().unwrap(), "search");

        // JSON repair: trailing comma in args stored as string.
        let mw_json = PatchToolCallsMiddleware::new(vec!["calculator".to_string()]);
        let mut state3 = json!({
            "messages": [
                {
                    "type": "ai",
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "calculator",
                            "args": "{\"expr\": \"2+2\", }"
                        }
                    ]
                }
            ]
        });

        mw_json.after_model(&mut state3).await.unwrap();
        let args = &state3["messages"][0]["tool_calls"][0]["args"];
        assert_eq!(args["expr"], "2+2", "Trailing comma should be repaired");
    }

    // -------------------------------------------------------------------
    // 9. test_state_backend_persistence
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_state_backend_persistence() {
        let backend = StateBackend::new();

        // Initially empty.
        let sessions = backend.list_sessions().await.unwrap();
        assert!(sessions.is_empty());

        let missing = backend.load_state("nonexistent").await.unwrap();
        assert!(missing.is_none());

        // Save a session.
        let state1 = json!({
            "messages": [
                {"type": "human", "content": "hello"}
            ],
            "metadata": {"turn": 1}
        });
        backend.save_state("session-1", &state1).await.unwrap();

        // Load it back.
        let loaded = backend.load_state("session-1").await.unwrap();
        assert_eq!(loaded, Some(state1.clone()));

        // Save another session.
        let state2 = json!({"messages": [], "count": 0});
        backend.save_state("session-2", &state2).await.unwrap();

        let sessions = backend.list_sessions().await.unwrap();
        assert_eq!(sessions.len(), 2);

        // Update the first session.
        let state1_updated = json!({
            "messages": [
                {"type": "human", "content": "hello"},
                {"type": "ai", "content": "hi there"}
            ],
            "metadata": {"turn": 2}
        });
        backend
            .save_state("session-1", &state1_updated)
            .await
            .unwrap();

        let loaded_updated = backend.load_state("session-1").await.unwrap();
        assert_eq!(loaded_updated, Some(state1_updated));
    }

    // -------------------------------------------------------------------
    // 10. test_filesystem_backend_round_trip
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_filesystem_backend_round_trip() {
        let tmpdir = tempfile::tempdir().unwrap();
        let backend = FilesystemBackend::new(tmpdir.path());

        // Initially empty.
        let sessions = backend.list_sessions().await.unwrap();
        assert!(sessions.is_empty());

        let missing = backend.load_state("no-such-session").await.unwrap();
        assert!(missing.is_none());

        // Save and load.
        let state = json!({
            "messages": [{"type": "human", "content": "hello"}],
            "count": 5
        });
        backend.save_state("session-abc", &state).await.unwrap();

        let loaded = backend.load_state("session-abc").await.unwrap();
        assert_eq!(loaded, Some(state.clone()));

        // Save a second session.
        let state2 = json!({"messages": [{"type": "ai", "content": "hi"}]});
        backend.save_state("session-def", &state2).await.unwrap();

        let mut sessions = backend.list_sessions().await.unwrap();
        sessions.sort();
        assert_eq!(
            sessions,
            vec!["session-abc".to_string(), "session-def".to_string()]
        );

        // Overwrite and verify.
        let state_updated = json!({"messages": [], "count": 10});
        backend
            .save_state("session-abc", &state_updated)
            .await
            .unwrap();
        let loaded_updated = backend.load_state("session-abc").await.unwrap();
        assert_eq!(loaded_updated, Some(state_updated));
    }

    // -------------------------------------------------------------------
    // 11. test_deep_agent_max_iterations
    // -------------------------------------------------------------------

    #[test]
    fn test_deep_agent_max_iterations() {
        // Verify max_iterations is stored in the config correctly.
        let config = DeepAgentConfig::default().with_max_iterations(3);
        assert_eq!(config.max_iterations, 3);

        let config2 = DeepAgentConfig::default();
        assert_eq!(config2.max_iterations, 25, "Default max_iterations is 25");

        // The agent should still compile with a custom max_iterations.
        let model = Arc::new(FakeMessagesListChatModel::new(vec![Message::Ai(
            AIMessage::new("done"),
        )]));
        let result = create_deep_agent(model, config);
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------
    // 12. test_config_builder_methods
    // -------------------------------------------------------------------

    #[test]
    fn test_config_builder_methods() {
        let tool1: Arc<dyn BaseTool> = Arc::new(MockTool::new("tool1", "r1"));
        let tool2: Arc<dyn BaseTool> = Arc::new(MockTool::new("tool2", "r2"));
        let memory_mw = Arc::new(MemoryMiddleware::new(5)) as Arc<dyn Middleware>;
        let fs_backend = Box::new(StateBackend::new()) as Box<dyn Backend>;

        let config = DeepAgentConfig::default()
            .with_model("gpt-4")
            .with_max_iterations(10)
            .with_system_prompt("You are a test agent.")
            .with_tool(tool1)
            .with_tool(tool2)
            .with_middleware(memory_mw)
            .with_backend(fs_backend);

        assert_eq!(config.model_name, "gpt-4");
        assert_eq!(config.max_iterations, 10);
        assert_eq!(
            config.system_prompt,
            Some("You are a test agent.".to_string())
        );
        assert_eq!(config.tools.len(), 2);
        assert_eq!(config.middleware.len(), 1);

        // with_tools replaces all tools.
        let tool3: Arc<dyn BaseTool> = Arc::new(MockTool::new("tool3", "r3"));
        let config2 = DeepAgentConfig::default().with_tools(vec![tool3]);
        assert_eq!(config2.tools.len(), 1);
        assert_eq!(config2.tools[0].name(), "tool3");
    }

    // -------------------------------------------------------------------
    // Additional integration tests
    // -------------------------------------------------------------------

    /// Verify that MemoryMiddleware injects context into state before model call.
    #[tokio::test]
    async fn test_memory_middleware_injects_context_into_state() {
        let mw = MemoryMiddleware::new(10);
        mw.remember("project", "cognis").await;
        mw.remember("user", "Bob").await;

        let mut state = json!({
            "messages": [
                {"type": "human", "content": "What project am I working on?"}
            ]
        });

        mw.before_model(&mut state).await.unwrap();

        let messages = state["messages"].as_array().unwrap();
        // Original 1 human message + 1 injected system message = 2
        assert_eq!(messages.len(), 2);

        let injected = &messages[0]; // Inserted at position 0 (no existing system msg)
        assert_eq!(injected["type"].as_str().unwrap(), "system");
        let content = injected["content"].as_str().unwrap();
        assert!(content.contains("Remembered context"));
    }

    /// Verify that MemoryMiddleware inserts after existing system prompt.
    #[tokio::test]
    async fn test_memory_middleware_inserts_after_system_prompt() {
        let mw = MemoryMiddleware::new(10);
        mw.remember("key", "value").await;

        let mut state = json!({
            "messages": [
                {"type": "system", "content": "You are a helpful assistant."},
                {"type": "human", "content": "Hello"}
            ]
        });

        mw.before_model(&mut state).await.unwrap();

        let messages = state["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        // System prompt still first.
        assert_eq!(
            messages[0]["content"].as_str().unwrap(),
            "You are a helpful assistant."
        );
        // Memory context inserted at index 1.
        assert!(messages[1]["content"]
            .as_str()
            .unwrap()
            .contains("Remembered context"));
        // Human message last.
        assert_eq!(messages[2]["content"].as_str().unwrap(), "Hello");
    }

    /// Verify empty memory does not inject anything.
    #[tokio::test]
    async fn test_memory_middleware_empty_is_noop() {
        let mw = MemoryMiddleware::new(10);

        let mut state = json!({
            "messages": [{"type": "human", "content": "Hello"}]
        });

        mw.before_model(&mut state).await.unwrap();

        let messages = state["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
    }

    /// Verify SummarizationMiddleware triggers when messages exceed threshold.
    #[tokio::test]
    async fn test_summarization_middleware_triggers_on_overflow() {
        let model = Arc::new(FakeMessagesListChatModel::new(vec![Message::Ai(
            AIMessage::new("Summary of earlier conversation."),
        )]));

        let mw = SummarizationMiddleware::new(model)
            .with_max_messages(3)
            .with_keep_recent(2);

        let mut state = json!({
            "messages": [
                {"type": "human", "content": "msg1"},
                {"type": "ai", "content": "resp1"},
                {"type": "human", "content": "msg2"},
                {"type": "ai", "content": "resp2"}
            ]
        });

        mw.before_model(&mut state).await.unwrap();

        let messages = state["messages"].as_array().unwrap();
        // 1 summary + 2 recent = 3
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["type"].as_str().unwrap(), "system");
        assert!(messages[0]["content"].as_str().unwrap().contains("Summary"));
    }

    /// Verify SkillsMiddleware injects skill listing and triggers.
    #[tokio::test]
    async fn test_skills_middleware_injects_listing() {
        let mut mw = SkillsMiddleware::new();
        mw.add_skill(Skill {
            name: "code_review".to_string(),
            description: "Review code changes".to_string(),
            instructions: "Look for bugs and style issues.".to_string(),
            trigger: Some("/review".to_string()),
        });

        let mut state = json!({
            "messages": [
                {"type": "human", "content": "Please /review my PR"}
            ]
        });

        mw.before_model(&mut state).await.unwrap();

        let messages = state["messages"].as_array().unwrap();
        // 1 original + 1 listing + 1 triggered instructions = 3
        assert_eq!(messages.len(), 3);

        let listing = messages[0]["content"].as_str().unwrap();
        assert!(listing.contains("Available Skills"));
        assert!(listing.contains("code_review"));

        let triggered = messages[1]["content"].as_str().unwrap();
        assert!(triggered.contains("Skill Instructions: code_review"));
        assert!(triggered.contains("Look for bugs"));
    }

    /// Verify the NoopMiddleware default implementations are truly no-ops.
    #[tokio::test]
    async fn test_noop_middleware_is_noop() {
        let mw = NoopMiddleware;
        assert_eq!(mw.name(), "noop");

        let mut state = json!({"messages": [{"type": "human", "content": "hello"}]});
        let original = state.clone();

        mw.before_model(&mut state).await.unwrap();
        assert_eq!(state, original);

        mw.after_model(&mut state).await.unwrap();
        assert_eq!(state, original);

        mw.before_tool(&mut state, "some_tool").await.unwrap();
        assert_eq!(state, original);

        mw.after_tool(&mut state, "some_tool", "result")
            .await
            .unwrap();
        assert_eq!(state, original);
    }

    /// Verify DeepAgentError variants format correctly.
    #[test]
    fn test_deep_agent_error_display() {
        let err = DeepAgentError::BackendError("disk full".to_string());
        assert_eq!(format!("{err}"), "backend error: disk full");

        let err = DeepAgentError::MiddlewareError("timeout".to_string());
        assert_eq!(format!("{err}"), "middleware error: timeout");

        let err = DeepAgentError::ConfigError("invalid model".to_string());
        assert_eq!(format!("{err}"), "config error: invalid model");

        let err = DeepAgentError::Other("something went wrong".to_string());
        assert_eq!(format!("{err}"), "something went wrong");
    }

    /// Verify FilesystemMiddleware provides the expected set of tools.
    #[test]
    fn test_filesystem_middleware_provides_tools() {
        use cognisagent::middleware::filesystem::FilesystemMiddleware;

        let mw = FilesystemMiddleware::new("/tmp/test");
        let tools = mw.tools();
        assert_eq!(tools.len(), 5);

        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"read_file"));
        assert!(names.contains(&"write_file"));
        assert!(names.contains(&"list_dir"));
        assert!(names.contains(&"glob"));
        assert!(names.contains(&"grep"));
    }
}
