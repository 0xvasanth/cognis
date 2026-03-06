use async_trait::async_trait;
use rustchain_core::agents::{AgentAction, AgentFinish};
use rustchain_core::callbacks::{CallbackHandler, CallbackManager};
use rustchain_core::error::Result;
use rustchain_core::outputs::LLMResult;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

/// A mock handler that records which callbacks were invoked.
struct MockHandler {
    events: Arc<Mutex<Vec<String>>>,
}

impl MockHandler {
    fn new(events: Arc<Mutex<Vec<String>>>) -> Self {
        Self { events }
    }
}

#[async_trait]
impl CallbackHandler for MockHandler {
    async fn on_llm_start(
        &self,
        _serialized: &Value,
        _prompts: &[String],
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.events.lock().unwrap().push("on_llm_start".into());
        Ok(())
    }

    async fn on_llm_new_token(
        &self,
        token: &str,
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.events
            .lock()
            .unwrap()
            .push(format!("on_llm_new_token:{}", token));
        Ok(())
    }

    async fn on_llm_end(
        &self,
        _response: &LLMResult,
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.events.lock().unwrap().push("on_llm_end".into());
        Ok(())
    }

    async fn on_agent_action(
        &self,
        action: &AgentAction,
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.events
            .lock()
            .unwrap()
            .push(format!("on_agent_action:{}", action.tool));
        Ok(())
    }

    async fn on_agent_finish(
        &self,
        _finish: &AgentFinish,
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.events.lock().unwrap().push("on_agent_finish".into());
        Ok(())
    }

    async fn on_text(
        &self,
        text: &str,
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.events
            .lock()
            .unwrap()
            .push(format!("on_text:{}", text));
        Ok(())
    }

    async fn on_retry(
        &self,
        _retry_state: &Value,
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.events.lock().unwrap().push("on_retry".into());
        Ok(())
    }

    async fn on_custom_event(
        &self,
        name: &str,
        _data: &Value,
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.events
            .lock()
            .unwrap()
            .push(format!("on_custom_event:{}", name));
        Ok(())
    }
}

#[tokio::test]
async fn callback_manager_dispatches_llm_events() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let handler = Arc::new(MockHandler::new(events.clone()));

    let mut manager = CallbackManager::new(vec![], None);
    manager.add_handler(handler, true);

    let run_id = Uuid::new_v4();
    manager
        .on_llm_start(&Value::Null, &["prompt".into()], run_id)
        .await
        .unwrap();
    manager.on_llm_new_token("Hello", run_id).await.unwrap();
    let result = LLMResult {
        generations: vec![],
        llm_output: None,
        run: None,
    };
    manager.on_llm_end(&result, run_id).await.unwrap();

    let recorded = events.lock().unwrap();
    assert_eq!(recorded.len(), 3);
    assert_eq!(recorded[0], "on_llm_start");
    assert_eq!(recorded[1], "on_llm_new_token:Hello");
    assert_eq!(recorded[2], "on_llm_end");
}

#[tokio::test]
async fn callback_manager_dispatches_to_multiple_handlers() {
    let events1 = Arc::new(Mutex::new(Vec::new()));
    let events2 = Arc::new(Mutex::new(Vec::new()));

    let mut manager = CallbackManager::new(vec![], None);
    manager.add_handler(Arc::new(MockHandler::new(events1.clone())), true);
    manager.add_handler(Arc::new(MockHandler::new(events2.clone())), true);

    let run_id = Uuid::new_v4();
    manager.on_text("hello", run_id).await.unwrap();

    assert_eq!(events1.lock().unwrap().len(), 1);
    assert_eq!(events2.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn callback_manager_agent_events() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let handler = Arc::new(MockHandler::new(events.clone()));

    let mut manager = CallbackManager::new(vec![], None);
    manager.add_handler(handler, true);

    let run_id = Uuid::new_v4();
    let action = AgentAction::new("search", Value::String("query".into()), "Searching...");
    manager.on_agent_action(&action, run_id).await.unwrap();

    let finish = AgentFinish::new(HashMap::new(), "Done");
    manager.on_agent_finish(&finish, run_id).await.unwrap();

    let recorded = events.lock().unwrap();
    assert_eq!(recorded[0], "on_agent_action:search");
    assert_eq!(recorded[1], "on_agent_finish");
}

#[tokio::test]
async fn callback_manager_remove_handler() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let handler = Arc::new(MockHandler::new(events.clone()));

    let mut manager = CallbackManager::new(vec![], None);
    manager.add_handler(handler, true);
    assert_eq!(manager.handlers().len(), 1);

    manager.remove_handler(0);
    assert_eq!(manager.handlers().len(), 0);
}

#[tokio::test]
async fn callback_manager_custom_event() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let handler = Arc::new(MockHandler::new(events.clone()));

    let mut manager = CallbackManager::new(vec![], None);
    manager.add_handler(handler, true);

    let run_id = Uuid::new_v4();
    manager
        .on_custom_event("my_event", &Value::Bool(true), run_id)
        .await
        .unwrap();

    let recorded = events.lock().unwrap();
    assert_eq!(recorded[0], "on_custom_event:my_event");
}

#[tokio::test]
async fn callback_manager_retry_event() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let handler = Arc::new(MockHandler::new(events.clone()));

    let mut manager = CallbackManager::new(vec![], None);
    manager.add_handler(handler, true);

    let run_id = Uuid::new_v4();
    manager
        .on_retry(&serde_json::json!({"attempt": 2}), run_id)
        .await
        .unwrap();

    let recorded = events.lock().unwrap();
    assert_eq!(recorded[0], "on_retry");
}
