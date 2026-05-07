//! Multi-agent orchestration — dispatch a single user request across
//! a fleet of agents and merge their outputs via a pluggable handoff
//! strategy.
//!
//! Three strategies ship in the box; users plug in custom ones via the
//! [`HandoffStrategy`] trait:
//!
//! - [`Sequential`] — call agents in order; each receives the previous
//!   agent's output as additional context. Good for "research → plan →
//!   write" pipelines.
//! - [`Supervisor`] — first agent (the supervisor) routes to one of the
//!   remaining agents based on its own response. Good for triage.
//! - [`ParallelVote`] — call all agents in parallel and pick the answer
//!   that the most agents agree on (string-equality vote). Good for
//!   reasoning ensembles.
//!
//! Inter-agent communication uses [`AgentMessage`], a typed envelope
//! carrying source/target agent ids and payload. The orchestrator
//! exposes a [`MessageBus`] trait so users can swap in their own
//! transport (Redis pub-sub, a queue, etc.).

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use cognis_core::{CognisError, Message, Result};

use crate::agent::{Agent, AgentResponse};

/// Envelope for inter-agent messages.
#[derive(Debug, Clone)]
pub struct AgentMessage {
    /// Source agent id (or `"user"` / `"system"` for non-agent senders).
    pub from: String,
    /// Destination agent id.
    pub to: String,
    /// Payload — usually the previous agent's reply.
    pub content: Message,
    /// Free-form metadata bag for custom routing decisions.
    pub metadata: serde_json::Value,
}

/// Pluggable inter-agent transport. Stock impl: [`InMemoryMessageBus`].
#[async_trait]
pub trait MessageBus: Send + Sync {
    /// Publish a message; returns once accepted by the bus.
    async fn publish(&self, msg: AgentMessage) -> Result<()>;
    /// Drain every message addressed to `agent_id` since the last drain.
    async fn drain(&self, agent_id: &str) -> Result<Vec<AgentMessage>>;
}

/// In-memory bus — single-process, lossless, no persistence.
#[derive(Default)]
pub struct InMemoryMessageBus {
    inboxes: Mutex<HashMap<String, Vec<AgentMessage>>>,
}

impl InMemoryMessageBus {
    /// Empty bus.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl MessageBus for InMemoryMessageBus {
    async fn publish(&self, msg: AgentMessage) -> Result<()> {
        self.inboxes
            .lock()
            .await
            .entry(msg.to.clone())
            .or_default()
            .push(msg);
        Ok(())
    }
    async fn drain(&self, agent_id: &str) -> Result<Vec<AgentMessage>> {
        Ok(self
            .inboxes
            .lock()
            .await
            .get_mut(agent_id)
            .map(std::mem::take)
            .unwrap_or_default())
    }
}

/// Strategy for routing a request through a fleet.
#[async_trait]
pub trait HandoffStrategy: Send + Sync {
    /// Run `input` through `agents` and return the final response.
    /// `bus` is available for inter-agent traffic.
    async fn run(
        &self,
        agents: &[(String, Arc<Mutex<Agent>>)],
        input: Message,
        bus: Arc<dyn MessageBus>,
    ) -> Result<AgentResponse>;

    /// Friendly name.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }
}

// ---------------------------------------------------------------------------
// Sequential
// ---------------------------------------------------------------------------

/// Sequential handoff: agents called in registration order, each
/// receiving the previous agent's reply as additional input context.
pub struct Sequential;

#[async_trait]
impl HandoffStrategy for Sequential {
    async fn run(
        &self,
        agents: &[(String, Arc<Mutex<Agent>>)],
        input: Message,
        bus: Arc<dyn MessageBus>,
    ) -> Result<AgentResponse> {
        if agents.is_empty() {
            return Err(CognisError::Configuration(
                "Sequential handoff: no agents registered".into(),
            ));
        }
        let mut current_input = input.clone();
        let mut last_response: Option<AgentResponse> = None;
        let mut prev_id: String = "user".into();
        for (id, agent) in agents {
            let mut a = agent.lock().await;
            let resp = a.run(current_input.clone()).await?;
            bus.publish(AgentMessage {
                from: prev_id.clone(),
                to: id.clone(),
                content: current_input.clone(),
                metadata: serde_json::Value::Null,
            })
            .await?;
            current_input = Message::human(resp.content.clone());
            prev_id = id.clone();
            last_response = Some(resp);
        }
        last_response.ok_or_else(|| CognisError::Internal("sequential ran no agents".into()))
    }

    fn name(&self) -> &str {
        "Sequential"
    }
}

// ---------------------------------------------------------------------------
// Supervisor
// ---------------------------------------------------------------------------

/// Boxed supervisor parser: `text → Option<(target_id, instruction)>`.
pub type SupervisorParser = Arc<dyn Fn(&str) -> Option<(String, String)> + Send + Sync>;

/// Supervisor handoff. The first agent acts as a router: its response
/// is parsed for the target worker name (the supervisor's response must
/// start with `target_id:` followed by the routed-to instruction).
///
/// Customization: override the parser via [`Supervisor::with_parser`].
pub struct Supervisor {
    parser: SupervisorParser,
}

impl Default for Supervisor {
    fn default() -> Self {
        Self {
            parser: Arc::new(default_supervisor_parser),
        }
    }
}

impl Supervisor {
    /// New supervisor with the default `id: instruction` parser.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the supervisor's response parser. Returns
    /// `Some((target_id, instruction))` to route, `None` to halt with
    /// the supervisor's response as the final answer.
    pub fn with_parser<F>(mut self, parser: F) -> Self
    where
        F: Fn(&str) -> Option<(String, String)> + Send + Sync + 'static,
    {
        self.parser = Arc::new(parser);
        self
    }
}

fn default_supervisor_parser(s: &str) -> Option<(String, String)> {
    let trimmed = s.trim();
    let (id, rest) = trimmed.split_once(':')?;
    Some((id.trim().to_string(), rest.trim().to_string()))
}

#[async_trait]
impl HandoffStrategy for Supervisor {
    async fn run(
        &self,
        agents: &[(String, Arc<Mutex<Agent>>)],
        input: Message,
        bus: Arc<dyn MessageBus>,
    ) -> Result<AgentResponse> {
        if agents.len() < 2 {
            return Err(CognisError::Configuration(
                "Supervisor handoff requires at least 2 agents (supervisor + 1 worker)".into(),
            ));
        }
        let (supervisor_id, supervisor) = &agents[0];
        let workers = &agents[1..];

        let mut sup = supervisor.lock().await;
        let sup_response = sup.run(input.clone()).await?;
        drop(sup);
        bus.publish(AgentMessage {
            from: "user".into(),
            to: supervisor_id.clone(),
            content: input,
            metadata: serde_json::Value::Null,
        })
        .await?;

        let routed = (self.parser)(&sup_response.content);
        let (target_id, instruction) = match routed {
            Some(v) => v,
            None => return Ok(sup_response),
        };
        let worker = workers
            .iter()
            .find(|(id, _)| id == &target_id)
            .ok_or_else(|| {
                CognisError::Configuration(format!(
                    "supervisor routed to unknown worker `{target_id}`"
                ))
            })?;
        bus.publish(AgentMessage {
            from: supervisor_id.clone(),
            to: target_id.clone(),
            content: Message::human(instruction.clone()),
            metadata: serde_json::Value::Null,
        })
        .await?;

        let mut w = worker.1.lock().await;
        w.run(Message::human(instruction)).await
    }

    fn name(&self) -> &str {
        "Supervisor"
    }
}

// ---------------------------------------------------------------------------
// ParallelVote
// ---------------------------------------------------------------------------

/// Parallel-vote handoff: every agent runs concurrently with the same
/// input; the response that the most agents return verbatim wins.
/// Ties are broken by registration order.
pub struct ParallelVote;

#[async_trait]
impl HandoffStrategy for ParallelVote {
    async fn run(
        &self,
        agents: &[(String, Arc<Mutex<Agent>>)],
        input: Message,
        _bus: Arc<dyn MessageBus>,
    ) -> Result<AgentResponse> {
        if agents.is_empty() {
            return Err(CognisError::Configuration(
                "ParallelVote: no agents registered".into(),
            ));
        }
        let mut handles = Vec::with_capacity(agents.len());
        for (_id, agent) in agents {
            let agent = agent.clone();
            let input = input.clone();
            handles.push(tokio::spawn(
                async move { agent.lock().await.run(input).await },
            ));
        }
        let mut responses: Vec<AgentResponse> = Vec::with_capacity(handles.len());
        for h in handles {
            match h.await {
                Ok(Ok(r)) => responses.push(r),
                Ok(Err(e)) => return Err(e),
                Err(e) => return Err(CognisError::Internal(format!("parallel-vote join: {e}"))),
            }
        }
        // Tally by content equality.
        let mut tallies: HashMap<String, usize> = HashMap::new();
        for r in &responses {
            *tallies.entry(r.content.clone()).or_insert(0) += 1;
        }
        let winning = responses
            .into_iter()
            .max_by_key(|r| tallies.get(&r.content).copied().unwrap_or(0))
            .ok_or_else(|| CognisError::Internal("ParallelVote produced no responses".into()))?;
        Ok(winning)
    }

    fn name(&self) -> &str {
        "ParallelVote"
    }
}

// ---------------------------------------------------------------------------
// MultiAgentOrchestrator
// ---------------------------------------------------------------------------

/// Holds a fleet of agents and a [`HandoffStrategy`]. Cheap to clone
/// (agents live behind `Arc<Mutex<...>>`).
#[derive(Clone)]
pub struct MultiAgentOrchestrator {
    agents: Vec<(String, Arc<Mutex<Agent>>)>,
    strategy: Arc<dyn HandoffStrategy>,
    bus: Arc<dyn MessageBus>,
}

impl MultiAgentOrchestrator {
    /// Build with a strategy + the default in-memory bus.
    pub fn new<S>(strategy: S) -> Self
    where
        S: HandoffStrategy + 'static,
    {
        Self {
            agents: Vec::new(),
            strategy: Arc::new(strategy),
            bus: Arc::new(InMemoryMessageBus::new()),
        }
    }

    /// Override the message bus.
    pub fn with_bus(mut self, bus: Arc<dyn MessageBus>) -> Self {
        self.bus = bus;
        self
    }

    /// Register an agent under `id`.
    pub fn add(mut self, id: impl Into<String>, agent: Agent) -> Self {
        self.agents.push((id.into(), Arc::new(Mutex::new(agent))));
        self
    }

    /// Borrow registered agent ids.
    pub fn agent_ids(&self) -> Vec<&str> {
        self.agents.iter().map(|(id, _)| id.as_str()).collect()
    }

    /// Run the request through the configured strategy.
    pub async fn run(&self, input: impl Into<Message>) -> Result<AgentResponse> {
        self.strategy
            .run(&self.agents, input.into(), self.bus.clone())
            .await
    }

    /// Borrow the message bus (e.g. to inspect inter-agent traffic).
    pub fn bus(&self) -> &Arc<dyn MessageBus> {
        &self.bus
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentBuilder;
    use cognis_llm::Client;
    use cognis_llm::{provider::LLMProvider, Provider};

    use async_trait::async_trait;
    use cognis_core::RunnableStream;
    use cognis_llm::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk};

    /// Test provider that echoes a fixed response template.
    struct CannedProvider {
        response: String,
    }
    #[async_trait]
    impl LLMProvider for CannedProvider {
        fn name(&self) -> &str {
            "canned"
        }
        fn provider_type(&self) -> Provider {
            Provider::Ollama
        }
        async fn chat_completion(&self, _: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message::ai(self.response.clone()),
                usage: None,
                finish_reason: "stop".into(),
                model: "canned".into(),
            })
        }
        async fn chat_completion_stream(
            &self,
            _: Vec<Message>,
            _: ChatOptions,
        ) -> Result<RunnableStream<StreamChunk>> {
            unimplemented!()
        }
        async fn health_check(&self) -> Result<HealthStatus> {
            Ok(HealthStatus::Healthy { latency_ms: 0 })
        }
    }

    fn agent_with_response(text: &str) -> Agent {
        let client = Client::new(Arc::new(CannedProvider {
            response: text.to_string(),
        }));
        AgentBuilder::new()
            .with_llm(client)
            .stateless()
            .build()
            .expect("agent builds")
    }

    #[tokio::test]
    async fn sequential_runs_agents_in_order() {
        let orch = MultiAgentOrchestrator::new(Sequential)
            .add("first", agent_with_response("from-first"))
            .add("second", agent_with_response("from-second"));
        let resp = orch.run("hello").await.unwrap();
        // The last agent's response wins for sequential.
        assert!(resp.content.contains("from-second"));
        // The bus saw the user → first → second handoffs.
        let inbox_first = orch.bus().drain("first").await.unwrap();
        let inbox_second = orch.bus().drain("second").await.unwrap();
        assert!(!inbox_first.is_empty());
        assert!(!inbox_second.is_empty());
    }

    #[tokio::test]
    async fn supervisor_routes_to_named_worker() {
        // Supervisor's canned response is "worker-a: do the thing".
        let sup = agent_with_response("worker-a: do the thing");
        let a = agent_with_response("a-handled");
        let b = agent_with_response("b-handled");
        let orch = MultiAgentOrchestrator::new(Supervisor::new())
            .add("supervisor", sup)
            .add("worker-a", a)
            .add("worker-b", b);
        let resp = orch.run("input").await.unwrap();
        assert_eq!(resp.content, "a-handled");
    }

    #[tokio::test]
    async fn supervisor_returns_supervisor_response_when_parser_returns_none() {
        let sup = agent_with_response("just answering directly");
        let a = agent_with_response("a-handled");
        let orch = MultiAgentOrchestrator::new(Supervisor::new())
            .add("supervisor", sup)
            .add("worker-a", a);
        let resp = orch.run("input").await.unwrap();
        assert_eq!(resp.content, "just answering directly");
    }

    #[tokio::test]
    async fn parallel_vote_picks_majority_response() {
        let orch = MultiAgentOrchestrator::new(ParallelVote)
            .add("a", agent_with_response("answer-X"))
            .add("b", agent_with_response("answer-X"))
            .add("c", agent_with_response("answer-Y"));
        let resp = orch.run("input").await.unwrap();
        assert_eq!(resp.content, "answer-X");
    }

    #[tokio::test]
    async fn empty_orchestrator_errors() {
        let orch = MultiAgentOrchestrator::new(Sequential);
        let res = orch.run("input").await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn in_memory_bus_isolates_inboxes() {
        let bus = InMemoryMessageBus::new();
        bus.publish(AgentMessage {
            from: "u".into(),
            to: "alice".into(),
            content: Message::human("hi"),
            metadata: serde_json::Value::Null,
        })
        .await
        .unwrap();
        bus.publish(AgentMessage {
            from: "u".into(),
            to: "bob".into(),
            content: Message::human("hi"),
            metadata: serde_json::Value::Null,
        })
        .await
        .unwrap();
        assert_eq!(bus.drain("alice").await.unwrap().len(), 1);
        assert_eq!(bus.drain("bob").await.unwrap().len(), 1);
        // Drained inboxes are now empty.
        assert!(bus.drain("alice").await.unwrap().is_empty());
    }
}
