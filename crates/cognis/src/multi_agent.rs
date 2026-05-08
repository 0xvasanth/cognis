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
// RoundRobin
// ---------------------------------------------------------------------------

/// Round-robin handoff: each `run` call picks the next agent in registration
/// order, cycling. Useful for load distribution across agents that share
/// the same role (e.g. "answerer" pool behind a load balancer) — every
/// request goes to one agent only.
///
/// State is shared across all `run()` invocations on this strategy, so
/// clone it (`Arc::clone(&strategy)`) if you want isolated counters.
pub struct RoundRobin {
    next: std::sync::atomic::AtomicUsize,
}

impl Default for RoundRobin {
    fn default() -> Self {
        Self {
            next: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl RoundRobin {
    /// New with counter at 0.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl HandoffStrategy for RoundRobin {
    async fn run(
        &self,
        agents: &[(String, Arc<Mutex<Agent>>)],
        input: Message,
        bus: Arc<dyn MessageBus>,
    ) -> Result<AgentResponse> {
        if agents.is_empty() {
            return Err(CognisError::Configuration(
                "RoundRobin: no agents registered".into(),
            ));
        }
        let idx = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % agents.len();
        let (id, agent) = &agents[idx];
        bus.publish(AgentMessage {
            from: "user".into(),
            to: id.clone(),
            content: input.clone(),
            metadata: serde_json::json!({"strategy": "round_robin", "index": idx}),
        })
        .await?;
        let mut a = agent.lock().await;
        a.run(input).await
    }

    fn name(&self) -> &str {
        "RoundRobin"
    }
}

// ---------------------------------------------------------------------------
// Hierarchical
// ---------------------------------------------------------------------------

/// Multi-level supervisor tree. Each level's agent receives the request,
/// emits a routing decision, and delegates to a deeper agent. The leaf
/// agent's response is the final answer.
///
/// Routing format defaults to `target_id: instruction` (same parser as
/// [`Supervisor`]). Override via [`Hierarchical::with_parser`].
///
/// Walks at most `max_depth` levels before forcing the current agent's
/// reply as the final result — guards against parse loops where every
/// level keeps routing onward.
pub struct Hierarchical {
    parser: SupervisorParser,
    max_depth: usize,
}

impl Default for Hierarchical {
    fn default() -> Self {
        Self {
            parser: Arc::new(default_supervisor_parser),
            max_depth: 8,
        }
    }
}

impl Hierarchical {
    /// New tree walker with the default parser and depth cap.
    pub fn new() -> Self {
        Self::default()
    }

    /// Cap the maximum routing hops.
    pub fn with_max_depth(mut self, n: usize) -> Self {
        self.max_depth = n.max(1);
        self
    }

    /// Override the routing parser. `Some((next_id, instruction))` to
    /// delegate; `None` to halt with the current agent's response.
    pub fn with_parser<F>(mut self, parser: F) -> Self
    where
        F: Fn(&str) -> Option<(String, String)> + Send + Sync + 'static,
    {
        self.parser = Arc::new(parser);
        self
    }
}

#[async_trait]
impl HandoffStrategy for Hierarchical {
    async fn run(
        &self,
        agents: &[(String, Arc<Mutex<Agent>>)],
        input: Message,
        bus: Arc<dyn MessageBus>,
    ) -> Result<AgentResponse> {
        if agents.is_empty() {
            return Err(CognisError::Configuration(
                "Hierarchical: no agents registered".into(),
            ));
        }
        let (mut current_id, mut current_agent) = {
            let (id, a) = &agents[0];
            (id.clone(), a.clone())
        };
        let mut current_input = input;
        let mut prev_id: String = "user".into();
        for hop in 0..self.max_depth {
            bus.publish(AgentMessage {
                from: prev_id.clone(),
                to: current_id.clone(),
                content: current_input.clone(),
                metadata: serde_json::json!({"strategy": "hierarchical", "hop": hop}),
            })
            .await?;
            let response = {
                let mut a = current_agent.lock().await;
                a.run(current_input.clone()).await?
            };
            match (self.parser)(&response.content) {
                Some((next_id, instruction)) => {
                    let next = agents
                        .iter()
                        .find(|(id, _)| id == &next_id)
                        .ok_or_else(|| {
                            CognisError::Configuration(format!(
                                "Hierarchical: routed to unknown agent `{next_id}`"
                            ))
                        })?;
                    prev_id = current_id;
                    current_id = next.0.clone();
                    current_agent = next.1.clone();
                    current_input = Message::human(instruction);
                }
                None => return Ok(response),
            }
        }
        // Depth cap reached — run the current agent one more time and return.
        let mut a = current_agent.lock().await;
        a.run(current_input).await
    }

    fn name(&self) -> &str {
        "Hierarchical"
    }
}

// ---------------------------------------------------------------------------
// Consensus
// ---------------------------------------------------------------------------

/// Weighted-quorum voting. Every registered agent runs in parallel with
/// the same input; their replies are tallied by content equality with
/// optional per-agent weights. The highest-weighted answer wins iff it
/// clears `quorum`; otherwise [`CognisError::Configuration`] is returned
/// with the tally.
///
/// Unlike [`ParallelVote`] (which always returns the plurality),
/// `Consensus` is honest about the absence of agreement.
pub struct Consensus {
    quorum: f32,
    weights: HashMap<String, f32>,
}

impl Default for Consensus {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl Consensus {
    /// Build with a quorum threshold expressed as a sum of agent weights.
    /// Per-agent weight defaults to 1.0; e.g. `quorum=2.0` means at least
    /// two unweighted agents must agree, `quorum=N` means all N.
    pub fn new(quorum: f32) -> Self {
        Self {
            quorum: quorum.max(0.0),
            weights: HashMap::new(),
        }
    }

    /// Set the weight of an agent. Default is 1.0.
    pub fn weight(mut self, agent_id: impl Into<String>, weight: f32) -> Self {
        self.weights.insert(agent_id.into(), weight.max(0.0));
        self
    }

    fn weight_of(&self, id: &str) -> f32 {
        self.weights.get(id).copied().unwrap_or(1.0)
    }
}

#[async_trait]
impl HandoffStrategy for Consensus {
    async fn run(
        &self,
        agents: &[(String, Arc<Mutex<Agent>>)],
        input: Message,
        _bus: Arc<dyn MessageBus>,
    ) -> Result<AgentResponse> {
        if agents.is_empty() {
            return Err(CognisError::Configuration(
                "Consensus: no agents registered".into(),
            ));
        }
        let mut handles = Vec::with_capacity(agents.len());
        for (id, agent) in agents {
            let agent = agent.clone();
            let id = id.clone();
            let input = input.clone();
            handles.push(tokio::spawn(async move {
                let resp = agent.lock().await.run(input).await;
                (id, resp)
            }));
        }
        let mut tally: HashMap<String, (f32, AgentResponse)> = HashMap::new();
        let mut total_weight = 0.0_f32;
        for h in handles {
            let (id, res) = h
                .await
                .map_err(|e| CognisError::Internal(format!("consensus join: {e}")))?;
            let resp = res?;
            let w = self.weight_of(&id);
            total_weight += w;
            tally
                .entry(resp.content.clone())
                .and_modify(|(acc, _)| *acc += w)
                .or_insert_with(|| (w, resp));
        }
        let (winning_content, (winning_weight, winning_resp)) = tally
            .into_iter()
            .max_by(|(_, (a, _)), (_, (b, _))| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| CognisError::Internal("Consensus: no responses".into()))?;
        if winning_weight >= self.quorum {
            Ok(winning_resp)
        } else {
            Err(CognisError::Configuration(format!(
                "Consensus: no answer reached quorum {} (best: {:.2}/{:.2} for {:?})",
                self.quorum, winning_weight, total_weight, winning_content
            )))
        }
    }

    fn name(&self) -> &str {
        "Consensus"
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
    async fn round_robin_cycles_through_agents() {
        // Wrap the strategy in Arc so we can hand the same instance to
        // multiple orchestrators sharing one counter. (Or, equivalently,
        // run on the same orchestrator three times.)
        let orch = MultiAgentOrchestrator::new(RoundRobin::new())
            .add("a", agent_with_response("from-a"))
            .add("b", agent_with_response("from-b"))
            .add("c", agent_with_response("from-c"));
        let r0 = orch.run("input").await.unwrap();
        let r1 = orch.run("input").await.unwrap();
        let r2 = orch.run("input").await.unwrap();
        let r3 = orch.run("input").await.unwrap();
        assert_eq!(r0.content, "from-a");
        assert_eq!(r1.content, "from-b");
        assert_eq!(r2.content, "from-c");
        assert_eq!(r3.content, "from-a", "wraps back to first");
    }

    #[tokio::test]
    async fn round_robin_publishes_routing_metadata() {
        let orch =
            MultiAgentOrchestrator::new(RoundRobin::new()).add("only", agent_with_response("ok"));
        orch.run("input").await.unwrap();
        let inbox = orch.bus().drain("only").await.unwrap();
        assert_eq!(inbox.len(), 1);
        let strategy = inbox[0].metadata.get("strategy").and_then(|v| v.as_str());
        assert_eq!(strategy, Some("round_robin"));
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

    #[tokio::test]
    async fn hierarchical_routes_through_two_levels() {
        // ceo -> "vp_eng: build feature" → vp_eng -> "ic: implement" → ic
        let ceo = agent_with_response("vp_eng: build feature");
        let vp = agent_with_response("ic: implement");
        let ic = agent_with_response("done implementing");
        let orch = MultiAgentOrchestrator::new(Hierarchical::new())
            .add("ceo", ceo)
            .add("vp_eng", vp)
            .add("ic", ic);
        let resp = orch.run("ship X").await.unwrap();
        assert_eq!(resp.content, "done implementing");
    }

    #[tokio::test]
    async fn hierarchical_halts_on_unrouted_response() {
        let ceo = agent_with_response("answering directly");
        let vp = agent_with_response("would have routed");
        let orch = MultiAgentOrchestrator::new(Hierarchical::new())
            .add("ceo", ceo)
            .add("vp", vp);
        let resp = orch.run("hi").await.unwrap();
        assert_eq!(resp.content, "answering directly");
    }

    #[tokio::test]
    async fn hierarchical_unknown_route_target_errors() {
        let ceo = agent_with_response("ghost: do work");
        let orch = MultiAgentOrchestrator::new(Hierarchical::new()).add("ceo", ceo);
        let res = orch.run("hi").await;
        assert!(res.is_err(), "should fail on unknown route");
    }

    #[tokio::test]
    async fn consensus_returns_winner_when_quorum_met() {
        // 2 of 3 agree on "X" — quorum = 2.0 (default weights).
        let orch = MultiAgentOrchestrator::new(Consensus::new(2.0))
            .add("a", agent_with_response("X"))
            .add("b", agent_with_response("X"))
            .add("c", agent_with_response("Y"));
        let resp = orch.run("q").await.unwrap();
        assert_eq!(resp.content, "X");
    }

    #[tokio::test]
    async fn consensus_errors_when_quorum_not_met() {
        // 3 way tie, each with weight 1.0. Quorum 2.0 → no winner.
        let orch = MultiAgentOrchestrator::new(Consensus::new(2.0))
            .add("a", agent_with_response("X"))
            .add("b", agent_with_response("Y"))
            .add("c", agent_with_response("Z"));
        let err = orch.run("q").await.unwrap_err();
        assert!(err.to_string().contains("quorum"), "got: {err}");
    }

    #[tokio::test]
    async fn consensus_respects_weights() {
        // Weighted: a=3.0 alone outvotes b+c (1.0 each).
        let orch = MultiAgentOrchestrator::new(Consensus::new(2.5).weight("a", 3.0))
            .add("a", agent_with_response("X"))
            .add("b", agent_with_response("Y"))
            .add("c", agent_with_response("Y"));
        let resp = orch.run("q").await.unwrap();
        assert_eq!(resp.content, "X");
    }
}
