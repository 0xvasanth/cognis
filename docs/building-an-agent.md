# Building an Agent Application

The README's Quick Start covers *chains* (`prompt → model → parser`). Agents are different: they loop, call tools, and decide what to do next. This guide walks through the happy path end-to-end using what cognis already ships — so you don't hand-roll the ReAct loop, a JSON parser, or a scratchpad formatter.

If you've read the README and now want to build something that can actually *do things*, start here.

> All snippets are adapted from runnable examples in [`examples/`](../examples/). If a snippet elides setup for brevity, the filename at the top of each section points to the full version.

---

## TL;DR: the smallest working agent

```rust
use std::sync::Arc;
use cognis::agents::AgentExecutor;
use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::Message;
use cognis_core::tools::base::BaseTool;
use cognis_core::tools::simple::SimpleTool;

let search = SimpleTool::new(
    "search",
    "Search for information about a topic",
    |q: &str| Ok(format!("Results for '{q}': ...")),
);

let executor = AgentExecutor::builder()
    .model(model)                                   // Arc<dyn BaseChatModel>
    .tool(Arc::new(search) as Arc<dyn BaseTool>)
    .max_iterations(10)
    .build();

let result = executor.run(&[Message::human("What is Rust?")]).await?;
println!("{}", result.output);
```

That's it. No hand-written loop, no JSON parser, no scratchpad formatting. The executor drives the model, executes tool calls, appends `ToolMessage`s to history, and stops when the model returns a final answer (or you hit `max_iterations`).

The rest of this guide is the *why* and *when* behind each knob.

---

## 1. Authoring a `BaseTool`

Three ways to define a tool, from easiest to most expressive:

### `SimpleTool` — one string input, sync closure

```rust
use cognis_core::tools::simple::SimpleTool;

let search = SimpleTool::new(
    "search",
    "Search for information about a topic",
    |query: &str| Ok(format!("Results for '{query}': ...")),
);
```

Use when your tool takes a single string and returns a string. Zero ceremony.

### `StructuredTool` — typed args, async closure, schema

```rust
use std::collections::HashMap;
use serde_json::{json, Value};
use cognis_core::tools::structured::StructuredTool;

let calculator = StructuredTool::new(
    "calculator",
    "Perform arithmetic between two numbers",
    json!({
        "type": "object",
        "properties": {
            "a":  { "type": "number" },
            "b":  { "type": "number" },
            "op": { "type": "string", "enum": ["add", "sub", "mul", "div"] }
        },
        "required": ["a", "b", "op"]
    }),
    |args: HashMap<String, Value>| async move {
        let a  = args.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let b  = args.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let op = args.get("op").and_then(|v| v.as_str()).unwrap_or("add");
        let result = match op {
            "add" => a + b, "sub" => a - b, "mul" => a * b,
            "div" if b != 0.0 => a / b,
            _ => return Ok(json!({ "error": "bad op" })),
        };
        Ok(json!({ "result": result }))
    },
);
```

The schema is shipped to the model as the tool's parameter definition. The model will call your tool with args that match it.

### Implementing `BaseTool` directly — full control

```rust
use async_trait::async_trait;
use cognis_core::tools::BaseTool;
use cognis_core::tools::types::{ToolInput, ToolOutput};

struct GetCurrentTimeTool;

#[async_trait]
impl BaseTool for GetCurrentTimeTool {
    fn name(&self) -> &str { "get_current_time" }

    fn description(&self) -> &str {
        "Get the current date and time in a given timezone."
    }

    fn args_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "type": "object",
            "properties": {
                "timezone": { "type": "string", "description": "IANA tz name" }
            }
        }))
    }

    async fn _run(&self, input: ToolInput) -> cognis_core::error::Result<ToolOutput> {
        // ...pull `timezone` out of input, call chrono, return ToolOutput::Content(json!(...))
    }
}
```

Do this when you need custom error types, retries inside the tool, access to per-tool resources (DB pool, HTTP client), or async work that doesn't fit a closure.

> **Built-ins** — `cognis` ships `calculator`, `shell`, `filesystem`, HTTP, and more. Check `cognis::tools::*` before writing your own.
>
> **Derive** — `cognis-macros` provides `#[derive(Tool)]` which generates `BaseTool` from a struct. See [`examples/tools/derive_tool.rs`](../examples/tools/derive_tool.rs).

---

## 2. Binding tools to the model

Every `BaseChatModel` exposes `bind_tools`. You usually don't call it directly — `AgentExecutor` and `create_react_agent` do it for you — but it's the primitive underneath.

```rust
use cognis_core::tools::base::ToolSchema;
use cognis_core::language_models::chat_model::ToolChoice;

let schema = ToolSchema {
    name:        "calculator".into(),
    description: "Arithmetic".into(),
    parameters:  calculator.args_schema(),
    extras:      None,
};

let bound: Box<dyn BaseChatModel> = model.bind_tools(
    &[schema],
    Some(ToolChoice::Auto),     // Auto | Any | Tool(name) | None
)?;
```

`bound` is a new model that will emit `tool_calls` on its `AIMessage` when the model decides to call a tool. Works across `ChatOpenAI`, `ChatAnthropic`, `ChatGoogleGenAI`, and `ChatOllama` — the provider-specific serialization is handled for you.

---

## 3. Reading `AIMessage.tool_calls` directly

If you want raw control — debugging, custom routing, non-standard tool execution — skip the executor and read tool calls yourself:

```rust
use cognis_core::messages::Message;

let messages = vec![Message::human("What is 6 * 7?")];
let result = bound._generate(&messages, None).await?;
let gen = result.generations.first().expect("at least one generation");

if let Message::Ai(ai) = &gen.message {
    for tc in &ai.tool_calls {
        println!("model wants to call {} with {:?}", tc.name, tc.args);
        // dispatch however you like
    }
}
```

`tc.name` is the tool name, `tc.args` is `HashMap<String, Value>`, `tc.id` is the call id — pass it back as `ToolMessage { tool_call_id, .. }` so the model can correlate the observation to the call.

For the structured version that returns `AgentOutput::Actions(..)` or `AgentOutput::Finish(..)`:

```rust
use cognis::agents::{parse_ai_message_to_agent_output, AgentOutput};

let ai_json = serde_json::to_value(&gen.message)?;
match parse_ai_message_to_agent_output(&ai_json)? {
    AgentOutput::Actions(actions) => { /* call tools */ }
    AgentOutput::Finish(finish)   => { /* done */ }
}
```

---

## 4. `AgentExecutor` — the loop you don't have to write

```rust
use cognis::agents::{AgentExecutor, EarlyStoppingMethod};

let executor = AgentExecutor::builder()
    .model(model)
    .tools(vec![
        Arc::new(search)     as Arc<dyn BaseTool>,
        Arc::new(calculator) as Arc<dyn BaseTool>,
    ])
    .max_iterations(10)
    .max_execution_time_secs(60)
    .return_intermediate_steps(true)
    .handle_parsing_errors(true)
    .early_stopping_method(EarlyStoppingMethod::GenerateResponse)
    .build();

let result = executor.run(&[Message::human("What's 42 * 7?")]).await?;
```

| Knob | What it does |
|---|---|
| `max_iterations(n)` | Hard cap on model↔tool round-trips. Default 10. |
| `max_execution_time_secs(n)` | Wall-clock budget. |
| `return_intermediate_steps(true)` | Populate `result.intermediate_steps` with every `(AgentAction, observation)` pair. |
| `handle_parsing_errors(true)` | On malformed output, feed the error back to the model and retry instead of bailing. |
| `early_stopping_method(..)` | What happens on limit hit: `Force` returns what you have; `GenerateResponse` asks the model for a final answer; unset returns `Err(RecursionLimitExceeded)`. |

`AgentResult` gives you `messages` (full conversation), `output` (final string), and `intermediate_steps` (if requested).

---

## 5. Streaming intermediate steps with `CallbackHandler`

```rust
use async_trait::async_trait;
use uuid::Uuid;
use cognis_core::callbacks::CallbackHandler;
use cognis_core::agents::{AgentAction, AgentFinish};

struct StepLogger;

#[async_trait]
impl CallbackHandler for StepLogger {
    fn name(&self) -> &str { "step-logger" }

    async fn on_agent_action(
        &self, action: &AgentAction, _run_id: Uuid, _parent: Option<Uuid>,
    ) -> cognis_core::error::Result<()> {
        println!("→ calling {} with {}", action.tool, action.tool_input);
        Ok(())
    }

    async fn on_tool_end(
        &self, output: &str, _run_id: Uuid, _parent: Option<Uuid>,
    ) -> cognis_core::error::Result<()> {
        println!("← observation: {output}");
        Ok(())
    }

    async fn on_agent_finish(
        &self, finish: &AgentFinish, _run_id: Uuid, _parent: Option<Uuid>,
    ) -> cognis_core::error::Result<()> {
        println!("✓ final: {:?}", finish.return_values.get("output"));
        Ok(())
    }
}

let executor = AgentExecutor::builder()
    .model(model)
    .tool(Arc::new(search) as Arc<dyn BaseTool>)
    .callback(Arc::new(StepLogger))
    .build();
```

Per-phase hooks available: `on_llm_start/new_token/end/error`, `on_chain_start/end/error`, `on_tool_start/end/error`, `on_agent_action/finish`, `on_retriever_start/end/error`. Filter flags (`ignore_llm`, `ignore_tool`, etc.) let a handler opt out of categories. For token streaming specifically, pair this with FR-1 once it lands — `on_llm_new_token` is already wired.

---

## 6. Structured JSON output with `with_structured_output`

When you want the model to return JSON that matches a schema instead of freeform text:

```rust
use serde_json::json;
use cognis::chat_models::structured::with_structured_output;

let structured = with_structured_output(
    Box::new(model),
    json!({
        "title": "Person",
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "age":  { "type": "integer" }
        },
        "required": ["name", "age"]
    }),
    Some("tool_calling"),   // or "json_mode"
    false,                  // include_raw
)?;

let messages = vec![Message::human("Alice is 30")];
let result = structured._generate(&messages, None).await?;
// result.generations[0].text is valid JSON matching the schema
```

`tool_calling` binds a synthetic tool with your schema and forces the model to call it — works everywhere. `json_mode` uses native JSON mode (OpenAI etc.) when you want to preserve normal `tool_calls` alongside structured output.

If the model still hands you a malformed blob (trailing commas, stray prose), wrap your parser in `OutputFixingParser`:

```rust
use cognis::output_parsers::OutputFixingParser;

let robust = OutputFixingParser::builder()
    .parser(json_parser)
    .llm(model.clone())
    .build();
// robust.parse() retries malformed output through the LLM with the error attached.
```

---

## 7. `format_to_tool_messages` — scratchpad formatting

If you're running your own loop (not using `AgentExecutor`), you still need to turn the list of `(AgentAction, observation)` pairs into messages the model can see. Don't hand-roll this — use:

```rust
use cognis::agents::format_to_tool_messages;

let scratchpad = format_to_tool_messages(&intermediate_steps);
let messages: Vec<Message> = initial_messages.iter()
    .cloned()
    .chain(scratchpad)
    .collect();
let result = model._generate(&messages, None).await?;
```

This emits one `ToolMessage` per observation with the correct `tool_call_id` — which is what every provider actually expects and what lets the model correlate calls to results. Shoving observations into user messages breaks tool-call chaining on most providers.

---

## Alternative: `create_react_agent` (graph-based)

`AgentExecutor` is a linear loop. If you want checkpointing, interrupts, or to compose the agent into a larger `StateGraph`, use the graph-based ReAct agent from `cognisgraph`:

```rust
use cognisgraph::prebuilt::create_react_agent;

let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(GetCurrentTimeTool), Arc::new(CalculatorTool)];
let graph = create_react_agent(model, tools)?;

let input = serde_json::json!({
    "messages": [{ "type": "human", "content": "What time is it in IST?" }]
});
let result = graph.invoke(input).await?;
```

`graph` is a `CompiledStateGraph` — you get `invoke`, `stream`, `get_state`, `update_state`, and checkpointer integration for free. Same mental model, different substrate.

See [`examples/agents/react_agent.rs`](../examples/agents/react_agent.rs) for a full working version with Ollama auto-detection.

---

## What you avoided re-implementing

| If you were about to hand-roll | Use this instead |
|---|---|
| ReAct decision JSON + parser | `cognis::agents::ReActOutputParser`, `cognisgraph::prebuilt::create_react_agent` |
| Multi-step loop with iteration cap | `cognis::agents::AgentExecutor` with `max_iterations` |
| Tool-call history shoved into user messages | `Message::Tool(ToolMessage)` with `tool_call_id` |
| Hand-written `parse_lenient` for malformed JSON | `cognis::output_parsers::OutputFixingParser` |
| `mpsc::Sender<Thought>` for progress | `CallbackHandler` with per-phase hooks |
| Hand-parsed `{"action": "tool", "tool": "..."}` | `cognis::agents::parse_ai_message_to_agent_output` (consumes native `tool_calls`) |
| Domain-specific stringification of tool calls | `cognis::agents::format_to_tool_messages` |
| Domain-specific JSON-shaped prompts | `cognis::chat_models::structured::with_structured_output(schema)` |

---

## Where to go next

- [`examples/tools/tool_calling_agent.rs`](../examples/tools/tool_calling_agent.rs) — `SimpleTool`, `StructuredTool`, `CachedTool`, `AgentExecutor` together
- [`examples/agents/react_agent.rs`](../examples/agents/react_agent.rs) — graph-based ReAct with Ollama
- [`examples/agents/conversational_agent.rs`](../examples/agents/conversational_agent.rs) — memory + tools
- [`examples/agents/plan_and_execute.rs`](../examples/agents/plan_and_execute.rs) — planner/executor split
- [`examples/agents/execution_hooks.rs`](../examples/agents/execution_hooks.rs) — callback-driven observability
- [`cognisagent`](../crates/cognisagent/) — high-level `create_deep_agent` factory when you want the batteries included (filesystem, memory, subagents, planning)
