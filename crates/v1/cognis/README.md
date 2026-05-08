<div align="center">

# cognis

**LLM providers, chains, agents, memory, and tools for Rust.**

[![crates.io](https://img.shields.io/crates/v/cognis.svg)](https://crates.io/crates/cognis)
[![docs.rs](https://docs.rs/cognis/badge.svg)](https://docs.rs/cognis)
[![MIT](https://img.shields.io/crates/l/cognis.svg)](https://opensource.org/licenses/MIT)

[Workspace](https://github.com/0xvasanth/cognis) | [API Docs](https://docs.rs/cognis)

</div>

---

`cognis` is the implementation layer of the [Cognis](https://github.com/0xvasanth/cognis) framework. It provides concrete LLM provider integrations, agent execution, composable chains, conversation memory, document processing, and built-in tools — all behind feature flags so you only compile what you use.

## Quick Start

```toml
[dependencies]
cognis = { version = "0.1", features = ["anthropic"] }
cognis-core = "0.1"
tokio = { version = "1", features = ["full"] }
```

```rust,ignore
use cognis::chat_models::anthropic::ChatAnthropic;
use cognis_core::runnables::Runnable;
use serde_json::json;

let model = ChatAnthropic::new("claude-sonnet-4-20250514");
let result = model.invoke(json!({"messages": []}), None).await?;
```

## LLM Providers

Enable only what you need:

```toml
cognis = { version = "0.1", features = ["openai", "anthropic"] }  # pick providers
cognis = { version = "0.1", features = ["all-providers"] }         # or grab everything
```

| Feature | Provider |
|---------|----------|
| `anthropic` | Anthropic Claude |
| `openai` | OpenAI GPT |
| `google` | Google Gemini |
| `ollama` | Ollama (local models) |
| `azure` | Azure OpenAI |

## What's Inside

**Agents** — Executor with a pluggable middleware pipeline: retry, PII redaction, tool selection, human-in-the-loop, summarization, and more.

**Chains** — LLM chain, conversation chain, sequential chain, extraction chain, structured output chain, router chain, retrieval QA chain.

**Memory** — Buffer, window, summary, and entity memory strategies for multi-turn conversations.

**Document Loaders** — Text, CSV, JSON, and directory loaders for ingesting data.

**Text Splitters** — Character, recursive, markdown, HTML, JSON, code, and token-aware splitters for chunking documents.

**Embeddings** — OpenAI and Ollama embedding providers with batch support.

**Retrievers** — Caching, reranking, query translation, and multi-query retrievers.

**Tools** — Calculator, shell command, and JSON query tools out of the box.

## Additional Feature Flags

| Feature | What it adds |
|---------|-------------|
| `pdf` | PDF document loader |
| `yaml` | YAML document loader |
| `toml-loader` | TOML document loader |
| `sqlite` | SQLite-backed stores |

## Part of the Cognis Workspace

| Crate | Role |
|-------|------|
| [cognis-core](https://crates.io/crates/cognis-core) | Foundation traits and types |
| **cognis** | LLM providers, chains, memory, tools (you are here) |
| [cognisgraph](https://crates.io/crates/cognisgraph) | State graph orchestration engine |
| [cognisagent](https://crates.io/crates/cognisagent) | High-level agent framework |
