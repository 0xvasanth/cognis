# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

RustChain is a Rust port of the LangChain/LangGraph/DeepAgents Python ecosystem. It's a Cargo workspace with four library crates and one example crate.

## Build & Test Commands

```bash
# Build entire workspace
cargo build

# Build a single crate
cargo build -p rustchain-core

# Build with provider features (required for chat model code)
cargo build -p rustchain --features openai,anthropic
cargo build -p rustchain --features all-providers

# Run all tests
cargo test

# Run tests for a single crate
cargo test -p rustchain-core
cargo test -p langgraph

# Run a single test by name
cargo test -p rustchain-core test_name

# Run an example (examples live in examples/ but are registered under crates/examples)
cargo run -p rustchain-examples --example simple_chain

# Check without building
cargo check --workspace
```

## Workspace Architecture

```
crates/
├── rustchain-core   # Zero workspace deps. Base traits: BaseChatModel, BaseLLM, BaseTool,
│                    # Runnable, Message enum, output parsers, prompts, vectorstores, embeddings
├── rustchain        # Depends on rustchain-core. Concrete implementations: chat model providers
│                    # (behind feature flags), agents, chains, memory, document loaders,
│                    # text splitters, tools. Re-exports core as `rustchain::core`.
├── langgraph        # Depends on rustchain-core. StateGraph builder, Pregel execution engine,
│                    # channels, checkpointing (sqlite/postgres behind features), streaming,
│                    # prebuilt ReAct agent. Exports START/END constants.
├── deepagents       # Depends on all three above. High-level agent factory (create_deep_agent),
│                    # middleware system, backends, tool registry, planning, multi-agent.
└── examples         # Non-publishable crate that registers all examples from examples/ dir
```

## Dependency Rules

- `rustchain-core` has **zero** workspace crate dependencies
- `langgraph` depends only on `rustchain-core`
- `rustchain` depends only on `rustchain-core`
- `deepagents` depends on `rustchain-core`, `rustchain`, and `langgraph`

## Feature Flags

**rustchain crate:** LLM providers are feature-gated. Each provider (`openai`, `anthropic`, `google`, `ollama`, `azure`) pulls in `reqwest` and `secrecy`. Use `all-providers` for everything. Loader features: `pdf`, `yaml`, `toml-loader`. Storage: `sqlite`.

**langgraph crate:** `sqlite` and `postgres` features gate checkpoint persistence via `sqlx`.

## Key Patterns

- **Traits over inheritance:** Python base classes → Rust traits (`BaseChatModel`, `BaseTool`, `Runnable`)
- **`async-trait`:** All async trait methods use the `async-trait` crate
- **`serde_json::Value`:** Used extensively as the generic state/input/output type across runnables and graph nodes
- **`thiserror`:** Each crate defines its own error enum
- **Graph constants:** `langgraph::START` and `langgraph::END` are string constants used as node identifiers in graph edges
- **Middleware trait:** `deepagents::middleware::Middleware` provides before/after hooks for model and tool calls
