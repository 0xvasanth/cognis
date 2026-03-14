# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Cognis is a Rust LLM framework inspired by the LangChain/LangGraph/DeepAgents Python ecosystem. It's a Cargo workspace with four library crates and one example crate.

## Build & Test Commands

```bash
# Build entire workspace
cargo build

# Build a single crate
cargo build -p cognis-core

# Build with provider features (required for chat model code)
cargo build -p cognis --features openai,anthropic
cargo build -p cognis --features all-providers

# Run all tests
cargo test

# Run tests for a single crate
cargo test -p cognis-core
cargo test -p cognisgraph

# Run a single test by name
cargo test -p cognis-core test_name

# Run an example (examples live in examples/ but are registered under crates/examples)
cargo run -p cognis-examples --example simple_chain

# Check without building
cargo check --workspace
```

## Workspace Architecture

```
crates/
├── cognis-core     # Zero workspace deps. Base traits: BaseChatModel, BaseLLM, BaseTool,
│                   # Runnable, Message enum, output parsers, prompts, vectorstores, embeddings
├── cognis          # Depends on cognis-core. Concrete implementations: chat model providers
│                   # (behind feature flags), agents, chains, memory, document loaders,
│                   # text splitters, tools. Re-exports core as `cognis::core`.
├── cognisgraph     # Depends on cognis-core. StateGraph builder, Pregel execution engine,
│                   # channels, checkpointing (sqlite/postgres behind features), streaming,
│                   # prebuilt ReAct agent. Exports START/END constants.
├── cognisagent     # Depends on all three above. High-level agent factory (create_deep_agent),
│                   # middleware system, backends, tool registry, planning, multi-agent.
└── examples        # Non-publishable crate that registers all examples from examples/ dir
```

## Dependency Rules

- `cognis-core` has **zero** workspace crate dependencies
- `cognisgraph` depends only on `cognis-core`
- `cognis` depends only on `cognis-core`
- `cognisagent` depends on `cognis-core`, `cognis`, and `cognisgraph`

## Feature Flags

**cognis crate:** LLM providers are feature-gated. Each provider (`openai`, `anthropic`, `google`, `ollama`, `azure`) pulls in `reqwest` and `secrecy`. Use `all-providers` for everything. Loader features: `pdf`, `yaml`, `toml-loader`. Storage: `sqlite`.

**cognisgraph crate:** `sqlite` and `postgres` features gate checkpoint persistence via `sqlx`.

## Key Patterns

- **Traits over inheritance:** Python base classes → Rust traits (`BaseChatModel`, `BaseTool`, `Runnable`)
- **`async-trait`:** All async trait methods use the `async-trait` crate
- **`serde_json::Value`:** Used extensively as the generic state/input/output type across runnables and graph nodes
- **`thiserror`:** Each crate defines its own error enum
- **Graph constants:** `cognisgraph::START` and `cognisgraph::END` are string constants used as node identifiers in graph edges
- **Middleware trait:** `cognisagent::middleware::Middleware` provides before/after hooks for model and tool calls

## Worktrees

Worktree directory: `.worktrees/` (project-local, globally gitignored)

```bash
# Create a worktree for a new feature
git worktree add .worktrees/my-feature -b feature/my-feature

# List active worktrees
git worktree list

# Remove when done
git worktree remove .worktrees/my-feature
```
