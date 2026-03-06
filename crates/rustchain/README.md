# rustchain

Implementation layer for the RustChain LLM framework. Provides concrete chat model
integrations, agent execution, chains, memory, document loaders, text splitters,
embedding providers, and built-in tools.

## Chat Model Providers

Each provider is behind a feature flag to keep compile times and dependencies minimal.

| Feature | Provider | Module |
|---------|----------|--------|
| `anthropic` | Anthropic Claude | `chat_models::anthropic` |
| `openai` | OpenAI GPT | `chat_models::openai` |
| `google` | Google Gemini | `chat_models::google` |
| `ollama` | Ollama (local) | `chat_models::ollama` |
| `azure` | Azure OpenAI | `chat_models::azure` |
| `all-providers` | All of the above | -- |

## Key Modules

- **agents** -- Agent executor with middleware pipeline (retry, PII redaction, summarization, human-in-the-loop, tool selection, and more)
- **chains** -- LLM chain, conversation chain, sequential chain
- **memory** -- Buffer, window, and summary memory strategies
- **document_loaders** -- Text, CSV, JSON, and directory loaders
- **text_splitter** -- Character, recursive, markdown, HTML, JSON, code, and token splitters
- **embeddings** -- OpenAI and Ollama embedding providers
- **tools** -- Calculator, shell command, and JSON query tools

## Usage

```toml
[dependencies]
rustchain = { path = "../rustchain", features = ["anthropic"] }
```

```rust,ignore
use rustchain::chat_models::anthropic::ChatAnthropic;
use rustchain_core::runnables::Runnable;
use serde_json::json;

let model = ChatAnthropic::new("claude-sonnet-4-20250514");
let result = model.invoke(json!({"messages": []}), None).await.unwrap();
```

## Feature Flags

| Feature | Adds |
|---------|------|
| `openai` | `reqwest`, `secrecy` |
| `anthropic` | `reqwest`, `secrecy` |
| `google` | `reqwest`, `secrecy` |
| `ollama` | `reqwest` |
| `azure` | `reqwest`, `secrecy` |
| `all-providers` | All provider features |
