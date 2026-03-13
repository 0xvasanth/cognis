# Contributing to Rustchain

Thanks for your interest in contributing to rustchain! This document provides guidelines and information for contributors.

## Getting Started

1. **Fork the repository** and clone your fork
2. **Install Rust** (stable, latest version recommended)
3. **Build the workspace**: `cargo build --workspace`
4. **Run tests**: `cargo test --workspace`

## Project Structure

```
rustchain/
├── crates/
│   ├── rustchain-core/     # Base traits & types (no workspace dependencies)
│   ├── rustchain/          # Agent framework, chat models, tools, integrations
│   ├── langgraph/          # State graphs, Pregel engine, checkpointing
│   └── deepagents/         # High-level agent factory, middleware, backends
├── examples/               # Example applications
└── Cargo.toml              # Workspace root
```

## Development Workflow

### Before You Start

- Check existing [issues](https://github.com/0xvasanth/rustchain/issues) and [discussions](https://github.com/0xvasanth/rustchain/discussions)
- For large changes, open an issue or discussion first to align on approach
- For new provider integrations, use the Provider Integration issue template

### Making Changes

1. Create a feature branch from `main`
2. Make your changes
3. Ensure all checks pass:

```bash
cargo fmt --all              # Format code
cargo clippy --workspace     # Lint
cargo test --workspace       # Run tests
```

4. Write tests for new functionality
5. Submit a pull request

### Code Guidelines

- **Dependency boundaries**: `rustchain-core` must have zero dependencies on other workspace crates
- **Feature flags**: Use feature flags for optional provider integrations
- **Error handling**: Use `thiserror` for error types, return `Result<T, E>`
- **Async**: Use `tokio` for all async functions
- **Documentation**: Add `///` doc comments to all public APIs
- **Testing**: Aim for comprehensive test coverage

### Commit Messages

Use conventional commits:
- `feat(crate): description` - New features
- `fix(crate): description` - Bug fixes
- `docs(crate): description` - Documentation
- `refactor(crate): description` - Code refactoring
- `test(crate): description` - Tests
- `chore: description` - Maintenance tasks

### Provider Integrations

Each LLM provider should:
- Be gated behind a feature flag (e.g., `features = ["anthropic"]`)
- Implement the `ChatModel` trait from `rustchain-core`
- Include comprehensive tests with mock responses
- Include an example in the `examples/` directory

## Reporting Issues

- **Bugs**: Use the Bug Report template
- **Features**: Use the Feature Request template
- **Provider requests**: Use the Provider Integration template
- **Questions**: Use [Discussions](https://github.com/0xvasanth/rustchain/discussions)

## Code of Conduct

Be respectful and constructive. We're building something together.

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
