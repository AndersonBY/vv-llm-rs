# vv-llm-rs Engineering Docs

These docs are the project knowledge base for `vv-llm-rs`. `AGENTS.md` is only the navigation entry point; this directory is the source of truth for architecture, provider behavior, testing, security, and development workflow.

## Map

- `ARCHITECTURE.md` - workspace layout, public API boundaries, settings resolution, normalized types, and extension points.
- `PROVIDERS.md` - provider adapter behavior for OpenAI-compatible APIs, Anthropic direct, Anthropic Bedrock, Vertex OpenAI-compatible endpoints, embeddings, and rerank.
- `TESTING.md` - local tests, live integration tests, fixture policy, and coverage expectations.
- `DEVELOPMENT.md` - common workflows for adding providers, changing streaming, changing settings, and updating tokenizer behavior.
- `CONTRACT.md` - vendored language-neutral schemas, fixtures, catalog, lock verification, and synchronization.
- `SECURITY.md` - credential handling, secret-bearing fixtures, logs, auth flows, and safe debugging.
- `RELEASE.md` - crates.io release setup, version tags, and publish workflow behavior.

## Documentation Principles

- Keep the README focused on user-facing usage.
- Keep `AGENTS.md` short and link to deeper docs.
- Put stable engineering facts in this directory instead of relying on chat history or external notes.
- Update docs in the same change as public API, settings, provider, live-test, or behavior changes.
- Prefer checklists and concrete commands over broad prose.

## Current Scope

`vv-llm-rs` provides typed Rust clients for:

- Chat completions and normalized chat streaming.
- OpenAI-compatible chat and embeddings through `async-openai`.
- Anthropic direct Messages API through the `anthropic` Rust SDK.
- Anthropic on AWS Bedrock through Bedrock Converse.
- OpenAI-compatible Gemini models on Vertex AI with Google access-token exchange.
- Custom JSON HTTP rerank APIs.
- Shared settings resolution, multimodal message types, tool calls, token counting, and small utility helpers.

For implementation details, start with `ARCHITECTURE.md`, then narrow down to `PROVIDERS.md` and `TESTING.md`.
