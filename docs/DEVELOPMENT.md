# Development Workflow

This document captures common change workflows for `vv-llm-rs`.

## Before Editing

- Read `AGENTS.md`.
- Read the focused docs for the area you will touch.
- Check `git status --short` and preserve unrelated worktree changes.
- Do not inspect or print `crates/vv-llm/tests/fixtures/dev_settings.json` unless the task is explicitly about live credentials, and even then never reveal values.

## Updating Contract Artifacts

The language-neutral schemas, deterministic fixtures, and default catalog are
vendored under `crates/vv-llm/contract/v1.0.0/`. A read-only check uses only
the checked-in tree:

```bash
python scripts/sync_contract.py --check
```

To synchronize, provide an explicit contract release directory:

```bash
python scripts/sync_contract.py --source /secure/path/vv-llm-contract/dist/release-v1.0.0
VV_LLM_CONTRACT_SOURCE=/secure/path/vv-llm-contract/dist/release-v1.0.0 python scripts/sync_contract.py
```

The check is also the first release CI gate. Never put credentials or live
provider payloads in this artifact tree; use the separately documented
live-test settings workflow instead.

## Adding A Chat Backend

1. Decide whether the backend is truly OpenAI-compatible or needs a native adapter.
2. If OpenAI-compatible, add the `BackendType` variant and route it through `create_chat_client_from_resolved`.
3. If native, add a focused module under `chat_clients/` and keep provider-specific request/response types inside that module.
4. Preserve `ChatRequest`, `ChatResponse`, and `ChatStreamDelta` as the public contract.
5. Add settings tests for backend name and endpoint resolution.
6. Add chat request-shape tests for roles, text, images, tools, tool results, options, and provider model id override as applicable.
7. Add stream-normalization tests when the backend supports streaming.
8. Add live tests only when credentials and provider stability make them useful.
9. Update README and docs.

## Changing Streaming

Streaming is user-facing API behavior. Treat it as high risk.

Checklist:

- Keep `ChatStreamDelta` normalized across providers.
- Preserve content, reasoning, tool-call, usage, raw content, and done semantics.
- Add deterministic tests with representative provider chunks/events.
- For OpenAI-compatible streams, cover content chunks, tool-call chunks, usage chunks, finish reasons, and tagged reasoning.
- For Bedrock streams, cover content block start, text deltas, reasoning deltas, tool-use deltas, metadata usage, and message stop.
- Run local tests and live streaming tests when credentials are configured.

Do not make callers parse provider-specific stream JSON to reconstruct common events.

## Changing Settings

Settings compatibility matters because this crate is meant to consume shared VectorVein model catalogs.

Checklist:

- Keep unknown fields through `extra` maps where present.
- Keep string and object endpoint binding support.
- Preserve transport metadata: `endpoint_type`, `region`, `is_bedrock`, `is_vertex`, `credentials`.
- Preserve model metadata: `context_length`, `max_output_tokens`, `function_call_available`, `response_format_available`, `native_multimodal`, `max_image_dimension`, `protocol`, `request_mapping`, and `response_mapping`.
- Add tests for any new field or resolution rule.
- Avoid changing resolution order unless the behavior change is intentional and documented.

## Changing Tool Calls

Tool calling must support both assistant tool-call turns and tool-result turns.

Checklist:

- Keep `ChatTool` provider-neutral.
- Keep `ToolCall.arguments` as a JSON string so callers can forward or parse it consistently.
- Test initial tool-call requests.
- Test assistant messages that include prior tool calls.
- Test `MessageRole::Tool` messages with `tool_call_id`.
- For streaming tool calls, test incremental argument assembly or normalized chunks for each provider path.

## Changing Multimodal Support

Checklist:

- Use `MessageContent::ImageUrl` for images.
- Preserve text and image ordering inside user messages when the provider supports it.
- Use data URLs when a provider requires inline image bytes.
- Keep image decoding and media-type mapping inside the provider adapter.
- Add request-shape tests and, when possible, a live image-understanding test.

## Changing Vertex Or Bedrock Auth

Checklist:

- Read `docs/SECURITY.md`.
- Do not log credentials or token response bodies.
- Keep refreshed Vertex access tokens in memory only.
- Keep Bedrock AWS credentials inside endpoint config and SDK credential providers.
- Test missing credential fields with configuration errors.
- Test token cache behavior without making live calls.
- Run live tests only with confirmed credentials.

## Changing Token Counting

Checklist:

- Prefer `tiktoken-rs` for exact encodings when available.
- Keep deterministic fallback behavior for unknown models.
- Add tests for model-family routing and at least one known count.
- Document any approximation.

## Documentation Updates

Update documentation in the same change when:

- Public API changes.
- Provider support changes.
- Settings shape or resolution behavior changes.
- Streaming normalization changes.
- Live-test setup changes.
- Credential or auth behavior changes.

README files should explain usage. `docs/` should explain engineering behavior and maintenance rules.
