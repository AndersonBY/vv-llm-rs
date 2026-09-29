# Changelog

## 0.7.1 - 2026-09-29

- Adopt catalog revision 13 with Claude Opus 5.5, Claude Sonnet 5.5, Gemini 3.8 Flash, GPT-6 Sol, and GPT-6 Luna.

## 0.7.0 - 2026-09-29

- Adopt vv-llm-contract 1.2.0 and catalog revision 10 with per-model effort choices, aliases, and endpoint capability overrides.
- Add effort validation policies to OpenAI-compatible, Anthropic, Bedrock, and Vertex adapters and preserve requested effort during fallback.
- Preserve thinking budgets and explicit legacy capability overrides, validate malformed capability metadata, and keep settings round trips stable.
- Reject conflicting wire-model or reasoning controls and unsupported Responses endpoints.

### Compatibility

Warn remains the default policy; conflicting controls fail even with passthrough. ModelCapabilities has additional public fields, so exhaustive struct literals must include them or use ..Default::default(). Existing last-write-wins parameter overrides must be reconciled by callers.
