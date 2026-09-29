# Changelog

## 0.7.0 - 2026-09-29

- Adopt vv-llm-contract 1.2.0 and catalog revision 10 with per-model effort choices, aliases, and endpoint capability overrides.
- Add effort validation policies to OpenAI-compatible, Anthropic, Bedrock, and Vertex adapters and preserve requested effort during fallback.
- Preserve thinking budgets and explicit legacy capability overrides, validate malformed capability metadata, and keep settings round trips stable.
- Reject conflicting wire-model or reasoning controls and unsupported Responses endpoints.

### Compatibility

Warn remains the default policy; conflicting controls fail even with passthrough. ModelCapabilities has additional public fields, so exhaustive struct literals must include them or use ..Default::default(). Existing last-write-wins parameter overrides must be reconciled by callers.
