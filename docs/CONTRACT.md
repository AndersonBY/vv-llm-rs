# vv-llm Contract Artifacts

`vv-llm-rs` consumes the language-neutral `vv-llm-contract` release as a
vendored, locked artifact tree. The current tree is
`crates/vv-llm/contract/v1.0.0/` and contains the manifest, consumer lock,
checksum index, default model catalog, deterministic fixtures, and all
versioned JSON schemas. It contains no credentials or live-test settings.

## Source Selection And Synchronization

The vendored tree is the only implicit input. A check without a source is
offline and validates every bundled file against `consumer-lock.v1.json`:

```bash
python scripts/sync_contract.py --check
```

Synchronization requires an explicit contract release directory. Use either the
CLI argument or the environment variable; the CLI argument takes precedence:

```bash
python scripts/sync_contract.py --source /secure/path/vv-llm-contract/dist/release-v1.0.0
VV_LLM_CONTRACT_SOURCE=/secure/path/vv-llm-contract/dist/release-v1.0.0 python scripts/sync_contract.py
```

The script validates the source lock, manifest, checksum index, and all
artifacts before copying. `--check --source ...` additionally requires the
source and vendored trees to match byte-for-byte. A plain `--check` never
probes for a source checkout or assumes a machine-specific path.
The vendored tree check also rejects files under the vendor root that are not
listed by the consumer lock; source repositories may contain their normal
documentation and release files outside the artifact tree.

## Runtime Access

The crate exposes `contract_metadata()`, `contract_manifest_json()`, and
`contract_consumer_lock_json()` for diagnostics and downstream tooling.
`CONTRACT_CONSUMER_LOCK_SHA256` pins the exact lock bytes embedded in the
crate. The default chat catalog is loaded from the vendored catalog at compile
time, and protocol tests read the vendored OpenAI-compatible fixture directly.

## Release Rules

- Keep `consumer-lock.v1.json`, `manifest.json`, `checksums.sha256`, and every
  lock-listed artifact from one contract release.
- Run the contract check before `cargo fmt --check`, `cargo test`, and clippy.
- Require `--source` or `VV_LLM_CONTRACT_SOURCE` for any write/synchronization.
- Do not add credentials, secret settings, or live provider responses to the
  contract tree.
- Bump the vendored `v<contract_version>` directory and update the runtime
  constants/accessors in one change when consuming a future contract release.
