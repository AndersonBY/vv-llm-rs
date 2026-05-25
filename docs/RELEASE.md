# Release

Rust packages are published to `crates.io`, the public package registry used by Cargo. This repository publishes the `vv-llm` crate from `crates/vv-llm`.

## One-Time Setup

1. Create or sign in to a `crates.io` account.
2. Generate a crates.io API token with publish permission for `vv-llm`.
3. Add the token to the GitHub repository secret `CARGO_REGISTRY_TOKEN`.
4. Create the GitHub Actions environment `crates-io` if the repository requires environment approval.

Do not commit the token. The workflow reads it only from GitHub Secrets.

## Version And Tag Rule

The publish workflow runs only for tags matching:

```text
v*.*.*
```

The tag must match `crates/vv-llm/Cargo.toml` exactly. For example:

```text
version = "0.1.0"
tag     = "v0.1.0"
```

If the tag and crate version differ, the workflow fails before publishing.

## Release Steps

Run from the repository root:

```bash
cargo fmt --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo publish --manifest-path crates/vv-llm/Cargo.toml --dry-run
```

Then create and push the matching tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

GitHub Actions will run local checks, run `cargo publish --dry-run`, and then publish to crates.io with `CARGO_REGISTRY_TOKEN`.

## Important Notes

- crates.io versions are immutable. If a publish succeeds, do not reuse the same version for different code.
- Update `Cargo.toml` version before each release.
- Live API tests are not part of the publish workflow because they require provider credentials and call paid APIs.
- `crates/vv-llm/tests/fixtures/dev_settings.json` must remain untracked and must never be uploaded.
