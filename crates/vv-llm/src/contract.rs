//! Metadata and embedded artifacts for the language-neutral vv-llm contract.
//!
//! The contract files are vendored under `contract/v1.0.1` and verified by
//! `scripts/sync_contract.py`. Runtime consumers can inspect the pinned
//! metadata without needing to locate files on disk.

/// Contract release represented by the vendored artifact tree.
pub const CONTRACT_VERSION: &str = "1.0.1";
/// Version of the language-neutral JSON schemas in this contract release.
pub const CONTRACT_SCHEMA_VERSION: u32 = 2;
/// Version of the deterministic protocol/settings fixtures in this release.
pub const CONTRACT_FIXTURE_VERSION: u32 = 2;
/// Revision of the default model catalog in this release.
pub const CONTRACT_CATALOG_REVISION: u32 = 2;
/// SHA-256 pin for the exact consumer lock embedded in this crate.
pub const CONTRACT_CONSUMER_LOCK_SHA256: &str =
    "3407cc7d398885284f32c453a8e71c6dbb2f40a10eb0cc9f2d21a0a7c7dc6b49";

/// A compact, provider-neutral description of the vendored contract release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContractMetadata {
    pub version: &'static str,
    pub schema_version: u32,
    pub fixture_version: u32,
    pub catalog_revision: u32,
}

/// Metadata for the contract consumed by this crate.
pub const CONTRACT_METADATA: ContractMetadata = ContractMetadata {
    version: CONTRACT_VERSION,
    schema_version: CONTRACT_SCHEMA_VERSION,
    fixture_version: CONTRACT_FIXTURE_VERSION,
    catalog_revision: CONTRACT_CATALOG_REVISION,
};

/// The vendored contract manifest, embedded for diagnostics and tooling.
pub const CONTRACT_MANIFEST_JSON: &str = include_str!("../contract/v1.0.1/manifest.json");
/// The vendored consumer lock, embedded for diagnostics and tooling.
pub const CONTRACT_CONSUMER_LOCK_JSON: &str =
    include_str!("../contract/v1.0.1/consumer-lock.v1.json");
/// The vendored artifact checksum index, embedded for diagnostics and tooling.
pub const CONTRACT_CHECKSUMS: &str = include_str!("../contract/v1.0.1/checksums.sha256");

/// Return metadata for the contract consumed by this crate.
pub const fn contract_metadata() -> ContractMetadata {
    CONTRACT_METADATA
}

/// Return the embedded contract manifest JSON.
pub const fn contract_manifest_json() -> &'static str {
    CONTRACT_MANIFEST_JSON
}

/// Return the embedded contract consumer lock JSON.
pub const fn contract_consumer_lock_json() -> &'static str {
    CONTRACT_CONSUMER_LOCK_JSON
}

/// Return the embedded contract checksum index.
pub const fn contract_checksums() -> &'static str {
    CONTRACT_CHECKSUMS
}
