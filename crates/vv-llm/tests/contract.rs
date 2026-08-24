use std::fmt::Write as _;

use serde_json::Value;
use sha2::{Digest, Sha256};
use vv_llm::{
    contract_consumer_lock_json, contract_manifest_json, contract_metadata, ContractMetadata,
    CONTRACT_CATALOG_REVISION, CONTRACT_CONSUMER_LOCK_SHA256, CONTRACT_FIXTURE_VERSION,
    CONTRACT_SCHEMA_VERSION, CONTRACT_VERSION,
};

const CONTRACT_CONSUMER_LOCK_BYTES: &[u8] =
    include_bytes!("../contract/v1.0.0/consumer-lock.v1.json");

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}

#[test]
fn exposes_pinned_contract_metadata() {
    assert_eq!(
        contract_metadata(),
        ContractMetadata {
            version: CONTRACT_VERSION,
            schema_version: CONTRACT_SCHEMA_VERSION,
            fixture_version: CONTRACT_FIXTURE_VERSION,
            catalog_revision: CONTRACT_CATALOG_REVISION,
        }
    );
    assert_eq!(contract_metadata().version, "1.0.0");
}

#[test]
fn embeds_manifest_and_consumer_lock() {
    let manifest: Value = serde_json::from_str(contract_manifest_json()).expect("manifest JSON");
    let lock: Value =
        serde_json::from_str(contract_consumer_lock_json()).expect("consumer lock JSON");

    assert_eq!(manifest["contract_version"], "1.0.0");
    assert_eq!(manifest["schema_version"], 2);
    assert_eq!(manifest["fixture_version"], 2);
    assert_eq!(manifest["catalog_revision"], 1);
    assert_eq!(lock["contract_version"], "1.0.0");
    assert_eq!(lock["schema_version"], 2);
    assert_eq!(lock["fixture_version"], 2);
    assert_eq!(lock["catalog_revision"], 1);
    assert!(lock["artifacts"]["catalog/default-chat-catalog.json"].is_string());
    assert!(lock["artifacts"]["fixtures/openai-compatible.v2.json"].is_string());
}

#[test]
fn consumer_lock_sha_pin_matches_str_and_bytes_includes() {
    assert_eq!(
        contract_consumer_lock_json().as_bytes(),
        CONTRACT_CONSUMER_LOCK_BYTES
    );
    assert_eq!(
        sha256_hex(contract_consumer_lock_json().as_bytes()),
        CONTRACT_CONSUMER_LOCK_SHA256
    );
    assert_eq!(
        sha256_hex(CONTRACT_CONSUMER_LOCK_BYTES),
        CONTRACT_CONSUMER_LOCK_SHA256
    );
}
