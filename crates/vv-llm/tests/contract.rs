use std::fmt::Write as _;

use serde_json::Value;
use sha2::{Digest, Sha256};
use vv_llm::{
    contract_consumer_lock_json, contract_manifest_json, contract_metadata, ContractMetadata,
    CONTRACT_CATALOG_REVISION, CONTRACT_CONSUMER_LOCK_SHA256, CONTRACT_FIXTURE_VERSION,
    CONTRACT_SCHEMA_VERSION, CONTRACT_VERSION,
};

const CONTRACT_CONSUMER_LOCK_BYTES: &[u8] =
    include_bytes!("../contract/v1.3.6/consumer-lock.v1.json");

#[test]
fn catalog_exposes_gpt_6_1_terra() {
    let catalog: Value = serde_json::from_str(include_str!(
        "../contract/v1.3.6/catalog/default-chat-catalog.json"
    ))
    .expect("catalog JSON");
    let terra = &catalog["backends"]["openai"]["models"]["gpt-6.1-terra"];
    assert_eq!(terra["context_length"], 1_050_000);
    assert_eq!(terra["max_output_tokens"], 128_000);
    assert_eq!(terra["native_multimodal"], true);
    assert_eq!(terra["function_call_available"], true);
    assert_eq!(terra["response_format_available"], true);
    assert_eq!(
        terra["capabilities"],
        serde_json::json!({
            "tools": true,
            "structured_output": "json_schema",
            "input_modalities": ["text", "image"],
            "reasoning_efforts": ["none", "low", "medium", "high", "xhigh", "max"]
        })
    );
}

#[test]
fn catalog_exposes_gemini_4_carbon() {
    let catalog: Value = serde_json::from_str(include_str!(
        "../contract/v1.3.6/catalog/default-chat-catalog.json"
    ))
    .expect("catalog JSON");
    let argon = &catalog["backends"]["gemini"]["models"]["gemini-4-argon"];
    let carbon = &catalog["backends"]["gemini"]["models"]["gemini-4-carbon"];
    let mut expected = argon.clone();
    expected["id"] = Value::String("gemini-4-carbon".to_string());
    assert_eq!(carbon, &expected);
}

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
    assert_eq!(contract_metadata().version, "1.3.6");
}

#[test]
fn embeds_manifest_and_consumer_lock() {
    let manifest: Value = serde_json::from_str(contract_manifest_json()).expect("manifest JSON");
    let lock: Value =
        serde_json::from_str(contract_consumer_lock_json()).expect("consumer lock JSON");

    assert_eq!(manifest["contract_version"], "1.3.6");
    assert_eq!(manifest["schema_version"], 2);
    assert_eq!(manifest["fixture_version"], 2);
    assert_eq!(manifest["catalog_revision"], 24);
    assert_eq!(lock["contract_version"], "1.3.6");
    assert_eq!(lock["schema_version"], 2);
    assert_eq!(lock["fixture_version"], 2);
    assert_eq!(lock["catalog_revision"], 24);
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
