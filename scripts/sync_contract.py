#!/usr/bin/env python3
"""Synchronize and verify the vendored vv-llm contract artifact tree.

Use ``--source`` or ``VV_LLM_CONTRACT_SOURCE`` when synchronizing from an
explicit contract release artifact directory. ``--check`` without a source is intentionally
offline and verifies only the vendored tree against its consumer lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Any


CONTRACT_VERSION = "1.0.0"
CONTRACT_METADATA = {
    "contract_version": CONTRACT_VERSION,
    "schema_version": 2,
    "fixture_version": 2,
    "catalog_revision": 1,
}
CONTRACT_CONSUMER_LOCK_SHA256 = (
    "a9ebd65253635e84564b971227f30ec2c81b35096100ef26034273eec3f54188"
)
ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "crates" / "vv-llm" / "contract" / f"v{CONTRACT_VERSION}"
ROOT_ARTIFACTS = ("manifest.json", "checksums.sha256", "consumer-lock.v1.json")


class ContractError(RuntimeError):
    """Raised when a contract tree does not satisfy its consumer lock."""


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object in {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ContractError(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def safe_relative_path(relative: str) -> Path:
    """Convert a lock's POSIX artifact path without permitting traversal."""

    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ContractError(f"unsafe artifact path in consumer lock: {relative!r}")
    return Path(*path.parts)


def artifact_members(lock: dict[str, Any]) -> list[str]:
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ContractError("consumer lock must contain a non-empty artifacts object")
    members: list[str] = []
    for relative, expected in artifacts.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ContractError("consumer lock artifact paths and hashes must be strings")
        safe_relative_path(relative)
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise ContractError(f"invalid SHA-256 for {relative!r}")
        members.append(relative)
    return sorted(members)


def parse_checksums(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ContractError(f"cannot read checksum index {path}: {exc}") from exc

    checksums: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ContractError(f"invalid checksum line {path}:{line_number}")
        digest, relative = parts
        if any(char not in "0123456789abcdef" for char in digest):
            raise ContractError(f"invalid checksum line {path}:{line_number}")
        safe_relative_path(relative)
        if relative in checksums:
            raise ContractError(f"duplicate checksum entry for {relative!r}")
        checksums[relative] = digest
    return checksums


def verify_tree(base: Path, *, reject_extra_files: bool = False) -> list[str]:
    """Verify one contract tree and return all files covered by the lock."""

    if not base.is_dir():
        raise ContractError(f"contract tree does not exist: {base}")

    lock_path = base / "consumer-lock.v1.json"
    manifest_path = base / "manifest.json"
    checksums_path = base / "checksums.sha256"
    actual_lock_sha256 = sha256(lock_path)
    if actual_lock_sha256 != CONTRACT_CONSUMER_LOCK_SHA256:
        raise ContractError(
            "consumer lock SHA-256 mismatch: "
            f"expected {CONTRACT_CONSUMER_LOCK_SHA256}, got {actual_lock_sha256}"
        )
    lock = read_json(lock_path)
    if lock.get("format") != "vv-llm-contract-consumer-lock.v1":
        raise ContractError(f"unsupported consumer lock format in {lock_path}")
    for key, expected in CONTRACT_METADATA.items():
        if lock.get(key) != expected:
            raise ContractError(f"expected {key}={expected!r} in {lock_path}")

    members = artifact_members(lock)
    expected_artifacts = lock["artifacts"]
    expected_roots = {
        "manifest.json": lock.get("manifest_sha256"),
        "checksums.sha256": lock.get("checksums_sha256"),
    }
    for relative, expected in expected_roots.items():
        if not isinstance(expected, str) or len(expected) != 64:
            raise ContractError(f"missing or invalid {relative} hash in {lock_path}")
        actual = sha256(base / relative)
        if actual != expected:
            raise ContractError(f"{relative} hash mismatch: expected {expected}, got {actual}")

    manifest = read_json(manifest_path)
    for key, expected in CONTRACT_METADATA.items():
        if manifest.get(key) != expected:
            raise ContractError(f"manifest {key} does not match consumer lock")
    if manifest.get("checksums") != "checksums.sha256":
        raise ContractError("manifest must point to checksums.sha256")

    manifest_artifacts = manifest.get("artifacts")
    if not isinstance(manifest_artifacts, dict):
        raise ContractError("manifest must contain an artifacts object")
    manifest_paths: list[str] = []
    for category in ("schemas", "fixtures", "catalog"):
        paths = manifest_artifacts.get(category)
        if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
            raise ContractError(f"manifest artifacts.{category} must be a string list")
        manifest_paths.extend(paths)
    manifest_paths.sort()
    if manifest_paths != members:
        raise ContractError("manifest artifacts do not exactly match consumer lock")

    checksums = parse_checksums(checksums_path)
    if checksums != expected_artifacts:
        raise ContractError("checksums.sha256 does not exactly match consumer lock artifacts")

    for relative in members:
        path = base / safe_relative_path(relative)
        expected = expected_artifacts[relative]
        actual = sha256(path)
        if actual != expected:
            raise ContractError(f"{relative} hash mismatch: expected {expected}, got {actual}")

    if reject_extra_files:
        expected_files = set(ROOT_ARTIFACTS) | set(members)
        actual_files = {
            path.relative_to(base).as_posix()
            for path in base.rglob("*")
            if path.is_file()
        }
        extras = sorted(actual_files - expected_files)
        if extras:
            raise ContractError(
                "vendored contract contains files not listed in consumer lock: "
                + ", ".join(extras)
            )

    return [*ROOT_ARTIFACTS, *members]


def compare_trees(source: Path, vendor: Path, members: list[str]) -> None:
    for relative in members:
        source_path = source / safe_relative_path(relative)
        vendor_path = vendor / safe_relative_path(relative)
        try:
            source_bytes = source_path.read_bytes()
            vendor_bytes = vendor_path.read_bytes()
        except OSError as exc:
            raise ContractError(f"cannot compare {relative}: {exc}") from exc
        if source_bytes != vendor_bytes:
            raise ContractError(f"vendored contract differs from source for {relative}")


def sync_tree(source: Path, vendor: Path, members: list[str]) -> None:
    vendor.mkdir(parents=True, exist_ok=True)
    for relative in members:
        source_path = source / safe_relative_path(relative)
        vendor_path = vendor / safe_relative_path(relative)
        vendor_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(source_path, vendor_path)
        except OSError as exc:
            raise ContractError(f"cannot vendor {relative}: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        help="explicit contract release artifact directory to sync/check",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify without writing; without --source, verify vendor from its lock",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_value = args.source or os.environ.get("VV_LLM_CONTRACT_SOURCE")
    source = Path(source_value).resolve() if source_value else None

    try:
        if args.check and source is None:
            verify_tree(VENDOR, reject_extra_files=True)
            print(f"contract check passed from vendored lock: {VENDOR}")
            return 0

        if source is None:
            raise ContractError(
                "synchronization requires --source or VV_LLM_CONTRACT_SOURCE; "
                "use --check for an offline vendored-lock check"
            )
        if not source.is_dir():
            raise ContractError(f"specified source does not exist: {source}")

        members = verify_tree(source)
        if args.check:
            compare_trees(source, VENDOR, members)
            verify_tree(VENDOR, reject_extra_files=True)
            print(f"contract check passed: {source} == {VENDOR}")
            return 0

        sync_tree(source, VENDOR, members)
        verify_tree(VENDOR, reject_extra_files=True)
        print(f"contract synchronized: {source} -> {VENDOR}")
        return 0
    except ContractError as exc:
        print(f"contract check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
