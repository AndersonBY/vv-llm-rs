#!/usr/bin/env python3
"""Regression tests for the Rust repository's contract synchronization guard."""

from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import sync_contract  # noqa: E402


class SyncContractTests(unittest.TestCase):
    def test_check_accepts_explicit_environment_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "source"
            sync_contract.shutil.copytree(sync_contract.VENDOR, source)
            with patch.dict(
                sync_contract.os.environ,
                {"VV_LLM_CONTRACT_SOURCE": str(source)},
                clear=True,
            ):
                with patch.object(
                    sync_contract,
                    "parse_args",
                    return_value=Namespace(source=None, check=True),
                ):
                    self.assertEqual(sync_contract.main(), 0)

    def test_sync_requires_explicit_source(self) -> None:
        with patch.dict(sync_contract.os.environ, {}, clear=True):
            with patch.object(
                sync_contract,
                "parse_args",
                return_value=Namespace(source=None, check=False),
            ):
                self.assertEqual(sync_contract.main(), 1)

    def test_tampered_lock_fails_before_json_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tree = Path(temporary_directory) / "contract"
            sync_contract.shutil.copytree(sync_contract.VENDOR, tree)
            (tree / "consumer-lock.v1.json").write_text(
                "{not-valid-json\n", encoding="utf-8"
            )

            with self.assertRaises(sync_contract.ContractError) as context:
                sync_contract.verify_tree(tree)

            self.assertIn("consumer lock SHA-256 mismatch", str(context.exception))

    def test_vendor_rejects_extra_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tree = Path(temporary_directory) / "contract"
            sync_contract.shutil.copytree(sync_contract.VENDOR, tree)
            (tree / "schemas" / "not-locked.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(sync_contract.ContractError) as context:
                sync_contract.verify_tree(tree, reject_extra_files=True)

            self.assertIn("not listed in consumer lock", str(context.exception))


if __name__ == "__main__":
    unittest.main()
