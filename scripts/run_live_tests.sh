#!/usr/bin/env bash
set -euo pipefail

if [[ "${VV_LLM_RUN_LIVE_TESTS:-}" != "1" && "${VV_LLM_RUN_LIVE_TESTS:-}" != "true" && "${VV_LLM_RUN_LIVE_TESTS:-}" != "yes" && "${VV_LLM_RUN_LIVE_TESTS:-}" != "on" ]]; then
  echo "Live tests are disabled. Set VV_LLM_RUN_LIVE_TESTS=1 to run."
  exit 1
fi

cargo test --test live_tests -- --ignored --test-threads=1

