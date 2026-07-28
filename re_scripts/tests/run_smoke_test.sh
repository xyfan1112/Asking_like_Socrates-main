#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-$(command -v python3 || command -v python)}
"$PY" "$ROOT/tests/smoke_test.py"
"$PY" "$ROOT/tests/test_v4_contracts.py"
"$PY" "$ROOT/tests/test_v432_regressions.py"
"$PY" "$ROOT/tests/test_v433_quality_gates.py"
