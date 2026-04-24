#!/usr/bin/env bash
# Verifier: overlay trusted tests (preserving agent's adapters.py), run pytest,
# write binary reward to /logs/verifier/reward.txt.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="/app"
LOGS_DIR="/logs/verifier"
mkdir -p "$LOGS_DIR"

echo "=== CS336 verifier ==="
echo "0" > "$LOGS_DIR/reward.txt"

nvidia-smi 2>&1 | head -20 | tee "$LOGS_DIR/gpu_check.txt" || echo "nvidia-smi unavailable"

mkdir -p "$WORKSPACE/tests"
AGENT_ADAPTERS_BACKUP="$(mktemp)"
if [ -f "$WORKSPACE/tests/adapters.py" ]; then
    cp "$WORKSPACE/tests/adapters.py" "$AGENT_ADAPTERS_BACKUP"
    HAS_AGENT_ADAPTERS=1
else
    HAS_AGENT_ADAPTERS=0
fi

rsync -a --delete "$SCRIPT_DIR/trusted_tests/" "$WORKSPACE/tests/"

if [ "$HAS_AGENT_ADAPTERS" = "1" ]; then
    cp "$AGENT_ADAPTERS_BACKUP" "$WORKSPACE/tests/adapters.py"
    echo "[verifier] using agent tests/adapters.py"
else
    echo "[verifier] agent provided no tests/adapters.py — using trusted stub (all tests will fail)"
fi
rm -f "$AGENT_ADAPTERS_BACKUP"

cd "$WORKSPACE"
set +e
python3 -m pytest tests/ \
    --tb=short \
    --json-report \
    --json-report-file="$LOGS_DIR/pytest_report.json" \
    2>&1 | tee "$LOGS_DIR/pytest.log"
PYTEST_EXIT=$?
set -e

python3 - <<'PY' || true
import json, os
report = "/logs/verifier/pytest_report.json"
metrics = {"total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0, "pytest_exit": None}
if os.path.exists(report):
    try:
        with open(report) as f:
            r = json.load(f)
        s = r.get("summary", {})
        metrics.update({
            "total": s.get("total", 0),
            "passed": s.get("passed", 0),
            "failed": s.get("failed", 0),
            "errors": s.get("error", 0),
            "skipped": s.get("skipped", 0),
        })
    except Exception as e:
        metrics["parse_error"] = str(e)
with open("/logs/verifier/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
PY

if [ "$PYTEST_EXIT" = "0" ]; then
    echo "1" > "$LOGS_DIR/reward.txt"
    echo "[verifier] all tests passed → reward 1"
else
    echo "0" > "$LOGS_DIR/reward.txt"
    echo "[verifier] pytest exit=$PYTEST_EXIT → reward 0"
fi

echo "=== metrics ==="
cat "$LOGS_DIR/metrics.json"
echo ""
echo "=== reward ==="
cat "$LOGS_DIR/reward.txt"
