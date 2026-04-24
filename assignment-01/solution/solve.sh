#!/usr/bin/env bash
# Oracle solution: overlay the completed cs336 assignment 1 solution onto /app.
# The oracle replaces /app entirely with the reference implementation so that
# the verifier pytest run produces all-passing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[oracle] overlaying reference solution onto /app..."
rsync -a --delete "$SCRIPT_DIR/oracle/" /app/

# Ensure no stale starter files remain that would shadow the solution.
# (rsync --delete above handles this, but be explicit about common offenders.)
rm -rf /app/cs336_basics 2>/dev/null || true

echo "[oracle] solution installed."
ls /app

