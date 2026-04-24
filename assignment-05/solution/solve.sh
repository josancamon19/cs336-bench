#!/usr/bin/env bash
# Oracle: overlay the reference solution onto /app.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "[oracle] overlaying reference solution onto /app..."
rsync -a --delete "$SCRIPT_DIR/oracle/" /app/
echo "[oracle] solution installed."
ls /app
