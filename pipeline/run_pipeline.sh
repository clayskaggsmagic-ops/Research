#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# run_pipeline.sh — Run the pipeline INDEPENDENTLY of any agent.
#
# This script runs detached (nohup) so even if the terminal or agent
# dies, the pipeline keeps going. All progress is checkpointed.
#
# Usage:
#   ./run_pipeline.sh              # Resume from last checkpoint
#   ./run_pipeline.sh --fresh      # Start from scratch
#   ./run_pipeline.sh --tail       # Resume + follow the live log
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/tmp/pipeline_venv/bin/python"
LOG="/tmp/pipeline_output/pipeline.log"
PID_FILE="/tmp/pipeline_output/pipeline.pid"
ENV_FILE="/tmp/pipeline_output/.env"

# ── Pre-flight checks ────────────────────────────────────────────
if [[ ! -x "$PYTHON" ]]; then
    echo "❌ Python venv not found at $PYTHON"
    echo "   Run: UV_CACHE_DIR=/tmp/uv-cache uv venv --python 3.12 /tmp/pipeline_venv"
    echo "   Then: UV_CACHE_DIR=/tmp/uv-cache uv pip install --python /tmp/pipeline_venv/bin/python -e ."
    exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
    echo "❌ No .env file at $ENV_FILE"
    echo "   Create it with GOOGLE_API_KEY and TAVILY_API_KEY"
    exit 1
fi

# Source env vars
set -a
source "$ENV_FILE"
set +a

# Check keys
if [[ -z "${GOOGLE_API_KEY:-}" ]]; then
    echo "❌ GOOGLE_API_KEY not set in $ENV_FILE"
    exit 1
fi
if [[ -z "${TAVILY_API_KEY:-}" ]]; then
    echo "❌ TAVILY_API_KEY not set in $ENV_FILE"
    exit 1
fi

# ── Check if already running ─────────────────────────────────────
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "⚠️  Pipeline already running (PID $OLD_PID)"
        echo "   Tail the log:  tail -f $LOG"
        echo "   Kill it:       kill $OLD_PID"
        exit 0
    else
        rm -f "$PID_FILE"
    fi
fi

# ── Parse args ───────────────────────────────────────────────────
EXTRA_ARGS=""
TAIL=false
for arg in "$@"; do
    if [[ "$arg" == "--tail" ]]; then
        TAIL=true
    else
        EXTRA_ARGS="$EXTRA_ARGS $arg"
    fi
done

# ── Ensure output dir ────────────────────────────────────────────
mkdir -p /tmp/pipeline_output

# ── Launch detached ──────────────────────────────────────────────
echo "🚀 Launching pipeline (detached)..."
echo "   Log: $LOG"
echo "   PID file: $PID_FILE"

nohup "$PYTHON" "$SCRIPT_DIR/run_all.py" \
    --concurrency 40 --batch-size 40 \
    $EXTRA_ARGS \
    >> "$LOG" 2>&1 &

echo $! > "$PID_FILE"
PID=$(cat "$PID_FILE")
echo "   PID: $PID"
echo ""
echo "✅ Pipeline is running in the background."
echo "   It will save progress after every batch — safe to close this terminal."
echo ""
echo "   📋 Commands:"
echo "     tail -f $LOG          # Watch live progress"
echo "     kill $PID             # Stop it"
echo "     cat $PID_FILE         # Check PID"
echo ""

if $TAIL; then
    echo "── Following log (Ctrl-C to stop watching, pipeline keeps running) ──"
    tail -f "$LOG"
fi
