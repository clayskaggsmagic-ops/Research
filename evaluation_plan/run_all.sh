#!/usr/bin/env bash
# Autonomous orchestrator: runs all 6 experiments sequentially with progress
# logging. Safe to run under `caffeinate -disu` so the Mac can stay awake with
# the lid closed. Each experiment is resumable (JSONL append + already_predicted)
# so if the box is reopened mid-run, `bash run_all.sh` resumes cleanly.

set -u
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1

# Load env (GOOGLE_API_KEY, TAVILY_API_KEY, DATABASE_URL).
set -a
# shellcheck disable=SC1091
source temporal_knowledge_base/.env
set +a

LOG_DIR="evaluation_plan/output/run_logs"
mkdir -p "$LOG_DIR"

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
MAIN_LOG="$LOG_DIR/run_${RUN_TS}.log"

{
echo "==============================================================="
echo "AUTONOMOUS EXPERIMENT RUN"
echo "started: $(date -u)"
echo "host:    $(hostname)"
echo "pwd:     $(pwd)"
echo "pid:     $$"
echo "log:     $MAIN_LOG"
echo "==============================================================="
} | tee -a "$MAIN_LOG"

# Order: cheap/no-briefing first (validates API), then briefing-heavy.
# E4 reuses E1's briefing cache so E1 must run first.
EXPERIMENTS=(e3 e5 e1 e4 e1p e2)

for EXP in "${EXPERIMENTS[@]}"; do
    EXP_LOG="$LOG_DIR/${EXP}_${RUN_TS}.log"
    {
    echo ""
    echo "--- [$EXP] start $(date -u) ---"
    } | tee -a "$MAIN_LOG"

    temporal_knowledge_base/.venv/bin/python3 -m evaluation_plan.src.run_experiment --experiment "$EXP" \
        --concurrency 8 \
        >> "$EXP_LOG" 2>&1
    RC=$?

    {
    echo "--- [$EXP] end $(date -u) rc=$RC ---"
    tail -n 3 "$EXP_LOG" | sed "s/^/  [$EXP] /"
    } | tee -a "$MAIN_LOG"

    if [ $RC -ne 0 ]; then
        echo "[$EXP] FAILED (rc=$RC) — check $EXP_LOG" | tee -a "$MAIN_LOG"
        # Don't abort the run — later experiments are independent and we want
        # partial progress. Scoring can skip experiments with errors.
    fi
done

{
echo ""
echo "==============================================================="
echo "ALL EXPERIMENTS DONE"
echo "finished: $(date -u)"
echo "==============================================================="
ls -la pipeline/output/predictions/*/predictions.jsonl 2>/dev/null || true
echo "Record counts:"
for EXP in e1 e1p e2 e3 e4 e5; do
    F="pipeline/output/predictions/${EXP}/predictions.jsonl"
    if [ -f "$F" ]; then
        COUNT=$(wc -l < "$F")
        ERRS=$(grep -c '"error": "' "$F" 2>/dev/null || echo 0)
        echo "  $EXP: $COUNT records ($ERRS errors)"
    fi
done
} | tee -a "$MAIN_LOG"
