#!/usr/bin/env bash
# Supervisor: keeps the orchestrator alive. If run_all.sh exits before every
# experiment's JSONL has at least samples_per_question * scorable_count records,
# we re-launch it. Each experiment is resumable (already_predicted) so restarts
# only replay un-written rows. Safe to re-run by hand anytime.

set -u
cd "$(dirname "$0")/.."

SUP_LOG="evaluation_plan/output/run_logs/supervisor.log"
mkdir -p evaluation_plan/output/run_logs

TARGET_PER_EXP=515   # 103 scorable questions × 5 samples
EXPERIMENTS=(e1 e1p e2 e3 e4 e5)

echo "[supervisor] start $(date -u)" >> "$SUP_LOG"

attempt=0
while true; do
    attempt=$((attempt+1))
    echo "[supervisor] launch attempt #$attempt $(date -u)" >> "$SUP_LOG"
    bash evaluation_plan/run_all.sh >> "$SUP_LOG" 2>&1
    RC=$?
    echo "[supervisor] run_all exited rc=$RC $(date -u)" >> "$SUP_LOG"

    # Check if every experiment has hit its target count.
    done_all=1
    for EXP in "${EXPERIMENTS[@]}"; do
        F="pipeline/output/predictions/${EXP}/predictions.jsonl"
        if [ ! -f "$F" ]; then done_all=0; break; fi
        n=$(wc -l < "$F" | tr -d ' ')
        if [ "$n" -lt "$TARGET_PER_EXP" ]; then done_all=0; fi
        echo "[supervisor]   ${EXP}: $n / $TARGET_PER_EXP" >> "$SUP_LOG"
    done

    if [ $done_all -eq 1 ]; then
        echo "[supervisor] all experiments at target; exit $(date -u)" >> "$SUP_LOG"
        break
    fi

    if [ $attempt -ge 20 ]; then
        echo "[supervisor] hit max attempts; giving up $(date -u)" >> "$SUP_LOG"
        break
    fi

    echo "[supervisor] restarting in 30s..." >> "$SUP_LOG"
    sleep 30
done
