#!/bin/bash
# Wait for the current frozen B/C runs, then launch scratch LoRA B/C locally.
# Usage: scripts/queue_dual_init_lora_scratch_local.sh B_PID C_PID GPU_B GPU_C [ASSET_ROOT]
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "usage: $0 B_PID C_PID GPU_B GPU_C [ASSET_ROOT]" >&2
    exit 2
fi

B_PID="$1"
C_PID="$2"
GPU_B="$3"
GPU_C="$4"
ASSET_ROOT="${5:-/mnt/disk/wl/MoRE-tri/CoopDiff}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
B_RUN="$PROJECT_ROOT/results/dual_init/arm_B_online_c1m1_seed73"
C_RUN="$PROJECT_ROOT/results/dual_init/arm_C_online_c1m1_seed73"
LORA_ROOT="$PROJECT_ROOT/results/dual_init_lora_scratch"
LOG_ROOT="$PROJECT_ROOT/logs"

if [[ "$GPU_B" == "$GPU_C" ]]; then
    echo "FATAL: B and C require distinct GPUs" >&2
    exit 2
fi
if [[ -e "$LORA_ROOT/arm_B_lora_scratch_seed73" \
        || -e "$LORA_ROOT/arm_C_lora_scratch_seed73" ]]; then
    echo "FATAL: a scratch LoRA output directory already exists" >&2
    exit 2
fi

mkdir -p "$LOG_ROOT"
printf '%s\n' "$$" >"$LOG_ROOT/dual-init-lora-scratch-queue.pid"
echo "$(date -Is) waiting for frozen B pid=$B_PID and C pid=$C_PID"
while [[ ! -f "$B_RUN/tricoopdiff_zero.pt" \
         || ! -f "$C_RUN/tricoopdiff_zero.pt" ]]; do
    if [[ ! -f "$B_RUN/tricoopdiff_zero.pt" ]] && ! kill -0 "$B_PID" 2>/dev/null; then
        sleep 10
        if [[ ! -f "$B_RUN/tricoopdiff_zero.pt" ]]; then
            echo "FATAL: frozen B exited without its completion artifact" >&2
            exit 1
        fi
    fi
    if [[ ! -f "$C_RUN/tricoopdiff_zero.pt" ]] && ! kill -0 "$C_PID" 2>/dev/null; then
        sleep 10
        if [[ ! -f "$C_RUN/tricoopdiff_zero.pt" ]]; then
            echo "FATAL: frozen C exited without its completion artifact" >&2
            exit 1
        fi
    fi
    sleep 30
done

echo "$(date -Is) both frozen runs complete; launching scratch LoRA arms"
"$PROJECT_ROOT/scripts/run_dual_init_lora_scratch_arm_local.sh" \
    B "$GPU_B" "$ASSET_ROOT" >"$LOG_ROOT/dual-init-lora-scratch-B.log" 2>&1 &
LORA_B_PID=$!
printf '%s\n' "$LORA_B_PID" >"$LOG_ROOT/dual-init-lora-scratch-B.pid"

"$PROJECT_ROOT/scripts/run_dual_init_lora_scratch_arm_local.sh" \
    C "$GPU_C" "$ASSET_ROOT" >"$LOG_ROOT/dual-init-lora-scratch-C.log" 2>&1 &
LORA_C_PID=$!
printf '%s\n' "$LORA_C_PID" >"$LOG_ROOT/dual-init-lora-scratch-C.pid"

echo "$(date -Is) started B pid=$LORA_B_PID gpu=$GPU_B; C pid=$LORA_C_PID gpu=$GPU_C"
set +e
wait "$LORA_B_PID"
B_STATUS=$?
wait "$LORA_C_PID"
C_STATUS=$?
set -e
echo "$(date -Is) scratch LoRA finished B_status=$B_STATUS C_status=$C_STATUS"
if [[ "$B_STATUS" -ne 0 || "$C_STATUS" -ne 0 ]]; then
    exit 1
fi
