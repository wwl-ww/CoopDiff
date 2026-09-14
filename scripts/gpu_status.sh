#!/usr/bin/env bash
# One-shot snapshot of every GPU and the processes using them.

set -u

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "error: nvidia-smi is not available" >&2
    exit 1
fi

echo "GPU snapshot: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo
nvidia-smi \
    --query-gpu=index,name,pstate,temperature.gpu,utilization.gpu,memory.used,memory.total \
    --format=csv

echo
echo "GPU process map:"
nvidia-smi pmon -c 1 -s um

pid_csv=$(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | awk 'NF {print $1}' | sort -nu | paste -sd, -
)

echo
echo "Process details (PPID 1 means detached/background):"
if [[ -z "$pid_csv" ]]; then
    echo "No compute processes found."
else
    ps -ww -p "$pid_csv" \
        -o user:12,pid,ppid,stat,etime,%cpu,%mem,args \
        || true
fi
