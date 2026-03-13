#!/bin/bash
# Background monitor for EB-ALFRED training
LOG_DIR="/root/workspace/VAGEN-eb-alfred/exps/vagen_experiments/grpo_eb_alfred_qwen25vl3b"

while true; do
    sleep 30

    unity_count=$(pgrep -cf "thor-Linux64" 2>/dev/null || echo 0)
    mem_percent=$(free | awk '/Mem:/{printf "%.0f",$3/$2*100}')

    # Kill old Unity processes if too many
    if [ "$unity_count" -gt 40 ]; then
        echo "[$(date)] WARN: ${unity_count} Unity procs, cleaning old ones"
        pgrep -f "thor-Linux64" 2>/dev/null | while read pid; do
            age=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')
            if [ -n "$age" ] && [ "$age" -gt 600 ]; then
                echo "[$(date)] Kill stale Unity pid=$pid age=${age}s"
                kill -9 "$pid" 2>/dev/null
            fi
        done
    fi

    # Emergency memory cleanup
    if [ "$mem_percent" -gt 85 ]; then
        echo "[$(date)] CRITICAL: Memory at ${mem_percent}%, killing all Unity"
        pkill -9 -f "thor-Linux64" 2>/dev/null
        sleep 5
    fi

    # Restart dead Xorg displays
    for d in 0 1; do
        lockfile="/tmp/.X${d}-lock"
        if [ -f "$lockfile" ]; then
            xpid=$(cat "$lockfile" 2>/dev/null | tr -d ' ')
            if ! kill -0 "$xpid" 2>/dev/null; then
                echo "[$(date)] WARN: Xorg display :${d} died (pid=$xpid), restarting..."
                rm -f "$lockfile" "/tmp/.X11-unix/X${d}"
                conf="/tmp/xorg-gpu${d}.conf"
                if [ -f "$conf" ]; then
                    Xorg ":${d}" -config "$conf" -noreset -logfile "/tmp/Xorg.${d}.log" &
                    sleep 3
                    echo "[$(date)] Xorg :${d} restarted"
                fi
            fi
        fi
    done

    # Check env server
    if ! curl -s --max-time 10 http://localhost:8000/health > /dev/null 2>&1; then
        echo "[$(date)] WARN: Env server not responding to health check"
    fi

    # Status report
    sessions=$(curl -s --max-time 5 http://localhost:8000/sessions 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin).get('num_sessions','?'))" 2>/dev/null || echo "?")
    gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr '\n' '/' || echo "?")
    echo "[$(date)] OK: mem=${mem_percent}% unity=${unity_count} sess=${sessions} gpu=[${gpu_mem}]"
done
