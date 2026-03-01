#!/bin/bash
# Run all 6 EB-ALFRED benchmark experiments with GPT-4.1.
#
# Prerequisites:
#   DISPLAY=:0 conda run -n embench python -m vagen.envs.eb_alfred.serve \
#       --port 8000 --max-sessions 64
#
# Usage:
#   cd VAGEN-eb-alfred
#   bash benchmarking/run_all.sh          # run all 6
#   bash benchmarking/run_all.sh 1        # run only experiment 1
#   bash benchmarking/run_all.sh 1 2      # run experiments 1 and 2

set -e
cd "$(dirname "$0")/.."

CONFIGS=(
    "benchmarking/configs/serial_500_10ep.yaml"
    "benchmarking/configs/parallel_500_128ep.yaml"
    "benchmarking/configs/serial_328_10ep.yaml"
    "benchmarking/configs/parallel_328_128ep.yaml"
    "benchmarking/configs/serial_96_10ep.yaml"
    "benchmarking/configs/parallel_96_128ep.yaml"
)
LABELS=(
    "Exp1: serial  500x500  10ep"
    "Exp2: parallel 500x500 128ep"
    "Exp3: serial  328x328  10ep"
    "Exp4: parallel 328x328 128ep"
    "Exp5: serial   96x96   10ep"
    "Exp6: parallel  96x96  128ep"
)

# If arguments given, run only those experiments
if [ $# -gt 0 ]; then
    INDICES=("$@")
else
    INDICES=(1 2 3 4 5 6)
fi

echo "============================================================"
echo "  EB-ALFRED Benchmark Suite (GPT-4.1)"
echo "============================================================"

for idx in "${INDICES[@]}"; do
    i=$((idx - 1))
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  ${LABELS[$i]}"
    echo "  Config: ${CONFIGS[$i]}"
    echo "────────────────────────────────────────────────────────────"
    START=$(date +%s)

    conda run -n vagen python -m vagen.evaluate.run_eval \
        --config "${CONFIGS[$i]}"

    END=$(date +%s)
    ELAPSED=$((END - START))
    echo "  Finished in ${ELAPSED}s ($(echo "scale=1; $ELAPSED/60" | bc) min)"
done

echo ""
echo "============================================================"
echo "  All experiments complete. Generating report..."
echo "============================================================"
conda run -n vagen python benchmarking/aggregate_results.py
echo "Done."
