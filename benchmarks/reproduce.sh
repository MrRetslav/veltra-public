#!/bin/bash
# Repeats the three-way CPU comparison on your own machine. Linux x86-64 (the code tasks of the eval need bwrap or systemd-run).
#
#   VELTRA_BIN=build/veltra-server VELTRA_MODEL=/models/qwen3-30b-a3b.vlt \
#   GGUF=/models/Qwen3-30B-A3B-Instruct-2507-UD-Q2_K_XL.gguf \
#   LLAMA_BIN=/src/llama.cpp/build/bin/llama-server IK_BIN=/src/ik_llama.cpp/build/bin/llama-server \
#   bash benchmarks/reproduce.sh [speed|eval|all] [out dir]
#
# Engines whose variable is unset are skipped. One server runs at a time on 127.0.0.1:8097. THREADS defaults to 8.
# The files get the names benchmarks/results/manifest.json expects. Afterwards (edit "hardware" and "versions" in a copy of the manifest first):
#   python3 benchmarks/assemble.py --raw benchmarks/results/mine --manifest benchmarks/results/manifest.json --out benchmarks/results/mine
#   python3 benchmarks/make_charts.py --results benchmarks/results/mine --out benchmarks/results/mine
set -eu
cd "$(dirname "$0")/.."
WHAT=${1:-all}
OUT=${2:-benchmarks/results/mine}
THREADS=${THREADS:-8}
mkdir -p "$OUT"
declare -a NAMES=() EVALS=() RSS=() CMDS=()
add() { NAMES+=("$1"); EVALS+=("$2"); RSS+=("$3"); CMDS+=("$4"); }   # speed name, eval name, eval RSS name, server command
if [ -n "${VELTRA_BIN:-}" ] && [ -n "${VELTRA_MODEL:-}" ]; then
  add veltra hf30_base base "$VELTRA_BIN --model $VELTRA_MODEL --threads $THREADS --thinking 0"
  add paretomax2 hf30_pmax6s2 pmax6s2 "$VELTRA_BIN --model $VELTRA_MODEL --threads $THREADS --thinking 0 --pareto-max"
fi
if [ -n "${GGUF:-}" ]; then
  [ -n "${LLAMA_BIN:-}" ] && add llamacpp llama llama "$LLAMA_BIN -m $GGUF -t $THREADS -c 4096 -ngl 0 --jinja -np 1 ${LLAMA_FLAGS:-}"
  [ -n "${IK_BIN:-}" ] && add ikllama ik ik "$IK_BIN -m $GGUF -t $THREADS -c 4096 -ngl 0 --jinja -np 1 ${IK_FLAGS:-}"
fi
[ ${#NAMES[@]} -gt 0 ] || { echo "nothing to run: set VELTRA_BIN/VELTRA_MODEL and/or GGUF with LLAMA_BIN, IK_BIN"; exit 2; }
if [ "$WHAT" = speed ] || [ "$WHAT" = all ]; then
  for i in "${!NAMES[@]}"; do
    n=${NAMES[$i]}
    echo "== speed: $n"
    # shellcheck disable=SC2086
    bash benchmarks/night/with_server.sh "$OUT/rss_speed_$n.txt" ${CMDS[$i]} -- python3 benchmarks/night/speed_bench.py --label "$n" --out "$OUT/speed_$n.json" --reps "${REPS:-3}"
  done
fi
if [ "$WHAT" = eval ] || [ "$WHAT" = all ]; then
  for i in "${!NAMES[@]}"; do
    n=${NAMES[$i]}
    e=${EVALS[$i]}
    echo "== eval (160 tasks): $n"
    # shellcheck disable=SC2086
    bash benchmarks/night/with_server.sh "$OUT/rss_${RSS[$i]}.txt" ${CMDS[$i]} -- python3 benchmarks/eval_public/run_eval.py --label "$n" --report "$OUT/eval_$e.json"
  done
fi
echo "raw results in $OUT (see the two commands at the top of this script)"
