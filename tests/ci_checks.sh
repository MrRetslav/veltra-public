#!/bin/bash
# Model-level checks on the synthetic CI models (tests/make_ci_model.py). Needs a built veltra-gen / veltra-selftest and an AVX2 CPU.
#   bash tests/ci_checks.sh <bin dir> <model dir>
set -u
BIN=$1; M=$2
export VELTRA_THREADS=4
fails=0
ok() { echo "ok   $1"; }
bad() { echo "FAIL $1"; fails=$((fails + 1)); }
tokens() { "$BIN/veltra-gen" --model "$1" --threads 4 --ids "$IDS" --n 48 "${@:2}" 2>/dev/null | sed -n 's/.*"tokens":\[\([0-9,]*\)\].*/\1/p'; }
IDS=$(python3 -c "print(','.join(str(3+(i*37)%600) for i in range(90)))")

for name in ci_dense ci_moe; do
  f=$M/$name.vlt
  # forward (one token at a time) vs forward_batch over a shared prefix: bit-identical logits for B = 1, 4, 8
  if "$BIN/veltra-selftest" --model "$f" --prompt-len 200 --steps 10 > /tmp/ci_$name.log 2>&1; then ok "$name selftest + shared-prefix batch"; else bad "$name selftest"; tail -5 /tmp/ci_$name.log; fi
  # decode-sized chunks must give the same logits as one token per forward
  "$BIN/veltra-gen" --model "$f" --threads 4 --ids "$IDS" --seq-logits 1 --dump-logits /tmp/ci_a.bin >/dev/null 2>&1
  "$BIN/veltra-gen" --model "$f" --threads 4 --ids "$IDS" --chunk-logits 8 --dump-logits /tmp/ci_b.bin >/dev/null 2>&1
  if [ -s /tmp/ci_a.bin ] && cmp -s /tmp/ci_a.bin /tmp/ci_b.bin; then ok "$name chunks of 8 == token by token"; else bad "$name chunked logits differ"; fi
  # greedy decoding with prompt-lookup speculation is lossless
  plain=$(tokens "$f" --spec 0); spec=$(tokens "$f" --spec 1 --lookup-n 8)
  if [ -n "$plain" ] && [ "$plain" = "$spec" ]; then ok "$name lookup speculation is token-exact"; else bad "$name speculation changed the output"; fi
  # the thread count must not change the result
  one=$("$BIN/veltra-gen" --model "$f" --threads 1 --ids "$IDS" --n 48 --spec 0 2>/dev/null | sed -n 's/.*"tokens":\[\([0-9,]*\)\].*/\1/p')
  if [ "$one" = "$plain" ]; then ok "$name 1 thread == 4 threads"; else bad "$name depends on the thread count"; fi
done

f=$M/ci_moe.vlt
base=$(tokens "$f" --spec 0)
if [ "$(tokens "$f" --spec 0 --moe-k 3)" = "$base" ]; then ok "moe-k = routed top-k changes nothing"; else bad "moe-k at the full top-k changed the output"; fi
k1=$(tokens "$f" --spec 0 --moe-k 1); k1s=$(tokens "$f" --spec 1 --lookup-n 8 --moe-k 1)
if [ -n "$k1" ] && [ "$k1" != "$base" ]; then ok "moe-k 1 really changes the routing"; else bad "moe-k had no effect"; fi
if [ "$k1" = "$k1s" ]; then ok "moe-k with speculation is token-exact"; else bad "moe-k + speculation differ"; fi

echo "ci_checks: $fails failures"
[ $fails -eq 0 ]
