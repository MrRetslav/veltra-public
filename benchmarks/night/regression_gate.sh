#!/bin/bash
# Default greedy mode must be bit-identical between a reference build and a new build.
#   regression_gate.sh REF_veltra-gen NEW_veltra-gen MODEL.vlt OUTDIR
REF=$1; NEW=$2; M=$3; P=$4; mkdir -p "$P"; FAIL=0
IDS=$(python3 -c "print(','.join(str(1000+(i*7919)%30000) for i in range(200)))")
LONG=$(python3 -c "print(','.join(str(1000+(i*6007)%30000) for i in range(1500)))")
for S in a b c d; do
  for V in ref new; do
    G=$REF; [ $V = new ] && G=$NEW
    case $S in
      a) ARGS="--ids $IDS --dump-logits $P/logits_$V.bin" ;;
      b) ARGS="--ids $IDS --n 96 --spec 0 --temp 0" ;;
      c) ARGS="--ids $IDS --n 96 --spec 1 --temp 0 --screen-head 1" ;;
      d) ARGS="--ids $LONG --n 48 --spec 1 --temp 0" ;;
    esac
    nice -n 10 $G --model $M --threads 8 $ARGS > $P/gen_${S}_$V.json 2> $P/gen_${S}_$V.err || { echo "case $S $V: EXIT $?"; FAIL=1; }
  done
  if [ $S = a ]; then
    if cmp -s $P/logits_ref.bin $P/logits_new.bin; then echo "case a logits IDENTICAL ($(stat -c %s $P/logits_new.bin) bytes)"; else echo "case a logits DIFFER"; FAIL=1; fi
  else
    python3 - $P/gen_${S}_ref.json $P/gen_${S}_new.json $S <<'PY' || FAIL=1
import json, sys
def tokens(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    d = json.loads(text[text.index("{"):])
    for k in ("tokens", "ids", "output_ids", "generated"):
        if isinstance(d.get(k), list):
            return d[k]
    raise SystemExit("no token list; keys: %s" % sorted(d))
a, b = tokens(sys.argv[1]), tokens(sys.argv[2])
same = len(a) > 0 and a == b
print("case %s tokens %s (%d vs %d)" % (sys.argv[3], "IDENTICAL" if same else "DIFFER", len(a), len(b)))
sys.exit(0 if same else 1)
PY
  fi
done
echo "--- --pareto-max startup (new build)"
nice -n 10 $NEW --model $M --threads 8 --ids $IDS --n 16 --temp 0 --pareto-max 2>&1 >/dev/null | grep -iE "pareto|screen"
[ $FAIL = 0 ] && echo "REGRESSION GATE: PASS" || echo "REGRESSION GATE: FAIL"
exit $FAIL
