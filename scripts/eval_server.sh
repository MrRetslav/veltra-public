#!/bin/bash
# One eval pass on a private server on 127.0.0.1:8097: eval_raptor (plain + schema) and sanity10. Only the server started here is stopped.
#   bash scripts/eval_server.sh <veltra-server> <out dir> <tag> <chat|completions> <server args...>
# Run from the repo root. Refuses to start when 8097 is taken.
set -u
BIN=$1; OUT=$2; TAG=$3; MODE=$4; shift 4
PORT=8097
mkdir -p "$OUT/data_$TAG"
if ss -Hltn "sport = :$PORT" | grep -q .; then echo "port $PORT is busy; not starting"; exit 2; fi
VELTRA_DATA_DIR="$OUT/data_$TAG" nice -n 10 "$BIN" "$@" --host 127.0.0.1 --port $PORT > "$OUT/server_$TAG.log" 2>&1 &
S=$!
trap 'kill $S 2>/dev/null; wait $S 2>/dev/null' EXIT
for i in $(seq 240); do
  curl -s -m 2 127.0.0.1:$PORT/health | grep -q ok && break
  kill -0 $S 2>/dev/null || { echo "server died"; tail -5 "$OUT/server_$TAG.log"; exit 1; }
  sleep 1
done
ID=$(curl -s -m 10 127.0.0.1:$PORT/v1/models | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
echo "== $TAG: serving $ID"
timeout 5400 nice -n 10 python3 scripts/eval_raptor.py --url http://127.0.0.1:$PORT --model "$ID" --mode "$MODE" --report "$OUT/raptor_$TAG.json" 2>&1 | tail -2
timeout 5400 nice -n 10 python3 scripts/eval_raptor.py --url http://127.0.0.1:$PORT --model "$ID" --mode "$MODE" --system-schema --report "$OUT/schema_$TAG.json" 2>&1 | tail -2
timeout 3600 nice -n 10 python3 scripts/run_sanity10.py --url http://127.0.0.1:$PORT --model "$ID" --report "$OUT/sanity_$TAG.json" 2>&1 | tail -1
echo "server peak RSS $(awk '/VmHWM/{printf "%.2f GB", $2/1048576}' /proc/$S/status 2>/dev/null) (mapped model pages included)"
python3 - "$OUT" "$TAG" <<'PY'
import json, sys
out, tag = sys.argv[1:3]
walls, think, outputs = [], 0, 0
for kind in ("raptor", "schema", "sanity"):
    rows = json.load(open("%s/%s_%s.json" % (out, kind, tag)))["results"]
    outputs += len(rows)
    think += sum(("<think>" in r["output"]) + ("</think>" in r["output"]) for r in rows)
    walls += [r["wall_s"] for r in rows if r.get("wall_s")]
    print("%s_%s: %d/%d failed=%s" % (kind, tag, sum(r["pass"] for r in rows), len(rows), ",".join(r["id"] for r in rows if not r["pass"]) or "-"))
print("think tags in %d outputs: %d | mean wall per raptor answer %.1f s" % (outputs, think, sum(walls) / max(1, len(walls))))
PY
