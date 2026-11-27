#!/bin/bash
# Starts ONE server on 127.0.0.1:8097, runs a client command against it, stops exactly that server and records its peak RSS.
#   bash benchmarks/night/with_server.sh <rss file> <server binary and args...> -- <client command...>
# The server gets --host 127.0.0.1 --port 8097 appended. Refuses to run when 8097 is taken. Never touches other processes.
set -u
RSS=$1; shift
SRV=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do SRV+=("$1"); shift; done
shift
PORT=8097
if ss -Hltn "sport = :$PORT" | grep -q .; then echo "port $PORT is busy; not starting"; exit 2; fi
# a server without SO_REUSEADDR (ik_llama.cpp) cannot bind while connections of the previous run are still in TIME-WAIT
for i in $(seq 120); do ss -Htan "sport = :$PORT" | grep -q . || break; sleep 1; done
SLOG=${RSS%.txt}.server.log
VELTRA_DATA_DIR=${RSS%.txt}.data nice -n 10 "${SRV[@]}" --host 127.0.0.1 --port $PORT > "$SLOG" 2>&1 &
S=$!
trap 'kill $S 2>/dev/null; wait $S 2>/dev/null' EXIT
up=0
for i in $(seq 600); do
  curl -s -m 2 127.0.0.1:$PORT/health | grep -q ok && { up=1; break; }
  kill -0 $S 2>/dev/null || break
  sleep 1
done
if [ $up != 1 ]; then echo "server did not become healthy"; tail -20 "$SLOG"; exit 1; fi
sleep 2
"$@"
rc=$?
awk '/VmHWM/{print $2}' /proc/$S/status > "$RSS" 2>/dev/null
echo "server peak RSS kB: $(cat "$RSS" 2>/dev/null)"
exit $rc
