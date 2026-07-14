#!/bin/bash
# B-only: profile the server processes holding client connections (:28553) during
# the IX client run. Polls ss until connections exist, then py-spy --idle per pid.
set -u
OUT=/data/jacchang_client_ab/srvprof
mkdir -p "$OUT"
PORT=28553
log() { echo "[profiler-B] $(date +%T) $*" >> "$OUT/profiler.log"; }

conn_pids() {
  ss -tnHp "( sport = :$PORT )" 2>/dev/null \
    | grep -oE 'pid=[0-9]+' | grep -oE '[0-9]+' | sort -u
}

until [ "$(curl -s -o /dev/null -w '%{http_code}' http://0.0.0.0:$PORT/health)" = "200" ]; do sleep 3; done
log "server healthy; polling for client connections"

# poll until the workload has live connections (past warmup/dispatch)
pids=""
for i in $(seq 1 120); do
  pids=$(conn_pids)
  [ -n "$pids" ] && break
  sleep 1
done
# give it a few more seconds to reach steady concurrency
sleep 8
pids=$(conn_pids)
log "B: client-facing pids = $(echo $pids | tr '\n' ' ')"
n=0
for p in $pids; do
  py-spy record --idle --rate 120 --duration 40 --format raw \
    -o "$OUT/B_$p.folded" --pid "$p" >> "$OUT/profiler.log" 2>&1 &
  n=$((n+1))
done
wait
cat "$OUT/B_"*.folded > "$OUT/B.folded" 2>/dev/null
log "B: merged $n pid profiles -> $(awk '{s+=$NF} END{print s}' "$OUT/B.folded" 2>/dev/null) samples"
log "B capture complete"
