#!/bin/bash
# Profile the sglang server processes that actually hold the client TCP
# connections (port 28553) during C (sglang-oai) vs B (IX) phases. We discover
# those PIDs via `ss -tnp` and run one py-spy --idle per PID (no --subprocesses,
# which segfaults on this tree). Folded stacks are merged per phase so we can see
# where the client-facing workers spend wall time and diff C vs B.
set -u
OUT=/data/jacchang_client_ab/srvprof
mkdir -p "$OUT"
D=/data/jacchang_client_ab/client_ab/mi355x_conc16_in1024_out1024_tp4_mtptrue_si1_srvprof
B_LOG="$D/B_ix_client.log"
PORT=28553
log() { echo "[profiler] $(date +%T) $*" >> "$OUT/profiler.log"; }

conn_pids() {  # PIDs holding ESTAB connections on :28553
  ss -tnHp "( sport = :$PORT )" 2>/dev/null \
    | grep -oE 'pid=[0-9]+' | grep -oE '[0-9]+' | sort -u
}

profile() {  # $1 = label
  local pids; pids=$(conn_pids)
  log "$1: client-facing pids = $(echo $pids | tr '\n' ' ')"
  local n=0
  for p in $pids; do
    py-spy record --idle --rate 150 --duration 40 --format raw \
      -o "$OUT/$1_$p.folded" --pid "$p" >> "$OUT/profiler.log" 2>&1 &
    n=$((n+1))
  done
  wait
  cat "$OUT/$1"_*.folded > "$OUT/$1.folded" 2>/dev/null
  log "$1: merged $n pid profiles -> $(wc -l < "$OUT/$1.folded" 2>/dev/null) stack lines, $(awk '{s+=$NF} END{print s}' "$OUT/$1.folded" 2>/dev/null) samples"
}

until [ "$(curl -s -o /dev/null -w '%{http_code}' http://0.0.0.0:$PORT/health)" = "200" ]; do sleep 3; done
log "server healthy"

# C phase: A (sglang-oai) starts right after health; give it time to ramp.
sleep 55
profile C

# B phase: trigger on IX main-run marker.
until grep -q "Starting main benchmark run" "$B_LOG" 2>/dev/null; do sleep 1; done
log "B main run started; sleeping 15s into B run"
sleep 15
profile B
log "all captures complete"
