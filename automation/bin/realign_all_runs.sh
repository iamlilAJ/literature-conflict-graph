#!/bin/bash
# Realign every run under outputs/runs/ to the current code on the deploy host.
#
# Run from the docker deploy host (the one serving graph.paper-universe.uk):
#
#   cd /workspace/literature-conflict-graph
#   bash automation/bin/realign_all_runs.sh
#
# What it does (zero LLM cost):
#   1. git pull origin main
#   2. docker compose up -d --build  (rebuild image with new code)
#   3. verify the container imports the latest module symbols
#   4. for every run dir with a non-empty claims.jsonl + papers.jsonl,
#      rebuild graph.json, anomalies.jsonl, and index.html in-place
#   5. smoke check the public URL
#
# Empty runs (failed claim extraction) are skipped. The container does not need
# to be restarted between runs — outputs/runs/ is host-mounted.
#
# Optional flags via env:
#   SKIP_PULL=1     skip git pull   (use the source already on disk)
#   SKIP_BUILD=1    skip docker compose --build  (image already current)
#   SKIP_REALIGN=1  skip the rebuild loop (only verify code)
#   PUBLIC_URL=...  smoke-check URL (defaults to https://app.paper-universe.uk)

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PUBLIC_URL="${PUBLIC_URL:-https://app.paper-universe.uk}"
SKIP_PULL="${SKIP_PULL:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_REALIGN="${SKIP_REALIGN:-0}"

cd "$REPO_ROOT"

step() {
  echo
  echo "===== $(date '+%H:%M:%S') $* ====="
}

step "0. context"
echo "  repo:       $REPO_ROOT"
echo "  HEAD:       $(git rev-parse --short HEAD 2>/dev/null) $(git log -1 --format='%s' 2>/dev/null)"
echo "  public URL: $PUBLIC_URL"

if [ "$SKIP_PULL" = "1" ]; then
  step "1. git pull SKIPPED"
else
  step "1. git pull origin main"
  git pull origin main || { echo "  ⚠ git pull failed, aborting"; exit 1; }
  echo "  new HEAD: $(git rev-parse --short HEAD) $(git log -1 --format='%s')"
fi

if [ "$SKIP_BUILD" = "1" ]; then
  step "2. docker compose up --build SKIPPED"
else
  step "2. docker compose up -d --build"
  docker compose up -d --build || { echo "  ⚠ build failed, aborting"; exit 1; }
  docker compose ps
fi

step "3. verify container imports latest symbols"
if docker compose exec -T aigraph python -c \
    "from aigraph.anomalies import _detect_replication_conflict; \
     from aigraph.graph import _resolve_entity_value, _add_bibliographic_coupling_edges; \
     print('aigraph code OK')" 2>&1; then
  echo "  ✓ container code current"
else
  echo "  ❌ import check failed — container is on stale code; aborting"
  exit 1
fi

if [ "$SKIP_REALIGN" = "1" ]; then
  step "4. realign existing runs SKIPPED"
else
  step "4. realign existing runs (build-graph + detect-anomalies + visualize)"
  rebuilt=0
  skipped=0
  failed=0
  for d in outputs/runs/*/; do
    bn=$(basename "$d")
    if [ ! -f "$d/claims.jsonl" ] || [ ! -s "$d/claims.jsonl" ] || [ ! -f "$d/papers.jsonl" ]; then
      echo "  skip   $bn  (no claims or papers)"
      skipped=$((skipped+1))
      continue
    fi
    t0=$(date +%s)
    if docker compose exec -T aigraph python -m aigraph.cli build-graph \
         --claims "/app/$d/claims.jsonl" --papers "/app/$d/papers.jsonl" \
         --output "/app/$d/graph.json" >/dev/null 2>&1 \
       && docker compose exec -T aigraph python -m aigraph.cli detect-anomalies \
         --graph "/app/$d/graph.json" --claims "/app/$d/claims.jsonl" \
         --output "/app/$d/anomalies.jsonl" >/dev/null 2>&1 \
       && docker compose exec -T aigraph python -m aigraph.cli visualize \
         --input-dir "/app/$d" --output "/app/$d/index.html" >/dev/null 2>&1; then
      dt=$(( $(date +%s) - t0 ))
      claims=$(wc -l < "$d/claims.jsonl" | tr -d ' ')
      anoms=$(wc -l < "$d/anomalies.jsonl" 2>/dev/null | tr -d ' ' || echo 0)
      echo "  ✓ $bn  ($claims claims → $anoms anomalies, ${dt}s)"
      rebuilt=$((rebuilt+1))
    else
      echo "  ❌ $bn  (one of build-graph/detect-anomalies/visualize failed)"
      failed=$((failed+1))
    fi
  done
  echo
  echo "  summary: rebuilt=$rebuilt  skipped=$skipped  failed=$failed"
fi

step "5. smoke check public URL"
code=$(curl -sS -o /dev/null -m 15 -w '%{http_code}' "$PUBLIC_URL/" 2>&1)
echo "  $PUBLIC_URL → HTTP $code"
if [ "$code" != "200" ]; then
  echo "  ⚠ public URL not 200 — check cloudflared / docker compose ps"
fi

step "DONE"
