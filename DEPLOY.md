# Deploy `literature-conflict-graph`

This project is designed to run as a Docker container and optionally be exposed through Cloudflare Tunnel.

Recommended target:

- app container on port `7860`
- public URL: `https://graph.paper-universe.uk`


## 1. Project path

On the server, the code should live at:

```bash
/workspace/literature-conflict-graph
```


## 2. Required software

The server should have:

- Docker
- Docker Compose
- optional: `cloudflared` for the public URL

Quick checks:

```bash
docker --version
docker compose version
```


## 3. Configure environment

From the project root:

```bash
cd /workspace/literature-conflict-graph
cp .env.example .env
```

Edit `.env` and set at least:

```env
OPENAI_API_KEY=...
AIGRAPH_BASE_URL=https://sub2api.kr.justafish.top/v1
AIGRAPH_MODEL=gpt-5.4
AIGRAPH_LLM_ENDPOINT=responses
AIGRAPH_REASONING_EFFORT=high
AIGRAPH_EXTRACT_CONCURRENCY=2
```

Notes:

- Keep `.env` on the server only.
- Do not commit API keys.


## 4. Build and run with Docker

From the project root:

```bash
cd /workspace/literature-conflict-graph
docker compose up -d --build
```

Check that the container is running:

```bash
docker ps
docker logs -f literature-conflict-graph
```

If the service name differs, run:

```bash
docker compose ps
```


## 5. Verify the app locally on the server

The app should answer on port `7860`.

```bash
curl http://127.0.0.1:7860
```

Expected result: HTML for the homepage.


## 6. Expose the public URL with Cloudflare Tunnel

Use this if you want:

```text
https://graph.paper-universe.uk
```

### Install `cloudflared`

Ubuntu example:

```bash
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
dpkg -i cloudflared-linux-amd64.deb
```

### Login

```bash
cloudflared tunnel login
```

Open the authorization link, log into Cloudflare, and approve the zone for:

```text
paper-universe.uk
```

### Create the tunnel

```bash
cloudflared tunnel create literature-conflict-graph
```

### Route DNS

```bash
cloudflared tunnel route dns literature-conflict-graph graph.paper-universe.uk
```

### Create the config file

Create:

```bash
/etc/cloudflared/config.yml
```

Template:

```yaml
tunnel: literature-conflict-graph
credentials-file: /root/.cloudflared/<TUNNEL_ID>.json

ingress:
  - hostname: graph.paper-universe.uk
    service: http://127.0.0.1:7860
  - service: http_status:404
```

Find the actual credentials file with:

```bash
ls /root/.cloudflared
```

Replace `<TUNNEL_ID>.json` with the real file name.

### Install as a service

```bash
cloudflared service install
systemctl enable cloudflared
systemctl restart cloudflared
systemctl status cloudflared
```


## 7. Final checks

Local:

```bash
curl http://127.0.0.1:7860
```

Public:

```bash
curl -I https://graph.paper-universe.uk
```

Open in a browser:

```text
https://graph.paper-universe.uk
```


## 8. Useful maintenance commands

Restart app:

```bash
cd /workspace/literature-conflict-graph
docker compose restart
```

Rebuild after code changes:

```bash
cd /workspace/literature-conflict-graph
docker compose up -d --build
```

Stop app:

```bash
cd /workspace/literature-conflict-graph
docker compose down
```

Watch tunnel logs:

```bash
journalctl -u cloudflared -f
```

Watch app logs:

```bash
docker compose logs -f
```


## 9. Troubleshooting

### `curl http://127.0.0.1:7860` fails

Check:

```bash
docker compose ps
docker compose logs -f
```

Common causes:

- `.env` is missing or incomplete
- Docker is not running
- the container failed during startup

### Public URL does not work

Check:

```bash
systemctl status cloudflared
journalctl -u cloudflared -f
```

Common causes:

- Cloudflare login was not completed
- wrong `credentials-file` path in `/etc/cloudflared/config.yml`
- DNS route was not created
- local app is not listening on `127.0.0.1:7860`


## 10. Minimal deploy flow

If you just want the shortest path:

```bash
cd /workspace/literature-conflict-graph
cp .env.example .env
# fill .env
docker compose up -d --build
curl http://127.0.0.1:7860
```

Then add Cloudflare Tunnel only after the local app is confirmed working.


## 11. Realign an existing deploy with upstream changes

When the upstream `main` branch has been updated (new schema, new anomaly
rules, new visualization, etc.) and you need the live `graph.paper-universe.uk`
to reflect them, run on the deploy host:

```bash
cd /workspace/literature-conflict-graph

# 1. Pull the new code.
git pull origin main

# 2. Rebuild the image and restart the container.
#    `--build` is what picks up the new code; `-d` keeps it detached.
docker compose up -d --build

# 3. Wait for the container to be healthy.
docker compose ps
docker compose logs --tail=40

# 4. Smoke test the local app.
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:7860/    # expect 200

# 5. Smoke test the public URL (Cloudflare Tunnel auto-reconnects).
curl -s -o /dev/null -w "%{http_code}\n" https://graph.paper-universe.uk/  # expect 200
```

If the new code adds Python modules / symbols, verify they import inside the
container:

```bash
docker compose exec literature-conflict-graph \
  python -c "from aigraph.anomalies import _detect_replication_conflict; \
             from aigraph.graph import _resolve_entity_value, _add_bibliographic_coupling_edges; \
             print('aigraph upgrade OK')"
```

### Realign existing runs after a schema change

When upstream changes node types, edge types, or anomaly rules (i.e.
`build-graph` / `detect-anomalies` / `visualize` would now produce a
different shape on the same `claims.jsonl`), every existing run dir under
`outputs/runs/` is now displaying a **stale graph** + **stale anomalies**.
Re-rendering the HTML alone is not enough — the underlying `graph.json`
and `anomalies.jsonl` still carry the pre-upgrade schema, and the new
JS palette will not have colors for the leftover node types, so the
visualization shows untyped grey nodes.

Run the full rebuild over every run on the deploy host. This is **purely
deterministic and costs $0** (no LLM calls — only the graph builder,
anomaly detector, and HTML renderer). The repo ships a wrapper script
that handles pull, rebuild, container-code verification, the loop, and
a public-URL smoke test:

```bash
cd /workspace/literature-conflict-graph
bash automation/bin/realign_all_runs.sh
```

Optional toggles via env (each step is independently skippable so you
can re-run a partial flow without burning time):

```bash
SKIP_PULL=1   bash automation/bin/realign_all_runs.sh   # source already current
SKIP_BUILD=1  bash automation/bin/realign_all_runs.sh   # image already current
SKIP_REALIGN=1 bash automation/bin/realign_all_runs.sh  # only verify, do not loop
```

If you prefer the equivalent inline loop (no script):

```bash
cd /workspace/literature-conflict-graph
git pull origin main
docker compose up -d --build
for d in outputs/runs/*/; do
  bn=$(basename "$d")
  [ -f "$d/claims.jsonl" ] && [ -s "$d/claims.jsonl" ] && [ -f "$d/papers.jsonl" ] || continue
  echo "=== rebuilding $bn ==="
  docker compose exec -T aigraph python -m aigraph.cli build-graph \
    --claims "/app/$d/claims.jsonl" --papers "/app/$d/papers.jsonl" \
    --output "/app/$d/graph.json"
  docker compose exec -T aigraph python -m aigraph.cli detect-anomalies \
    --graph "/app/$d/graph.json" --claims "/app/$d/claims.jsonl" \
    --output "/app/$d/anomalies.jsonl"
  docker compose exec -T aigraph python -m aigraph.cli visualize \
    --input-dir "/app/$d" --output "/app/$d/index.html"
done
```

Each run takes ~30 s for graph + ~10 min for anomalies + ~5 s for
visualize, so a fleet of ~15 runs is roughly 2-3 hours of cheap CPU.
Empty runs (no claims) are skipped.

After this completes, every run on the deploy host is on the current
schema, and `https://app.paper-universe.uk/runs/<any>/index.html`
matches what a developer sees locally.

### Update the served data

The container reads pipeline outputs from a host-mounted directory (typically
`outputs/runs/`). To serve a fresh full-corpus run produced on a separate
ingest box, copy the run dir into the deploy host first:

```bash
# from the ingest box (replace <deploy-host>):
rsync -av --partial /tmp/fullrun_v2/ <deploy-host>:/workspace/literature-conflict-graph/outputs/runs/<YYYYMMDD-HHMMSS-XXXXXX>/

# the run dir name MUST match YYYYMMDD-HHMMSS-XXXXXX (6 hex) — the server
# regex rejects others. Edit status.json so `status: "done"` and the run
# shows up in the home page.
```

The container picks up new run dirs without restart (the home page lists them
on every request).

### Common issues

- **`docker compose up --build` fails with hatch error** — the
  `pyproject.toml` may pin a git URL extra; ensure `[tool.hatch.metadata]
  allow-direct-references = true` is set (already in the repo, but worth
  double-checking after merges).
- **HTTP 530 on `graph.paper-universe.uk`** — Cloudflare can't reach the
  origin; the tunnel daemon (`cloudflared`) on the deploy host is down or
  the local app stopped. Check `systemctl status cloudflared` and
  `docker compose ps`.
- **New run shows "404 Not found" on `/search/<run_id>`** — the dir name
  doesn't match `RUN_ID_RE` regex (`^[0-9]{8}-[0-9]{6}-[a-f0-9]{6}$`).
  Rename the dir + update `status.json`'s `run_id` field accordingly.


## 12. Run 24/7 corpus ingest on the server

If you want the server to keep building the offline arXiv reasoning corpus
around the clock, run the corpus daemon on the host with the project virtualenv.

### Prepare the host Python environment

```bash
cd /workspace/literature-conflict-graph
python3 -m venv .venv
.venv/bin/pip install -e '.[real]'
```

### Configure the corpus loop

Add these to `.env` if you want to tune the ingest loop:

```env
AIGRAPH_CORPUS_ROOT=data/corpus/arxiv_reasoning
AIGRAPH_CORPUS_FROM_YEAR=2022
AIGRAPH_CORPUS_TO_YEAR=2026
AIGRAPH_CORPUS_SEED_LIMIT=50
AIGRAPH_CORPUS_BATCH_SIZE=5
AIGRAPH_CORPUS_SLEEP_SECONDS=300
AIGRAPH_CORPUS_SEED_EVERY=24
AIGRAPH_CORPUS_VALIDATE_EVERY=12
```

Meaning:

- `SEED_LIMIT`: how many metadata candidates to fetch per reasoning query
- `BATCH_SIZE`: how many unfinished papers to sync each loop
- `SLEEP_SECONDS`: how long to wait between loops
- `SEED_EVERY`: reseed every N loops
- `VALIDATE_EVERY`: write a fresh `summary.json` every N loops

### Smoke test one loop

```bash
cd /workspace/literature-conflict-graph
AIGRAPH_CORPUS_ONCE=1 ./automation/bin/corpus_daemon.sh
```

### Install as a systemd service

```bash
cd /workspace/literature-conflict-graph
chmod +x automation/bin/corpus_daemon.sh automation/bin/install_corpus_service.sh
./automation/bin/install_corpus_service.sh /workspace/literature-conflict-graph
```

### Watch the daemon

```bash
journalctl -u aigraph-corpus -f
```

### Check corpus outputs

```bash
ls /workspace/literature-conflict-graph/data/corpus/arxiv_reasoning
cat /workspace/literature-conflict-graph/data/corpus/arxiv_reasoning/summary.json
```
