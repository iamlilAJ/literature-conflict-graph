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


## 11. Run 24/7 corpus ingest on the server

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
