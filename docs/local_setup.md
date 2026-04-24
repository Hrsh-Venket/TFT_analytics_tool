# TFT Analytics — local setup & maintenance

Public URL: `https://tft-app.hrsh.uk` → Cloudflare tunnel → `localhost:8000` (app) → Postgres in `db` container.

## One-time setup

```bash
cd ~/TFT_analytics_tool

# .env must have POSTGRES_PASSWORD and a real RIOT_API_KEY
docker compose --env-file .env up -d --build
```

DB tables are created automatically on app startup (`ensure_tables()`).

### Cloudflare tunnel (already done, recorded here for reference)

Tunnel UUID: `7a747616-689b-433c-bb0e-3d908530068b`, runs as the `cloudflared` systemd service (remote-managed token).

DNS + ingress were set up via CLI + API using the cert at `~/.cloudflared/cert.pem`:

```bash
cloudflared tunnel login                     # one-time, writes cert.pem
cloudflared tunnel route dns <UUID> tft-app.hrsh.uk
# Ingress pushed to remote-managed tunnel via PUT /accounts/{acct}/cfd_tunnel/{uuid}/configurations
# (ingress: tft-app.hrsh.uk → http://localhost:8000)
```

To change ingress, re-run the `PUT` (see `docs/tunnel_ingress.md` below) or edit in the Zero Trust dashboard.

## Day-to-day: the interactive manager

One command handles all maintenance tasks with guided prompts:

```bash
docker compose exec -it app python scripts/manage.py
```

Menu:
1. **Refresh name mappings** — asks days since the set began, tier, sample size, runs `collect_subset.py` then `mapper.py --init-from-subset`.
2. **Collect match data** — asks days, tier, max matches per player, runs `collect.py`.
3. **Run clustering** — rebuilds `main_clusters` / `sub_clusters`.
4. **Show DB stats** — counts + time range.
5. **Show top clusters** — top 10 main clusters by avg placement.

Set `RIOT_API_KEY` in `.env` once and the tool picks it up automatically.

### Raw commands (if you prefer)

```bash
# Refresh mappings when a new set launches
docker compose exec app python scripts/collect_subset.py --api-key "$RIOT_API_KEY" --num-matches 100 --tier MASTER --days 7
docker compose exec app python -m tft_analytics.mapper --init-from-subset

# Collect
docker compose exec app python scripts/collect.py --api-key "$RIOT_API_KEY" --days 7 --tier MASTER

# Cluster
docker compose exec app python -m tft_analytics.clustering
```

Collection is idempotent (`ON CONFLICT DO NOTHING` on `(match_id, puuid)`; tracker file `global_matches_downloaded.json` skips already-downloaded matches). Clustering truncates and repopulates its tables.

## Day-to-day ops

```bash
# Service state
docker compose ps
docker compose logs -f app
docker compose logs -f db

# Restart app after code change
docker compose up -d --build app

# Psql into the db
docker compose exec db psql -U tft -d tft_analytics

# Stop / start everything
docker compose down
docker compose --env-file .env up -d

# Tunnel
systemctl status cloudflared
sudo systemctl restart cloudflared
```

## Health checks

```bash
curl -s http://localhost:8000/api/stats           # from host
curl -s https://tft-app.hrsh.uk/api/stats         # via tunnel
```

Both should return JSON with the same counts.

## Gotchas

- Host already runs Postgres 17 on `5432`; the db container does **not** publish `5432` to the host. Use `docker compose exec db psql ...` to connect.
- `collect.py` and `mapper.py` need a valid `RIOT_API_KEY` in the container env — set it in `.env` before `docker compose up`.
- Riot API keys expire every 24h unless you have a production key.
- `clustering.py` requires data in `match_participants`; run `collect.py` first or it'll produce no clusters.
