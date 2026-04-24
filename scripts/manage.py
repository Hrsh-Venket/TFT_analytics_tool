#!/usr/bin/env python3
"""TFT Analytics — interactive management tool.

Run inside the app container:

    docker compose exec -it app python scripts/manage.py

Walks through the common maintenance tasks (refresh mappings, collect
matches, run clustering, inspect DB) with prompts so you don't have to
remember flag names.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(REPO))

TIERS = ["CHALLENGER", "GRANDMASTER", "MASTER", "DIAMOND",
         "EMERALD", "PLATINUM", "GOLD", "SILVER", "BRONZE"]


def ask(prompt, default=None, cast=str, choices=None):
    hint_parts = []
    if choices:
        hint_parts.append("/".join(choices))
    if default is not None:
        hint_parts.append(f"default: {default}")
    hint = f" [{', '.join(hint_parts)}]" if hint_parts else ""
    while True:
        raw = input(f"{prompt}{hint}: ").strip()
        if not raw and default is not None:
            raw = str(default)
        if not raw:
            print("  (required)")
            continue
        if choices and raw.upper() not in [c.upper() for c in choices]:
            print(f"  choose one of: {', '.join(choices)}")
            continue
        try:
            return cast(raw)
        except ValueError as e:
            print(f"  invalid: {e}")


def get_api_key():
    key = os.environ.get("RIOT_API_KEY", "").strip()
    if key and not key.endswith("xxxxxxxx"):
        return key
    return ask("Riot API key (RGAPI-…)")


def run(args, cwd=None):
    print(f"\n>>> {' '.join(args)}\n")
    return subprocess.call(args, cwd=cwd)


def action_refresh_mappings():
    """Pull a sample of recent matches and rebuild name mappings."""
    print("\n-- Refresh name mappings --")
    days = ask("Days since this set began", default=14, cast=int)
    tier = ask("Minimum tier to sample from", default="MASTER", choices=TIERS).upper()
    count = ask("Sample size (matches)", default=100, cast=int)
    key = get_api_key()

    rc = run(["python", str(SCRIPTS / "collect_subset.py"),
              "--api-key", key,
              "--num-matches", str(count),
              "--tier", tier,
              "--days", str(days)])
    if rc != 0:
        print("collect_subset failed; not regenerating mappings.")
        return
    run(["python", "-m", "tft_analytics.mapper", "--init-from-subset"])


def action_collect():
    """Collect match data into Postgres."""
    print("\n-- Collect match data --")
    days = ask("Days of history to pull", default=7, cast=int)
    tier = ask("Minimum tier", default="CHALLENGER", choices=TIERS).upper()
    per_player = ask("Max matches per player (number or 'all')", default="all")
    key = get_api_key()

    run(["python", str(SCRIPTS / "collect.py"),
         "--api-key", key,
         "--days", str(days),
         "--tier", tier,
         "--max-matches-per-player", str(per_player)])


def action_cluster():
    """Regenerate main/sub cluster tables from current match data."""
    print("\n-- Run clustering --")
    run(["python", "-m", "tft_analytics.clustering"])


def action_update_mappings():
    """Apply current name mappings to existing DB rows."""
    print("\n-- Update DB using current mappings --")
    print("Re-derives unit names from raw character_id and applies")
    print("trait/item renames diffed against the previous snapshot.")
    dry = ask("Dry run?", default="n", choices=["y", "n"]).lower() == "y"
    args = ["python", str(SCRIPTS / "update_mappings.py")]
    if dry:
        args.append("--dry-run")
    run(args)


def action_stats():
    """Print DB stats without hitting the Flask layer."""
    from psycopg2.extras import RealDictCursor

    from tft_analytics.postgres import get_connection, put_connection

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    COUNT(DISTINCT match_id) AS matches,
                    COUNT(*)                 AS participants,
                    MIN(game_datetime)       AS earliest,
                    MAX(game_datetime)       AS latest
                FROM match_participants
            """)
            row = cur.fetchone() or {}
        print("\n-- Match data --")
        for k, v in row.items():
            print(f"  {k:12s} {v}")

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT COUNT(*) AS n FROM main_clusters")
            n_main = cur.fetchone()["n"]
            cur.execute("SELECT COUNT(*) AS n FROM sub_clusters")
            n_sub = cur.fetchone()["n"]
        print(f"\n-- Clusters --")
        print(f"  main:        {n_main}")
        print(f"  sub:         {n_sub}")
    except Exception as e:
        print(f"  error: {e}")
    finally:
        put_connection(conn)


def action_top_clusters():
    """Show the top main clusters by avg placement."""
    from psycopg2.extras import RealDictCursor

    from tft_analytics.postgres import get_connection, put_connection

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, size, avg_placement, winrate, top4_rate,
                       common_carries, top_units_display
                FROM main_clusters
                ORDER BY avg_placement ASC
                LIMIT 10
            """)
            rows = cur.fetchall()
    finally:
        put_connection(conn)

    if not rows:
        print("\nNo clusters yet. Run clustering first.")
        return
    print("\n-- Top 10 main clusters (by avg placement) --")
    for r in rows:
        print(f"  #{r['id']:>3}  n={r['size']:<5} avg={r['avg_placement']:.2f}  "
              f"win={r['winrate']:.1f}%  top4={r['top4_rate']:.1f}%  "
              f"carries={r['common_carries']}")


MENU = [
    ("Refresh name mappings (new TFT set)", action_refresh_mappings),
    ("Collect match data",                 action_collect),
    ("Run clustering analysis",            action_cluster),
    ("Update DB using current mappings",    action_update_mappings),
    ("Show DB stats",                      action_stats),
    ("Show top clusters",                  action_top_clusters),
]


def main():
    while True:
        print("\nTFT Analytics — manage")
        for i, (label, _) in enumerate(MENU, 1):
            print(f"  {i}) {label}")
        print("  q) Quit")
        try:
            choice = input("> ").strip().lower()
        except EOFError:
            return
        if choice in ("q", "quit", "exit", ""):
            return
        try:
            _, fn = MENU[int(choice) - 1]
        except (ValueError, IndexError):
            print("  invalid choice")
            continue
        try:
            fn()
        except KeyboardInterrupt:
            print("\n  cancelled")


if __name__ == "__main__":
    main()
