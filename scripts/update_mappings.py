#!/usr/bin/env python3
"""Apply the current name mappings to existing DB rows.

Unit names are re-derived from each unit's raw ``character_id`` using the
current mapping, so changes to ``units.csv`` always propagate.

Traits and items are stored only as cleaned strings (the raw API names
aren't kept in the DB), so the script diffs ``latest/*.csv`` against the
snapshot in ``latest/.previous/`` and applies the resulting
``old_clean -> new_clean`` transitions. The first run only seeds the
snapshot — edit your mappings and run again.

Usage (inside the app container):

    python scripts/update_mappings.py            # apply + refresh snapshot
    python scripts/update_mappings.py --dry-run  # show counts, no writes
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from psycopg2.extras import Json  # noqa: E402

from tft_analytics.mapper import LATEST_MAPPINGS_DIR  # noqa: E402
from tft_analytics.postgres import get_connection, put_connection  # noqa: E402

PREVIOUS_DIR = LATEST_MAPPINGS_DIR / ".previous"
MAPPING_FILES = ("units.csv", "traits.csv", "items.csv")


def _load_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            old = (row.get("old_name") or "").strip()
            new = (row.get("new_name") or "").strip()
            if old and new:
                out[old] = new
    return out


def _transitions(old: dict[str, str], new: dict[str, str]) -> dict[str, str]:
    """Build ``old_clean -> new_clean`` from raws present in both sides."""
    trans: dict[str, str] = {}
    for raw, old_clean in old.items():
        new_clean = new.get(raw)
        if new_clean and new_clean != old_clean:
            prior = trans.get(old_clean)
            if prior and prior != new_clean:
                print(f"  warning: {old_clean!r} has ambiguous new name "
                      f"({prior!r} vs {new_clean!r}); keeping {prior!r}")
                continue
            trans[old_clean] = new_clean
    return trans


def _snapshot():
    PREVIOUS_DIR.mkdir(parents=True, exist_ok=True)
    for fname in MAPPING_FILES:
        src = LATEST_MAPPINGS_DIR / fname
        if src.exists():
            shutil.copy2(src, PREVIOUS_DIR / fname)


def _remap_units(units, current_units_map, item_trans):
    changed = False
    out = []
    for u in units or []:
        if not isinstance(u, dict):
            out.append(u)
            continue
        new_u = dict(u)
        char_id = new_u.get("character_id") or ""
        mapped_name = current_units_map.get(char_id, new_u.get("name", ""))
        if mapped_name != new_u.get("name"):
            new_u["name"] = mapped_name
            changed = True
        items = new_u.get("item_names") or []
        if item_trans and items:
            new_items = [item_trans.get(i, i) for i in items]
            if new_items != items:
                new_u["item_names"] = new_items
                changed = True
        out.append(new_u)
    return out, changed


def _remap_traits(traits, trait_trans):
    if not trait_trans:
        return traits, False
    changed = False
    out = []
    for t in traits or []:
        if not isinstance(t, dict):
            out.append(t)
            continue
        name = t.get("name")
        new_name = trait_trans.get(name, name)
        if new_name != name:
            nt = dict(t)
            nt["name"] = new_name
            out.append(nt)
            changed = True
        else:
            out.append(t)
    return out, changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute changes but don't write to DB or refresh snapshot.")
    args = parser.parse_args()

    current_units = _load_csv(LATEST_MAPPINGS_DIR / "units.csv")
    current_traits = _load_csv(LATEST_MAPPINGS_DIR / "traits.csv")
    current_items = _load_csv(LATEST_MAPPINGS_DIR / "items.csv")

    if not PREVIOUS_DIR.exists():
        if args.dry_run:
            print("No snapshot at", PREVIOUS_DIR)
            print("First run will seed it. Nothing to diff yet.")
            return 0
        _snapshot()
        print(f"Seeded snapshot at {PREVIOUS_DIR}.")
        print("Edit your mappings (units/traits/items.csv) and run again to apply.")
        return 0

    prev_units = _load_csv(PREVIOUS_DIR / "units.csv")
    prev_traits = _load_csv(PREVIOUS_DIR / "traits.csv")
    prev_items = _load_csv(PREVIOUS_DIR / "items.csv")

    trait_trans = _transitions(prev_traits, current_traits)
    item_trans = _transitions(prev_items, current_items)

    print("Transitions vs previous snapshot:")
    print(f"  traits: {len(trait_trans)}  items: {len(item_trans)}")
    print("Unit names will be re-derived from character_id using current units.csv.")
    if trait_trans:
        sample = list(trait_trans.items())[:5]
        print("  sample trait renames:", ", ".join(f"{a}->{b}" for a, b in sample))
    if item_trans:
        sample = list(item_trans.items())[:5]
        print("  sample item renames:", ", ".join(f"{a}->{b}" for a, b in sample))

    conn = get_connection()
    scanned = 0
    updated = 0
    try:
        conn.autocommit = False
        with conn.cursor(name="mp_update_mappings") as read_cur:
            read_cur.itersize = 500
            read_cur.execute("""
                SELECT match_id, puuid, units, traits
                FROM match_participants
            """)
            write_cur = conn.cursor()
            try:
                for match_id, puuid, units, traits in read_cur:
                    scanned += 1
                    units_val = units if isinstance(units, list) else (
                        json.loads(units) if units else []
                    )
                    traits_val = traits if isinstance(traits, list) else (
                        json.loads(traits) if traits else []
                    )
                    new_units, uchanged = _remap_units(units_val, current_units, item_trans)
                    new_traits, tchanged = _remap_traits(traits_val, trait_trans)
                    if not (uchanged or tchanged):
                        continue
                    updated += 1
                    if args.dry_run:
                        continue
                    write_cur.execute("""
                        UPDATE match_participants
                        SET units = %s, traits = %s
                        WHERE match_id = %s AND puuid = %s
                    """, (Json(new_units), Json(new_traits), match_id, puuid))
            finally:
                write_cur.close()
        if args.dry_run:
            conn.rollback()
        else:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        put_connection(conn)

    print(f"Scanned {scanned} participants; "
          f"{'would update' if args.dry_run else 'updated'} {updated}.")

    if not args.dry_run and updated:
        _snapshot()
        print(f"Snapshot refreshed at {PREVIOUS_DIR}.")
    elif not args.dry_run and not updated:
        # Still refresh so mapping edits that were no-ops on data get picked up next time.
        _snapshot()
        print(f"No rows changed. Snapshot refreshed at {PREVIOUS_DIR}.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
