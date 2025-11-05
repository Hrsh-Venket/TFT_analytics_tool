#!/usr/bin/env python3
"""
Extract Sample Matches for ML Model Development

This script extracts a few sample matches from BigQuery in a human-readable format
to help decide on feature encoding for the TFT board strength prediction model.

Usage:
    python extract_sample_matches.py [--num-matches 2] [--output sample_matches.json]
"""

import argparse
import json
from datetime import datetime
from google.cloud import bigquery
from typing import List, Dict, Any


def extract_sample_matches(project_id: str = None, dataset_id: str = 'tft_analytics',
                          num_matches: int = 2) -> List[Dict[str, Any]]:
    """
    Extract sample matches with all participant data from BigQuery.

    Args:
        project_id: GCP project ID (auto-detected if None)
        dataset_id: BigQuery dataset ID
        num_matches: Number of matches to extract

    Returns:
        List of match dictionaries with all participant data
    """
    client = bigquery.Client(project=project_id)
    project_id = project_id or client.project

    # Query to get complete match data for N random matches
    query = f"""
    WITH random_matches AS (
        SELECT DISTINCT match_id
        FROM `{project_id}.{dataset_id}.match_participants`
        ORDER BY RAND()
        LIMIT {num_matches}
    )
    SELECT
        p.*
    FROM `{project_id}.{dataset_id}.match_participants` p
    INNER JOIN random_matches rm ON p.match_id = rm.match_id
    ORDER BY p.match_id, p.placement
    """

    print(f"Extracting {num_matches} sample matches from BigQuery...")
    print(f"Project: {project_id}, Dataset: {dataset_id}")

    # Execute query
    query_job = client.query(query)
    results = query_job.result()

    # Group results by match_id
    matches = {}
    for row in results:
        match_id = row.match_id

        if match_id not in matches:
            matches[match_id] = {
                'match_id': match_id,
                'game_datetime': row.game_datetime.isoformat() if row.game_datetime else None,
                'game_length': row.game_length,
                'game_version': row.game_version,
                'tft_set_number': row.tft_set_number,
                'tft_set_core_name': row.tft_set_core_name,
                'participants': []
            }

        # Convert units from STRUCT to dict
        units = []
        if row.units:
            for unit in row.units:
                units.append({
                    'character_id': unit['character_id'],
                    'name': unit['name'],
                    'tier': unit['tier'],  # Star level (1, 2, or 3)
                    'rarity': unit['rarity'],  # Cost (1-5)
                    'items': list(unit['item_names']) if unit.get('item_names') else []
                })

        # Convert traits from STRUCT to dict
        traits = []
        if row.traits:
            for trait in row.traits:
                traits.append({
                    'name': trait['name'],
                    'num_units': trait['num_units'],
                    'style': trait['style'],  # 0=inactive, 1-4=bronze/silver/gold/chromatic
                    'tier_current': trait['tier_current'],
                    'tier_total': trait['tier_total']
                })

        # Add participant data
        participant = {
            'placement': row.placement,
            'puuid': row.puuid,
            'summoner_name': f"{row.riot_id_game_name}#{row.riot_id_tagline}" if row.riot_id_game_name else None,
            'level': row.level,
            'last_round': row.last_round,
            'gold_left': row.gold_left,
            'players_eliminated': row.players_eliminated,
            'total_damage_to_players': row.total_damage_to_players,
            'units': units,
            'traits': traits,
            'num_units': len(units),
            'num_active_traits': sum(1 for t in traits if t['style'] > 0)
        }

        matches[match_id]['participants'].append(participant)

    print(f"✓ Extracted {len(matches)} matches with {sum(len(m['participants']) for m in matches.values())} participants")

    return list(matches.values())


def print_match_summary(matches: List[Dict[str, Any]]) -> None:
    """
    Print a human-readable summary of the extracted matches.

    Args:
        matches: List of match dictionaries
    """
    print("\n" + "="*80)
    print("SAMPLE MATCHES - HUMAN READABLE FORMAT")
    print("="*80)

    for match_idx, match in enumerate(matches, 1):
        print(f"\n{'─'*80}")
        print(f"MATCH {match_idx}: {match['match_id']}")
        print(f"{'─'*80}")
        print(f"Set: {match['tft_set_core_name']} (Set {match['tft_set_number']})")
        print(f"Date: {match['game_datetime']}")
        print(f"Duration: {match['game_length']:.1f}s ({match['game_length']/60:.1f} min)")
        print(f"Version: {match['game_version']}")

        print(f"\n{'Participants:':-^80}")
        for participant in match['participants']:
            print(f"\nPlacement #{participant['placement']} - {participant['summoner_name']}")
            print(f"  Level: {participant['level']} | Last Round: {participant['last_round']} | Gold Left: {participant['gold_left']}")
            print(f"  Damage Dealt: {participant['total_damage_to_players']} | Players Eliminated: {participant['players_eliminated']}")

            print(f"\n  Units ({participant['num_units']}):")
            for unit in sorted(participant['units'], key=lambda u: u['rarity'], reverse=True):
                star_str = "★" * unit['tier']
                items_str = ", ".join(unit['items']) if unit['items'] else "No items"
                print(f"    {unit['name']:20s} {star_str:4s} (Cost: {unit['rarity']}) - Items: {items_str}")

            print(f"\n  Active Traits ({participant['num_active_traits']}):")
            active_traits = [t for t in participant['traits'] if t['style'] > 0]
            for trait in sorted(active_traits, key=lambda t: t['tier_current'], reverse=True):
                style_name = ['Inactive', 'Bronze', 'Silver', 'Gold', 'Chromatic'][trait['style']]
                print(f"    {trait['name']:20s} {trait['num_units']} units - Tier {trait['tier_current']}/{trait['tier_total']} ({style_name})")

        print()


def save_matches_json(matches: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save matches to a JSON file.

    Args:
        matches: List of match dictionaries
        output_file: Output file path
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(matches, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved {len(matches)} matches to {output_file}")


def print_encoding_suggestions(matches: List[Dict[str, Any]]) -> None:
    """
    Print suggestions for feature encoding based on the sample data.

    Args:
        matches: List of match dictionaries
    """
    print("\n" + "="*80)
    print("FEATURE ENCODING SUGGESTIONS")
    print("="*80)

    # Collect statistics
    all_units = set()
    all_traits = set()
    all_items = set()
    unit_costs = {}
    max_units = 0
    max_traits = 0

    for match in matches:
        for participant in match['participants']:
            max_units = max(max_units, len(participant['units']))
            max_traits = max(max_traits, len([t for t in participant['traits'] if t['style'] > 0]))

            for unit in participant['units']:
                all_units.add(unit['name'])
                unit_costs[unit['name']] = unit['rarity']
                for item in unit['items']:
                    all_items.add(item)

            for trait in participant['traits']:
                if trait['style'] > 0:
                    all_traits.add(trait['name'])

    print(f"\nDataset Statistics:")
    print(f"  Unique units seen: {len(all_units)}")
    print(f"  Unique traits seen: {len(all_traits)}")
    print(f"  Unique items seen: {len(all_items)}")
    print(f"  Max units per board: {max_units}")
    print(f"  Max active traits: {max_traits}")

    print(f"\n{'Encoding Approaches:':-^80}")

    print("\n1. UNIT ENCODING:")
    print("   Option A: One-hot encoding per unit")
    print(f"     - Dimension: {len(all_units)} features (binary presence)")
    print("     - Pros: Simple, captures unit presence")
    print("     - Cons: Loses star level and item information")

    print("\n   Option B: Multi-dimensional unit encoding")
    print(f"     - Per unit: [presence, star_level, item1, item2, item3]")
    print(f"     - Dimension: {len(all_units)} × 5 = {len(all_units) * 5} features")
    print("     - Pros: Captures star level and items")
    print("     - Cons: Large feature space")

    print("\n   Option C: Sequence-based encoding (recommended for attention)")
    print(f"     - Sequence of {max_units} units, each encoded as vector")
    print(f"     - Per unit vector: [unit_id_embedding, star_level, cost, item_embeddings]")
    print("     - Pros: Natural for self-attention, maintains unit relationships")
    print("     - Cons: Requires embedding layers")

    print("\n2. TRAIT ENCODING:")
    print("   Option A: One-hot with activation tier")
    print(f"     - Dimension: {len(all_traits)} × 5 features (inactive + 4 tiers)")
    print("     - Pros: Simple, captures activation levels")

    print("\n   Option B: Trait vector")
    print(f"     - Per trait: [trait_id_embedding, num_units, tier_current, style]")
    print("     - Pros: More compact, embeddings can learn relationships")

    print("\n3. ITEM ENCODING:")
    print(f"   - Total unique items: {len(all_items)}")
    print("   - Can be embedded within unit encoding or separately")
    print("   - Consider: Item embeddings + unit context")

    print("\n4. GLOBAL FEATURES:")
    print("   - Player level (1-10)")
    print("   - Gold remaining")
    print("   - Board size (number of units)")
    print("   - Number of active traits")
    print("   - Total items on board")

    print("\n5. RECOMMENDED ARCHITECTURE:")
    print("   Input Layer:")
    print("     - Unit sequence: [max_units, unit_embedding_dim]")
    print("     - Trait sequence: [max_traits, trait_embedding_dim]")
    print("     - Global features: [5-10 features]")
    print("   ")
    print("   Processing:")
    print("     - Fully connected to expand features")
    print("     - Self-attention layers on unit/trait sequences")
    print("     - Concatenate attended features + global features")
    print("     - Fully connected layers for final prediction")
    print("   ")
    print("   Output:")
    print("     - Strength score (continuous value)")
    print("     - Or: Placement prediction (1-8 classification)")

    print("\n6. TRAINING APPROACH:")
    print("   - Input: 8 boards from a single match")
    print("   - Target: Relative rankings (1st to 8th)")
    print("   - Loss: Ranking loss (e.g., ListNet, RankNet) or pairwise loss")
    print("   - Alternative: Predict placement directly with cross-entropy")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Extract sample TFT matches from BigQuery for ML model development',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--num-matches', type=int, default=2,
                       help='Number of matches to extract (default: 2)')
    parser.add_argument('--output', type=str, default='sample_matches.json',
                       help='Output JSON file (default: sample_matches.json)')
    parser.add_argument('--project-id', type=str, default=None,
                       help='GCP project ID (auto-detected if not provided)')
    parser.add_argument('--dataset-id', type=str, default='tft_analytics',
                       help='BigQuery dataset ID (default: tft_analytics)')

    args = parser.parse_args()

    try:
        # Extract matches
        matches = extract_sample_matches(
            project_id=args.project_id,
            dataset_id=args.dataset_id,
            num_matches=args.num_matches
        )

        if not matches:
            print("✗ No matches found in database")
            return

        # Print human-readable summary
        print_match_summary(matches)

        # Save to JSON
        save_matches_json(matches, args.output)

        # Print encoding suggestions
        print_encoding_suggestions(matches)

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
