#!/usr/bin/env python3
"""
Extract Sample Matches for ML Model Development

Simple script to extract raw match data from BigQuery.
"""

import argparse
import json
from google.cloud import bigquery


def extract_sample_matches(project_id=None, dataset_id='tft_analytics', num_matches=2):
    """Extract sample matches with all participant data from BigQuery."""
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

    print(f"Extracting {num_matches} matches from BigQuery...")
    print(f"Project: {project_id}, Dataset: {dataset_id}\n")

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

        # Convert units
        units = []
        if row.units:
            for unit in row.units:
                units.append({
                    'character_id': unit['character_id'],
                    'name': unit['name'],
                    'tier': unit['tier'],
                    'rarity': unit['rarity'],
                    'items': list(unit['item_names']) if unit.get('item_names') else []
                })

        # Convert traits
        traits = []
        if row.traits:
            for trait in row.traits:
                traits.append({
                    'name': trait['name'],
                    'num_units': trait['num_units'],
                    'style': trait['style'],
                    'tier_current': trait['tier_current'],
                    'tier_total': trait['tier_total']
                })

        # Add participant
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
            'traits': traits
        }

        matches[match_id]['participants'].append(participant)

    return list(matches.values())


def main():
    parser = argparse.ArgumentParser(description='Extract sample TFT matches from BigQuery')
    parser.add_argument('--num-matches', type=int, default=2, help='Number of matches to extract')
    parser.add_argument('--output', type=str, default='sample_matches.json', help='Output JSON file')
    parser.add_argument('--project-id', type=str, default=None, help='GCP project ID')
    parser.add_argument('--dataset-id', type=str, default='tft_analytics', help='BigQuery dataset ID')
    args = parser.parse_args()

    try:
        matches = extract_sample_matches(
            project_id=args.project_id,
            dataset_id=args.dataset_id,
            num_matches=args.num_matches
        )

        if not matches:
            print("No matches found")
            return

        # Print raw data as JSON
        print(json.dumps(matches, indent=2))

        # Save to file
        with open(args.output, 'w') as f:
            json.dump(matches, f, indent=2)

        print(f"\n\nSaved to {args.output}")
        print(f"Extracted {len(matches)} matches with {sum(len(m['participants']) for m in matches)} participants")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
