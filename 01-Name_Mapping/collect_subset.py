"""
Collect Sample Match Data for Name Mapping Initialization

This script collects a subset of TFT matches and extracts only the
units, traits, and items data needed for name mapping initialization.

Usage:
    python collect_subset.py --api-key RGAPI-xxxxx [OPTIONS]

Examples:
    # Collect 100 matches from SILVER+ players
    python collect_subset.py --api-key RGAPI-xxxxx --num-matches 100 --tier SILVER

    # Collect from last 7 days, MASTER+ tier
    python collect_subset.py --api-key RGAPI-xxxxx --num-matches 300 --tier MASTER --days 7
"""

import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "02-Data_Collection"))

from riot_api_functions import RiotAPIClient

# Output file location
OUTPUT_FILE = Path(__file__).parent / "subset.json"


def extract_match_data(match_details: dict) -> dict:
    """
    Extract only units, traits, and items from match data.

    Args:
        match_details: Full match data from Riot API

    Returns:
        Stripped down dict with only relevant fields
    """
    extracted = {
        'match_id': match_details.get('metadata', {}).get('match_id', 'unknown'),
        'participants': []
    }

    participants = match_details.get('info', {}).get('participants', [])

    for participant in participants:
        participant_data = {
            'units': [],
            'traits': []
        }

        # Extract units with their items
        for unit in participant.get('units', []):
            unit_data = {
                'character_id': unit.get('character_id', ''),
                'item_names': unit.get('itemNames', [])
            }
            participant_data['units'].append(unit_data)

        # Extract traits
        for trait in participant.get('traits', []):
            trait_data = {
                'name': trait.get('name', '')
            }
            participant_data['traits'].append(trait_data)

        extracted['participants'].append(participant_data)

    return extracted


def collect_subset(api_key: str, num_matches: int, tier: str,
                   region_matches: str, region_players: str, days: int) -> list:
    """
    Collect sample matches and extract relevant data.

    Args:
        api_key: Riot API key
        num_matches: Target number of matches to collect
        tier: Minimum tier for players
        region_matches: Region for match data
        region_players: Region for player rankings
        days: Number of days to look back

    Returns:
        List of extracted match data
    """
    print(f"=== TFT Sample Data Collection ===")
    print(f"Target: {num_matches} matches from {tier}+ players")
    print(f"Region: {region_players} (players), {region_matches} (matches)")
    print(f"Days: {days}")
    print()

    client = RiotAPIClient(api_key)

    # Calculate time range
    current_time = int(time.time())
    start_time = current_time - (days * 86400)

    # Get player list
    print(f"Fetching {tier}+ player list...")
    players = client.get_players_tier_and_above(tier=tier, region=region_players)
    print(f"Found {len(players)} players")

    # Collect match IDs from players until we have enough
    print(f"\nCollecting match IDs...")
    all_match_ids = set()
    players_checked = 0

    for player in players:
        if len(all_match_ids) >= num_matches:
            break

        puuid = player.get('puuid')
        if not puuid:
            continue

        try:
            match_ids = client.get_period_match_ids(
                puuid, region_matches, start_time, current_time
            )
            all_match_ids.update(match_ids)
            players_checked += 1

            if players_checked % 10 == 0:
                print(f"  Checked {players_checked} players, found {len(all_match_ids)} unique matches")

        except Exception as e:
            print(f"  Error getting matches for player: {e}")
            continue

    # Limit to requested number
    match_ids = list(all_match_ids)[:num_matches]
    print(f"\nCollecting details for {len(match_ids)} matches...")

    # Fetch match details
    extracted_matches = []
    errors = 0

    for i, match_id in enumerate(match_ids, 1):
        try:
            match_details = client.get_match_details(match_id, region_matches)
            extracted = extract_match_data(match_details)
            extracted_matches.append(extracted)

            if i % 20 == 0:
                print(f"  Processed {i}/{len(match_ids)} matches")

        except Exception as e:
            errors += 1
            print(f"  Error fetching match {match_id}: {e}")
            continue

    print(f"\nCollection complete: {len(extracted_matches)} matches, {errors} errors")
    return extracted_matches


def save_subset(matches: list, output_file: Path):
    """Save extracted match data to JSON file."""
    data = {
        'collected_at': datetime.now().isoformat(),
        'num_matches': len(matches),
        'matches': matches
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"Saved to {output_file}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Collect TFT match data for name mapping initialization',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--api-key', type=str, required=True,
                        help='Riot Games API key')
    parser.add_argument('--num-matches', type=int, default=100,
                        help='Number of matches to collect (default: 100)')
    parser.add_argument('--tier', type=str, default='SILVER',
                        choices=['CHALLENGER', 'GRANDMASTER', 'MASTER', 'DIAMOND',
                                'EMERALD', 'PLATINUM', 'GOLD', 'SILVER', 'BRONZE'],
                        help='Minimum tier for players (default: SILVER)')
    parser.add_argument('--region-matches', type=str, default='sea',
                        help='Region for match data (default: sea)')
    parser.add_argument('--region-players', type=str, default='sg2',
                        help='Region for player rankings (default: sg2)')
    parser.add_argument('--days', type=int, default=7,
                        help='Days of match history to search (default: 7)')
    parser.add_argument('--output', type=str, default=None,
                        help=f'Output file (default: {OUTPUT_FILE})')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    output_file = Path(args.output) if args.output else OUTPUT_FILE

    print(f"TFT Sample Data Collection")
    print(f"=" * 40)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    matches = collect_subset(
        api_key=args.api_key,
        num_matches=args.num_matches,
        tier=args.tier,
        region_matches=args.region_matches,
        region_players=args.region_players,
        days=args.days
    )

    if matches:
        save_subset(matches, output_file)
        print(f"\nNext step: Run name_mapper.py to initialize Latest_Mappings")
        print(f"  python name_mapper.py --init-from-subset")
    else:
        print("\nNo matches collected. Check API key and parameters.")
