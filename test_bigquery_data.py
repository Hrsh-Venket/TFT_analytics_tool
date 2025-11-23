#!/usr/bin/env python3
"""
Test what data is actually in BigQuery.
"""

from ml.data_loader import load_matches_from_bigquery
import json


def test_bigquery_data():
    print("="*80)
    print("TESTING BIGQUERY DATA")
    print("="*80)

    # Load just 1 match
    print("\nLoading 1 match from BigQuery...")
    matches = load_matches_from_bigquery(limit=1)

    if not matches:
        print("❌ No matches found!")
        return

    match = matches[0]
    print(f"\n✓ Loaded match: {match['match_id']}")
    print(f"  Participants: {len(match['participants'])}")

    # Inspect first participant
    participant = match['participants'][0]

    print(f"\n{'='*80}")
    print("PARTICIPANT 1 DATA:")
    print(f"{'='*80}")
    print(json.dumps(participant, indent=2))

    print(f"\n{'='*80}")
    print("ANALYSIS:")
    print(f"{'='*80}")
    print(f"Placement: {participant.get('placement', 'MISSING')}")
    print(f"Level: {participant.get('level', 'MISSING')}")
    print(f"Units count: {len(participant.get('units', []))}")
    print(f"Traits count: {len(participant.get('traits', []))}")

    if participant.get('units'):
        print(f"\nFirst unit:")
        unit = participant['units'][0]
        print(f"  Name: {unit.get('name', 'MISSING')}")
        print(f"  Tier: {unit.get('tier', 'MISSING')}")
        print(f"  Items: {unit.get('items', 'MISSING')}")

    print(f"\n{'='*80}")
    print("DIAGNOSIS:")
    print(f"{'='*80}")

    if participant.get('level', 0) == 0:
        print("❌ PROBLEM: Level is 0 or missing!")
        print("   Your BigQuery data doesn't have level information.")

    if not participant.get('units'):
        print("❌ PROBLEM: No units in participant data!")
        print("   Your BigQuery data doesn't have unit information.")
        print("   Check your data_collection.py - it may not be storing units.")
    else:
        print(f"✓ Units found: {len(participant['units'])}")

    if not participant.get('traits'):
        print("⚠️  WARNING: No traits in participant data!")
    else:
        print(f"✓ Traits found: {len(participant['traits'])}")


if __name__ == "__main__":
    test_bigquery_data()
