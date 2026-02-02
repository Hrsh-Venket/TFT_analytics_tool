"""
TFT Data Collection Module

This module handles BigQuery integration and batch collection logic for TFT match data.
Uses riot_api_functions.py for all Riot Games API interactions.

Usage:
    python data_collection.py --api-key YOUR_API_KEY [OPTIONS]

Examples:
    # Basic collection with API key
    python data_collection.py --api-key RGAPI-xxxxx

    # Collection with custom settings
    python data_collection.py --api-key RGAPI-xxxxx --days 30 --tier MASTER

    # Test mode
    python data_collection.py --test-mode

    # Initialize tracker only
    python data_collection.py --init-tracker
"""

import sys
import time
import json
import os
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from riot_api_functions import RateLimiter, RiotAPIClient

try:
    from google.cloud import bigquery
    from bigquery_operations import BigQueryDataImporter
    from google.cloud.exceptions import NotFound, Forbidden
    import pandas as pd
    BIGQUERY_AVAILABLE = True
    print("BigQuery client available")
except ImportError as e:
    print(f"Warning: BigQuery functionality not available: {e}")
    BIGQUERY_AVAILABLE = False
except Exception as e:
    print(f"Warning: BigQuery client could not be initialized: {e}")
    BIGQUERY_AVAILABLE = False


def parse_max_matches(value):
    """
    Parse max-matches-per-player argument.

    Args:
        value: String value from command line

    Returns:
        int or None (None means unlimited)
    """
    if value is None:
        return None

    value_lower = value.lower()
    if value_lower in ['maximum', 'max', 'unlimited', 'infinity', 'inf', 'all']:
        return None

    try:
        num = int(value)
        if num <= 0:
            raise ValueError("max-matches-per-player must be positive")
        return num
    except ValueError:
        raise ValueError(f"Invalid max-matches-per-player value: {value}. Use a positive number or 'maximum'")


def initialize_global_tracker_from_existing_data(jsonl_files=None):
    """
    Initialize the global match tracker from existing JSONL files.
    Useful for first-time setup to avoid re-downloading existing matches.

    :param jsonl_files: List of JSONL files to scan, or None for auto-detection
    :return: Number of matches added to tracker
    """
    if jsonl_files is None:
        # Auto-detect common JSONL files
        jsonl_files = []
        for filename in ['matches.jsonl', 'matches_set15.jsonl', 'matches_set14.jsonl']:
            if os.path.exists(filename):
                jsonl_files.append(filename)

    if not jsonl_files:
        print("No existing JSONL files found to initialize tracker from.")
        return 0

    print(f"Initializing global match tracker from existing data...")
    global_tracker = GlobalMatchTracker()
    initial_count = len(global_tracker.downloaded_matches)

    total_scanned = 0
    for jsonl_file in jsonl_files:
        print(f"   Scanning {jsonl_file}...")
        file_matches = set()

        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            match = json.loads(line.strip())
                            match_id = match['metadata']['match_id']
                            file_matches.add(match_id)
                        except (KeyError, json.JSONDecodeError) as e:
                            print(f"     Warning: Invalid match data at line {line_num}: {e}")
                            continue

            print(f"     Found {len(file_matches)} matches in {jsonl_file}")
            global_tracker.mark_downloaded(list(file_matches))
            total_scanned += len(file_matches)

        except Exception as e:
            print(f"     Error reading {jsonl_file}: {e}")
            continue

    # Save the updated tracker
    global_tracker.save_tracker()

    new_matches_added = len(global_tracker.downloaded_matches) - initial_count
    print(f"\nGlobal tracker initialization complete:")
    print(f"   Files scanned: {len(jsonl_files)}")
    print(f"   Total matches scanned: {total_scanned}")
    print(f"   New matches added to tracker: {new_matches_added}")
    print(f"   Total matches in tracker: {len(global_tracker.downloaded_matches)}")

    return new_matches_added


class GlobalMatchTracker:
    """Global tracker for all downloaded matches to prevent duplicates across all sessions."""

    def __init__(self, tracker_file='global_matches_downloaded.json'):
        self.tracker_file = tracker_file
        self.downloaded_matches = set()
        self.load_tracker()

    def load_tracker(self):
        """Load the global match tracker from file."""
        if os.path.exists(self.tracker_file):
            try:
                with open(self.tracker_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.downloaded_matches = set(data.get('downloaded_matches', []))
                print(f"   Loaded global match tracker: {len(self.downloaded_matches)} matches already downloaded")
            except Exception as e:
                print(f"   Warning: Could not load global match tracker: {e}")
                self.downloaded_matches = set()

    def save_tracker(self):
        """Save the global match tracker to file."""
        try:
            data = {
                'downloaded_matches': list(self.downloaded_matches),
                'last_updated': time.time(),
                'total_matches': len(self.downloaded_matches)
            }
            with open(self.tracker_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"   Warning: Could not save global match tracker: {e}")

    def is_downloaded(self, match_id):
        """Check if a match has already been downloaded."""
        return match_id in self.downloaded_matches

    def mark_downloaded(self, match_ids):
        """Mark matches as downloaded."""
        if isinstance(match_ids, str):
            match_ids = [match_ids]

        new_matches = [mid for mid in match_ids if mid not in self.downloaded_matches]
        self.downloaded_matches.update(new_matches)
        return len(new_matches)

    def get_stats(self):
        """Get statistics about downloaded matches."""
        return {
            'total_downloaded': len(self.downloaded_matches),
            'tracker_type': 'file',
            'tracker_file': self.tracker_file
        }


class SetBasedCollectionProgress:
    """Tracks progress for set-based data collection with resume capability."""

    def __init__(self, progress_file, global_tracker=None):
        self.progress_file = progress_file
        self.processed_players = set()
        self.processed_matches = set()  # Keep for session tracking
        self.total_players = 0
        self.start_time = time.time()
        self.matches_collected = 0
        self.global_tracker = global_tracker or GlobalMatchTracker()

        self.load_progress()

    def load_progress(self):
        """Load existing progress if available."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    self.processed_players = set(data.get('processed_players', []))
                    self.processed_matches = set(data.get('processed_matches', []))
                    self.matches_collected = data.get('matches_collected', 0)
                    print(f"   Resumed: {len(self.processed_players)} players processed, {self.matches_collected} matches collected")
            except Exception as e:
                print(f"   Warning: Could not load progress file: {e}")


    def add_processed_player(self, puuid):
        """Mark a player as processed."""
        self.processed_players.add(puuid)

    def add_processed_matches(self, match_ids):
        """Mark matches as processed and update global tracker."""
        # Update session tracking
        new_matches = [mid for mid in match_ids if mid not in self.processed_matches]
        self.processed_matches.update(new_matches)

        # Update global tracking
        global_new = self.global_tracker.mark_downloaded(match_ids)
        self.matches_collected += len(new_matches)

        return len(new_matches)

    def is_player_processed(self, puuid):
        """Check if player has been processed."""
        return puuid in self.processed_players

    def is_match_processed(self, match_id):
        """Check if match has been processed (checks both session and global)."""
        return match_id in self.processed_matches or self.global_tracker.is_downloaded(match_id)

    def is_match_downloaded_globally(self, match_id):
        """Check if match has been downloaded in any previous session."""
        return self.global_tracker.is_downloaded(match_id)

    def save_progress(self):
        """Save current progress to file and update global tracker."""
        try:
            data = {
                'processed_players': list(self.processed_players),
                'processed_matches': list(self.processed_matches),
                'matches_collected': self.matches_collected,
                'timestamp': time.time()
            }
            with open(self.progress_file, 'w') as f:
                json.dump(data, f, indent=2)

            # Save global tracker
            self.global_tracker.save_tracker()
        except Exception as e:
            print(f"   Warning: Could not save progress: {e}")


    def print_progress(self):
        """Print current progress statistics."""
        elapsed = time.time() - self.start_time
        processed_count = len(self.processed_players)
        remaining = self.total_players - processed_count

        if processed_count > 0:
            rate = processed_count / elapsed * 60  # players per minute
            eta_minutes = remaining / rate if rate > 0 else 0
        else:
            rate = 0
            eta_minutes = 0

        print(f"   Progress: {processed_count}/{self.total_players} players "
              f"({processed_count/self.total_players*100:.1f}%) | "
              f"{self.matches_collected} matches | "
              f"{rate:.1f} players/min | "
              f"ETA: {eta_minutes:.0f}min")


def load_test_data():
    """Load test match data from structure.json for testing BigQuery integration"""
    try:
        with open('structure.json', 'r') as f:
            test_match = json.load(f)

        # Add collection info for BigQuery
        test_match['collection_info'] = {
            'start_timestamp': int(datetime.now().timestamp()),
            'collection_timestamp': int(datetime.now().timestamp())
        }

        print(f"Loaded test match: {test_match['metadata']['match_id']}")
        return [test_match]  # Return as list for batch processing
    except FileNotFoundError:
        print("structure.json not found - cannot run test mode")
        return []
    except Exception as e:
        print(f"Error loading test data: {e}")
        return []

def test_bigquery_integration():
    """Test BigQuery integration using structure.json data"""
    print("=" * 60)
    print("TESTING BIGQUERY INTEGRATION")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize BigQuery importer
    bigquery_importer = None
    if BIGQUERY_AVAILABLE:
        try:
            bigquery_importer = BigQueryDataImporter()
            print("BigQuery connection established")
            print(f"  Project: {bigquery_importer.project_id}")
            print(f"  Dataset: {bigquery_importer.dataset_id}")
        except Exception as e:
            print(f"BigQuery unavailable: {e}")
            print("  Will test JSONL fallback only")

    # Create tables if BigQuery is available
    if bigquery_importer:
        print("\n1. Creating BigQuery tables...")
        success, message = bigquery_importer.create_tables()
        if success:
            print(f"Tables ready: {message}")
        else:
            print(f"Table creation issue: {message}")

    # Load test data
    print("\n2. Loading test data...")
    test_matches = load_test_data()
    if not test_matches:
        print("Cannot proceed without test data")
        return

    # Test data storage
    print("\n3. Testing data storage...")
    inserted, duplicates, errors = store_matches_data(
        test_matches, None, "test_output.jsonl", bigquery_importer
    )

    print(f"  Inserted: {inserted}")
    print(f"  Duplicates: {duplicates}")
    print(f"  Errors: {errors}")

    # Test statistics if BigQuery available
    if bigquery_importer:
        print("\n4. Testing statistics queries...")
        try:
            count = bigquery_importer.get_match_count()
            print(f"Total matches in BigQuery: {count}")
        except Exception as e:
            print(f"Statistics query error: {e}")

    # Test duplicate detection
    print("\n5. Testing duplicate detection...")
    if bigquery_importer:
        match_id = test_matches[0]['metadata']['match_id']
        exists = bigquery_importer.check_match_exists(match_id)
        print(f"Match {match_id} exists: {exists}")

        # Try inserting again (should be duplicate)
        print("   Attempting duplicate insertion...")
        inserted2, duplicates2, errors2 = store_matches_data(
            test_matches, None, None, bigquery_importer
        )
        print(f"   Second attempt - Inserted: {inserted2}, Duplicates: {duplicates2}")

    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

def collect_period_match_data(api_key, start_timestamp, tier='CHALLENGER',
                                region_matches='sea', region_players='sg2',
                                output_file=None, batch_size=50, max_workers=5,
                                progress_file=None, max_matches_per_player=None):
    """
    Collect all TFT match data from a specific timestamp from high-tier players.

    :param api_key: Riot Games API key
    :param start_timestamp: Start timestamp (API seconds since June 16, 2021)
    :param tier: Minimum tier to collect data from
    :param region_matches: Region for match data
    :param region_players: Region for player rankings
    :param output_file: Output JSONL file (auto-generated if None)
    :param batch_size: Number of matches to process in each batch
    :param max_workers: Number of concurrent workers
    :param progress_file: Progress tracking file (auto-generated if None)
    :param max_matches_per_player: Maximum matches to collect per player (None = unlimited)
    """
    if output_file is None:
        output_file = 'matches.jsonl'

    if progress_file is None:
        progress_file = 'progress_period.json'

    start_date = datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d')
    print(f"=== TFT PERIOD DATA COLLECTION ===")
    print(f"Target: {tier}+ players from {region_players}")
    print(f"Start date: {start_date}")
    print(f"Output: {output_file}")
    print(f"Progress: {progress_file}")

    client = RiotAPIClient(api_key)
    global_tracker = GlobalMatchTracker()
    progress = SetBasedCollectionProgress(progress_file, global_tracker)

    # Initialize BigQuery importer
    bigquery_importer = None
    if BIGQUERY_AVAILABLE:
        try:
            bigquery_importer = BigQueryDataImporter()
            print("BigQuery connection established")
        except Exception as e:
            print(f"BigQuery unavailable: {e}")
            bigquery_importer = None

    # Use the provided start timestamp
    period_start_timestamp = start_timestamp
    print(f"Using start timestamp: {start_timestamp} ({datetime.fromtimestamp(start_timestamp)})")

    # Step 2: Get player list
    print(f"\n=== COLLECTING PLAYER LIST ===")
    players_list = client.get_players_tier_and_above(
        tier=tier, region=region_players
    )
    progress.total_players = len(players_list)

    # Filter out already processed players
    remaining_players = [p for p in players_list if not progress.is_player_processed(p['puuid'])]

    print(f"Total players: {len(players_list)}")
    print(f"Remaining players: {len(remaining_players)}")

    if not remaining_players:
        print("All players already processed!")
        return True

    # Step 3: Collect matches for each player
    print(f"\n=== COLLECTING PERIOD MATCHES ===")
    current_time = int(time.time())

    for i, player in enumerate(remaining_players, 1):
        puuid = player['puuid']
        player_name = player.get('summonerName', f'Player_{i}')

        print(f"\nPlayer {i}/{len(remaining_players)}: {player_name}")

        try:
            # Get all matches for this player within the period timeframe
            player_match_ids = client.get_period_match_ids(
                puuid, region_matches, period_start_timestamp, current_time
            )

            print(f"   Found {len(player_match_ids)} matches in period timeframe")

            # Limit matches per player if specified
            if max_matches_per_player and len(player_match_ids) > max_matches_per_player:
                player_match_ids = player_match_ids[:max_matches_per_player]
                print(f"   Limited to {max_matches_per_player} most recent matches")

            # Filter out already processed matches (including globally downloaded ones)
            session_new = [mid for mid in player_match_ids if not progress.is_match_processed(mid)]
            globally_downloaded = [mid for mid in player_match_ids if progress.is_match_downloaded_globally(mid)]

            print(f"   Already downloaded globally: {len(globally_downloaded)}")
            print(f"   New matches to collect: {len(session_new)}")

            new_match_ids = session_new

            if new_match_ids:
                # Process matches in batches
                for batch_start in range(0, len(new_match_ids), batch_size):
                    batch_ids = new_match_ids[batch_start:batch_start + batch_size]

                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_match = {
                            executor.submit(client.get_match_details, match_id, region_matches): match_id
                            for match_id in batch_ids
                        }

                        batch_details = []
                        valid_matches = []

                        for future in as_completed(future_to_match):
                            try:
                                match_details = future.result()
                                match_id = future_to_match[future]

                                # Add collection metadata
                                match_details['collection_info'] = {
                                    'start_timestamp': period_start_timestamp,
                                    'collection_timestamp': current_time
                                }
                                batch_details.append(match_details)
                                valid_matches.append(match_id)

                            except Exception as e:
                                match_id = future_to_match[future]
                                print(f"     Error fetching {match_id}: {e}")

                        # Store batch data (database and/or file)
                        if batch_details:
                            inserted, duplicates, errors = store_matches_data(
                                batch_details, progress, output_file, bigquery_importer
                            )
                            new_count = progress.add_processed_matches(valid_matches)
                            print(f"     Saved {len(batch_details)} valid matches ({new_count} new)")

            # Mark player as processed
            progress.add_processed_player(puuid)
            progress.save_progress()
            progress.print_progress()

        except Exception as e:
            print(f"   Error processing player {puuid}: {e}")
            continue

    # Final summary
    print(f"\n{'='*60}")
    print("PERIOD-BASED DATA COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Output file: {output_file}")
    print(f"Players processed: {len(progress.processed_players)}")
    print(f"Matches collected: {progress.matches_collected}")
    print(f"Collection timeframe: {datetime.fromtimestamp(period_start_timestamp)} to {datetime.now()}")

    # Clean up progress file on successful completion
    try:
        os.remove(progress_file)
        print(f"Progress file {progress_file} removed (collection complete)")
    except:
        pass

    return True


def append_to_jsonl(filename, matches):
    """
    Append matches to a JSONL file.

    :param filename: Path to the JSONL file
    :param matches: List of match dictionaries to append
    """
    with open(filename, 'a') as f:
        for match in matches:
            json.dump(match, f)
            f.write('\n')

def store_matches_data(matches, progress=None, output_file=None, bigquery_importer=None):
      """
      Store matches data to BigQuery and optionally JSONL file.

      :param matches: List of match data dictionaries
      :param progress: SetBasedCollectionProgress instance for tracking
      :param output_file: Optional JSONL output file for backup
      :param bigquery_importer: BigQueryDataImporter instance
      :return: Tuple of (inserted, duplicates, errors)
      """
      if not matches:
          return 0, 0, 0

      inserted = 0
      duplicates = 0
      errors = 0

      # Store to BigQuery if available
      if bigquery_importer and BIGQUERY_AVAILABLE:
          for match in matches:
              match_id = match['metadata']['match_id']

              # Check if match already exists
              if bigquery_importer.check_match_exists(match_id):
                  duplicates += 1
                  continue

              # Insert match data
              success, message = bigquery_importer.insert_match_data(match)
              if success:
                  inserted += 1
              else:
                  errors += 1
                  print(f"   BigQuery insert error for {match_id}: {message}")

      # Optional JSONL backup
      if output_file:
          try:
              append_to_jsonl(output_file, matches)
          except Exception as e:
              print(f"   JSONL backup error: {e}")

      return inserted, duplicates, errors


def collect_match_data(api_key, tier='CHALLENGER', region_matches='sea', region_players='sg2',
                      output_file='matches.jsonl', batch_size=50, max_workers=5):
    """
    Complete pipeline for collecting TFT match data from high-tier players.

    :param api_key: Riot Games API key
    :param tier: Minimum tier to collect data from
    :param region_matches: Region for match data
    :param region_players: Region for player rankings
    :param output_file: Output JSONL file
    :param batch_size: Number of matches to process in each batch
    :param max_workers: Number of concurrent workers for match detail fetching
    """
    print(f"Starting TFT data collection for {tier}+ players...")

    # Initialize API client and global tracker
    client = RiotAPIClient(api_key)
    global_tracker = GlobalMatchTracker()

    # Initialize BigQuery importer
    bigquery_importer = None
    if BIGQUERY_AVAILABLE:
        try:
            bigquery_importer = BigQueryDataImporter()
            print("BigQuery connection established")
        except Exception as e:
            print(f"BigQuery unavailable: {e}")
            bigquery_importer = None

    # Get match IDs
    print("1. Collecting match IDs from high-tier players...")
    all_match_ids = client.get_matches_tier_and_above(
        tier=tier,
        region_matches=region_matches,
        region_players=region_players
    )
    print(f"   Found {len(all_match_ids)} unique matches")

    # Filter out already downloaded matches
    new_match_ids = [mid for mid in all_match_ids if not global_tracker.is_downloaded(mid)]
    already_downloaded = len(all_match_ids) - len(new_match_ids)

    print(f"   Already downloaded: {already_downloaded} matches")
    print(f"   New matches to collect: {len(new_match_ids)}")

    if not new_match_ids:
        print("   All matches already downloaded!")
        return

    match_ids = new_match_ids

    # Process matches in batches
    print(f"2. Fetching match details (batch size: {batch_size}, workers: {max_workers})...")
    total_batches = (len(match_ids) + batch_size - 1) // batch_size

    for batch_num, i in enumerate(range(0, len(match_ids), batch_size), 1):
        batch_ids = match_ids[i:i + batch_size]
        print(f"   Processing batch {batch_num}/{total_batches} ({len(batch_ids)} matches)")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_match = {
                executor.submit(client.get_match_details, match_id, region_matches): match_id
                for match_id in batch_ids
            }

            batch_details = []
            for future in as_completed(future_to_match):
                try:
                    match_details = future.result()
                    batch_details.append(match_details)
                except Exception as e:
                    match_id = future_to_match[future]
                    print(f"     Error fetching details for match {match_id}: {e}")

        # Store batch data (database and/or file)
        if batch_details:
            inserted, duplicates, errors = store_matches_data(
                batch_details, None, output_file, bigquery_importer
            )

            # Mark matches as downloaded
            successful_match_ids = [match['metadata']['match_id'] for match in batch_details]
            global_tracker.mark_downloaded(successful_match_ids)
            global_tracker.save_tracker()

            print(f"     Saved {len(batch_details)} matches to {output_file}")

    print(f"\n{'='*60}")
    print("DATA COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Output file: {output_file}")
    print(f"Total matches collected: {len(match_ids)}")
    print("Ready for clustering and analysis!")


def parse_arguments():
    """Parse command-line arguments for data collection configuration."""
    parser = argparse.ArgumentParser(
        description='TFT Data Collection System - Collect match data from Riot Games API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  Basic collection:
    python data_collection.py --api-key RGAPI-xxxxx

  Custom date range (last 30 days):
    python data_collection.py --api-key RGAPI-xxxxx --days 30

  Master tier and above:
    python data_collection.py --api-key RGAPI-xxxxx --tier MASTER

  Test mode:
    python data_collection.py --test-mode

  Initialize tracker:
    python data_collection.py --init-tracker
        '''
    )

    # API and authentication
    parser.add_argument('--api-key', type=str, required=False,
                        help='Riot Games API key (required for data collection)')

    # Time range
    parser.add_argument('--days', type=int, default=45,
                        help='Number of days of match history to collect (default: 45)')

    # Tier settings
    parser.add_argument('--tier', type=str, default='CHALLENGER',
                        choices=['CHALLENGER', 'GRANDMASTER', 'MASTER', 'DIAMOND', 'EMERALD', 'PLATINUM', 'GOLD', 'SILVER', 'BRONZE'],
                        help='Minimum tier for data collection (default: CHALLENGER)')

    # Region settings
    parser.add_argument('--region-matches', type=str, default='sea',
                        help='Region for match data (default: sea)')
    parser.add_argument('--region-players', type=str, default='sg2',
                        help='Region for player rankings (default: sg2)')

    # Output settings
    parser.add_argument('--output-file', type=str, default='matches.jsonl',
                        help='Output JSONL file for match data (default: matches.jsonl)')

    # Performance settings
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size for processing matches (default: 50)')
    parser.add_argument('--max-workers', type=int, default=5,
                        help='Maximum concurrent workers (default: 5)')
    parser.add_argument('--max-matches-per-player', type=str, default=None,
                        help='Maximum matches to collect per player. Use "maximum" or "unlimited" for no limit (default: unlimited)')

    # Special modes
    parser.add_argument('--test-mode', action='store_true',
                        help='Test mode: use structure.json instead of API calls')
    parser.add_argument('--init-tracker', action='store_true',
                        help='Initialize global match tracker from existing data and exit')

    args = parser.parse_args()

    # Validate: API key required unless in special modes
    if not args.test_mode and not args.init_tracker and not args.api_key:
        parser.error('--api-key is required for data collection (unless using --test-mode or --init-tracker)')

    return args


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Parse max-matches-per-player argument
    try:
        max_matches_per_player = parse_max_matches(args.max_matches_per_player)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Calculate start timestamp for "last X days"
    # API time = seconds since June 16, 2021
    API_EPOCH = 1623772801
    current_api_time = int(time.time()) - API_EPOCH
    START_TIMESTAMP = current_api_time - (args.days * 86400)

    print("TFT Data Collection System")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Display current configuration
    print("Current Configuration:")
    if args.api_key:
        print(f"  API Key: {'*' * (len(args.api_key) - 4) + args.api_key[-4:]}")
    print(f"  Tier: {args.tier}")
    print(f"  Regions: {args.region_players} (players), {args.region_matches} (matches)")
    print(f"  Storage: {'BigQuery + JSONL' if BIGQUERY_AVAILABLE else 'JSONL only'}")
    print(f"  Output file: {args.output_file}")
    print(f"  API batch size: {args.batch_size}, Workers: {args.max_workers}")
    if max_matches_per_player is None:
        print(f"  Max matches per player: unlimited")
    else:
        print(f"  Max matches per player: {max_matches_per_player}")

    if args.test_mode:
        print(f"  Mode: TEST MODE - Using structure.json data")
    elif args.init_tracker:
        print(f"  Mode: Initialize tracker only")
    else:
        readable_time = datetime.fromtimestamp(START_TIMESTAMP).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  Mode: Period-based collection from {readable_time}")
        print(f"  Days: {args.days}")
    print()

    # Special mode: Test BigQuery integration
    if args.test_mode:
        print("TEST MODE: Testing BigQuery integration with structure.json...")
        test_bigquery_integration()
        exit(0)

    # Special mode: Initialize global tracker
    if args.init_tracker:
        print("Initializing global match tracker from existing data...")
        matches_added = initialize_global_tracker_from_existing_data()
        if matches_added > 0:
            print(f"\nSuccessfully initialized global tracker with {matches_added} matches")
        else:
            print("\nNo new matches added to tracker")
        exit(0)

    # Run period-based collection
    print(f"Starting period-based collection from timestamp {START_TIMESTAMP}")
    readable_time = datetime.fromtimestamp(API_EPOCH + START_TIMESTAMP).strftime('%Y-%m-%d %H:%M:%S')
    print(f"Start time: {readable_time}")

    success = collect_period_match_data(
        api_key=args.api_key,
        start_timestamp=START_TIMESTAMP,
        tier=args.tier,
        region_matches=args.region_matches,
        region_players=args.region_players,
        output_file=args.output_file,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        max_matches_per_player=max_matches_per_player
    )

    print("\n" + "="*60)
    if success:
        print("DATA COLLECTION COMPLETE")
    else:
        print("DATA COLLECTION FAILED")
    print("="*60)

    print("Files generated:")
    if os.path.exists(args.output_file):
        print(f"  - {args.output_file}: Match data")
    if os.path.exists('global_matches_downloaded.json'):
        print("  - global_matches_downloaded.json: Global match tracking")

    print("\nNext steps:")
    print("  - Run clustering: python clustering.py")
    print("  - Run querying: python querying.py")
    print("  - Deploy functions: ./deploy-functions.sh")
