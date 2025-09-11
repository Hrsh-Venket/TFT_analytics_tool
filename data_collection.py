"""
TFT Data Collection Module

This module handles all Riot Games API interactions for collecting TFT match data.
Includes rate limiting, player lookup, match history, and bulk data collection.
"""

# ===============================
# CONFIGURATION - EDIT THESE VALUES
# ===============================

API_KEY = "RGAPI-c5a7e012-13e0-40b2-9f5b-480679108e45"  # Replace with your actual Riot API key
days = 43
START_TIMESTAMP = (130032000 + (days * 86400))  # Epoch timestamp for period-based collection
# 0 is June 16th, 2021, counts seconds from them 
# 130002000 is July 30th 2025

INIT_TRACKER_ONLY = False      # Set to True to only initialize tracker
TEST_MODE = True               # Set to True to test with structure.json instead of API calls

# Collection settings
TIER = 'CHALLENGER'            # Minimum tier for data collection
REGION_MATCHES = 'sea'         # Region for match data
REGION_PLAYERS = 'sg2'         # Region for player rankings
OUTPUT_FILE = 'matches.jsonl'  # Output file (for backward compatibility)
BATCH_SIZE = 50                # Batch size for processing
MAX_WORKERS = 5                # Maximum concurrent workers


# ===============================
# END CONFIGURATION
# ===============================

import requests
import time
import json
import threading
import os
from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

try: 
    from google.cloud import bigquery
    from bigquery_opertions import BigQueryDataImporter
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


class RateLimiter:
    """
    Thread-safe rate limiter for Riot Games API.
    Handles both per-second and per-window rate limits.
    """
    
    def __init__(self, max_per_second=20, max_per_window=100, window_seconds=120):
        self.max_per_second = max_per_second
        self.max_per_window = max_per_window
        self.window_seconds = window_seconds
        self.second_requests = deque()
        self.window_requests = deque()
        self.lock = threading.Lock()

    def wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        with self.lock:
            while True:
                now = time.time()
                # Clean window
                while self.window_requests and self.window_requests[0] < now - self.window_seconds:
                    self.window_requests.popleft()
                # Check window
                if len(self.window_requests) >= self.max_per_window:
                    sleep_time = self.window_requests[0] + self.window_seconds - now + 0.01
                    time.sleep(sleep_time)
                    continue
                # Clean second
                while self.second_requests and self.second_requests[0] < now - 1:
                    self.second_requests.popleft()
                # Check second
                if len(self.second_requests) >= self.max_per_second:
                    sleep_time = self.second_requests[0] + 1 - now + 0.01
                    time.sleep(sleep_time)
                    continue
                break

    def record_request(self):
        """Record that a request was made."""
        with self.lock:
            now = time.time()
            self.window_requests.append(now)
            self.second_requests.append(now)


class RiotAPIClient:
    """
    Client for interacting with Riot Games TFT API.
    Handles rate limiting and retries automatically.
    """
    
    def __init__(self, api_key, max_per_second=20, max_per_window=100, window_seconds=120):
        self.api_key = api_key
        self.limiter = RateLimiter(max_per_second, max_per_window, window_seconds)
    
    def _api_get(self, url):
        """Make a rate-limited API request with retry logic."""
        while True:
            self.limiter.wait_if_needed()
            resp = requests.get(url)
            if resp.status_code != 429:
                self.limiter.record_request()
                return resp
            retry_after = int(resp.headers.get('Retry-After', '10'))
            time.sleep(retry_after)
    
    def get_puuid(self, summoner_name, tag_line, region):
        """
        Get the PUUID of a player using their Riot ID and tag line.
        
        :param summoner_name: The player's Riot ID
        :param tag_line: The player's tag line
        :param region: The region of the player
        :return: The PUUID of the player
        """
        api_url = (
            f"https://{region}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/"
            f"{summoner_name}/{tag_line}?api_key={self.api_key}"
        )
        
        resp = self._api_get(api_url)
        if resp.status_code == 200:
            player_info = resp.json()
            return player_info['puuid']
        else:
            raise Exception(f"Failed to get PUUID: {resp.status_code} - {resp.text}")

    def get_last_match_ids(self, puuid, region, start='0', count='20'):
        """
        Get the last matches played by a player.
        
        :param puuid: Player's PUUID
        :param region: Region of the player
        :param start: Starting index for match history (default is 0)
        :param count: Number of matches to retrieve (default is 20)
        :return: List of match IDs
        """
        api_url = (
            f"https://{region}.api.riotgames.com/tft/match/v1/matches/by-puuid/"
            f"{puuid}/ids?start={start}&count={count}&api_key={self.api_key}"
        )
        
        resp = self._api_get(api_url)
        if resp.status_code == 200:
            return resp.json()
        else:
            raise Exception(f"Failed to get match IDs: {resp.status_code} - {resp.text}")

    def get_period_match_ids(self, puuid, region, start_time, end_time):
        """
        Get matches played by a player in a specific time period.
        
        :param puuid: Player's PUUID
        :param region: Region of the player
        :param start_time: Epoch timestamp in seconds
        :param end_time: Epoch timestamp in seconds
        :return: List of match IDs
        """
        api_url = (
            f"https://{region}.api.riotgames.com/tft/match/v1/matches/by-puuid/"
            f"{puuid}/ids?endTime={end_time}&startTime={start_time}&api_key={self.api_key}"
        )
        
        resp = self._api_get(api_url)
        if resp.status_code == 200:
            return resp.json()
        else:
            raise Exception(f"Failed to get period match IDs: {resp.status_code} - {resp.text}")

    def get_match_details(self, match_id, region):
        """
        Get the details of a specific match.
        
        :param match_id: ID of the match
        :param region: Region of the player
        :return: Match details as a JSON object
        """
        api_url = (
            f"https://{region}.api.riotgames.com/tft/match/v1/matches/"
            f"{match_id}?api_key={self.api_key}"
        )
        
        resp = self._api_get(api_url)
        if resp.status_code == 200:
            return resp.json()
        else:
            raise Exception(f"Failed to get match details: {resp.status_code} - {resp.text}")

    def get_league_by_puuid(self, puuid, region):
        """
        Get the league information of a player using their PUUID.
        
        :param puuid: Player's PUUID
        :param region: Region of the player
        :return: League information as a JSON object
        """
        api_url = (
            f"https://{region}.api.riotgames.com/tft/league/v1/by-puuid/"
            f"{puuid}?api_key={self.api_key}"
        )
        
        resp = self._api_get(api_url)
        if resp.status_code == 200:
            return resp.json()
        else:
            raise Exception(f"Failed to get league info: {resp.status_code} - {resp.text}")

    def get_players_list(self, region, tier, division='I', queue='RANKED_TFT'):
        """
        Get a list of players in a specific league tier and division.
        
        :param region: Region of the players
        :param tier: League tier in lowercase (e.g., diamond)
        :param division: League division (e.g., III). Default 'I' if Master, Grandmaster, or Challenger
        :param queue: Queue type (default is 'RANKED_TFT')
        :return: List of players in the specified league
        """
        if tier not in ['master', 'grandmaster', 'challenger']:
            return self._get_low_tier_players_list(region, tier.upper(), division, queue)
        else:
            return self._get_high_tier_players_list(region, tier, queue)

    def _get_low_tier_players_list(self, region, tier, division, queue):
        """Helper function to get players list for low tiers with pagination."""
        players_list = []
        page = 1
        
        while True:
            api_url = (
                f"https://{region}.api.riotgames.com/tft/league/v1/entries/"
                f"{tier}/{division}?queue={queue}&page={page}&api_key={self.api_key}"
            )
            
            resp = self._api_get(api_url)
            if resp.status_code != 200:
                break
                
            resp_list = resp.json()
            if not resp_list:
                break
                
            players_list.extend(resp_list)
            page += 1
            
        return players_list

    def _get_high_tier_players_list(self, region, tier, queue):
        """Helper function to get players list for high tiers (no pagination needed)."""
        api_url = (
            f"https://{region}.api.riotgames.com/tft/league/v1/"
            f"{tier}?queue={queue}&api_key={self.api_key}"
        )
        
        resp = self._api_get(api_url)
        if resp.status_code == 200:
            return resp.json()
        else:
            raise Exception(f"Failed to get high tier players: {resp.status_code} - {resp.text}")

    def get_players_tier_and_above(self, tier, region='sg2', division='I', queue='RANKED_TFT'):
        """
        Get all players in a specific tier and above.
        
        :param tier: Starting tier (e.g., 'EMERALD')
        :param region: Region of the players
        :param division: Division of the players (default is 'I')
        :param queue: Queue type (default is 'RANKED_TFT')
        :return: List of all players in the tier and above
        """
        tier = tier.upper()
        tiers = ['BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'EMERALD', 'DIAMOND', 'MASTER', 'GRANDMASTER', 'CHALLENGER']
        divisions = ['I', 'II', 'III', 'IV']
        
        if tier not in tiers:
            raise ValueError(f"Invalid tier. Must be one of: {', '.join(tiers)}")
        if division not in divisions:
            raise ValueError(f"Invalid division. Must be one of: {', '.join(divisions)}")
        
        tier_index = tiers.index(tier)
        div_index = divisions.index(division) if tier not in ['MASTER', 'GRANDMASTER', 'CHALLENGER'] else 0
        
        players_list = []
        
        for t in tiers[tier_index:]:
            if t in ['MASTER', 'GRANDMASTER', 'CHALLENGER']:
                high_tier_data = self.get_players_list(region=region, tier=t.lower(), queue=queue)
                players_list.extend(high_tier_data['entries'])
                print(f"Retrieved players for tier: {t.lower()}")
            else:
                # For lower tiers, get all divisions from current up to I
                start_div = div_index if t == tier else 3  # Start from IV for subsequent tiers
                for div_idx in range(start_div, -1, -1):
                    players_list.extend(self.get_players_list(
                        region=region, tier=t.lower(), division=divisions[div_idx], queue=queue
                    ))
                    print(f"Retrieved players for tier: {t} division: {divisions[div_idx]}")
                
        return players_list

    def get_matches_tier_and_above(self, tier, region_matches='sea', region_players='sg2', division='I', queue='RANKED_TFT'):
        """
        Get match IDs for all players in a specific tier and above.
        
        :param tier: Starting tier (e.g., 'EMERALD')
        :param region_matches: Region for match data (e.g., 'sea')
        :param region_players: Region for player rankings (e.g., 'sg2')
        :param division: Division of the players (default is 'I')
        :param queue: Queue type (default is 'RANKED_TFT')
        :return: List of unique match IDs
        """
        players_list = self.get_players_tier_and_above(
            tier=tier, region=region_players, division=division, queue=queue
        )
        
        match_ids = []
        for player in players_list:
            puuid = player['puuid']
            try:
                player_match_ids = self.get_last_match_ids(puuid=puuid, region=region_matches)
                match_ids.extend(player_match_ids)
            except Exception as e:
                print(f"Error getting matches for player {puuid}: {e}")
                continue
        
        # Remove duplicates
        return list(set(match_ids))


# Set detection logic removed - use manual timestamp instead


# Set detection data loading removed - use manual timestamp instead


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
        
        print(f"✓ Loaded test match: {test_match['metadata']['match_id']}")
        return [test_match]  # Return as list for batch processing
    except FileNotFoundError:
        print("✗ structure.json not found - cannot run test mode")
        return []
    except Exception as e:
        print(f"✗ Error loading test data: {e}")
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
            print("✓ BigQuery connection established")
            print(f"  Project: {bigquery_importer.project_id}")
            print(f"  Dataset: {bigquery_importer.dataset_id}")
        except Exception as e:
            print(f"✗ BigQuery unavailable: {e}")
            print("  Will test JSONL fallback only")
    
    # Create tables if BigQuery is available
    if bigquery_importer:
        print("\n1. Creating BigQuery tables...")
        success, message = bigquery_importer.create_tables()
        if success:
            print(f"✓ Tables ready: {message}")
        else:
            print(f"⚠ Table creation issue: {message}")
    
    # Load test data
    print("\n2. Loading test data...")
    test_matches = load_test_data()
    if not test_matches:
        print("✗ Cannot proceed without test data")
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
            print(f"✓ Total matches in BigQuery: {count}")
        except Exception as e:
            print(f"⚠ Statistics query error: {e}")
    
    # Test duplicate detection
    print("\n5. Testing duplicate detection...")
    if bigquery_importer:
        match_id = test_matches[0]['metadata']['match_id']
        exists = bigquery_importer.check_match_exists(match_id)
        print(f"✓ Match {match_id} exists: {exists}")
        
        # Try inserting again (should be duplicate)
        print("   Attempting duplicate insertion...")
        inserted2, duplicates2, errors2 = store_matches_data(
            test_matches, None, None, bigquery_importer
        )
        print(f"   Second attempt - Inserted: {inserted2}, Duplicates: {duplicates2}")
    
    print(f"\n✓ Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

def collect_period_match_data(api_key, start_timestamp, tier='CHALLENGER', 
                                region_matches='sea', region_players='sg2',
                                output_file=None, batch_size=50, max_workers=5,
                                progress_file=None):
    """
    Collect all TFT match data from a specific timestamp from high-tier players.
    
    :param api_key: Riot Games API key
    :param start_timestamp: Start timestamp (epoch seconds)
    :param tier: Minimum tier to collect data from
    :param region_matches: Region for match data
    :param region_players: Region for player rankings
    :param output_file: Output JSONL file (auto-generated if None)
    :param batch_size: Number of matches to process in each batch
    :param max_workers: Number of concurrent workers
    :param progress_file: Progress tracking file (auto-generated if None)
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
            print("✓ BigQuery connection established")
        except Exception as e:
            print(f"⚠ BigQuery unavailable: {e}")
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
            print("✓ BigQuery connection established")
        except Exception as e:
            print(f"⚠ BigQuery unavailable: {e}")
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


if __name__ == "__main__":
    print("TFT Data Collection System")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check API key
    if API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Please set your API key in the API_KEY variable at the top of this file")
        exit(1)
    

    # Display current configuration
    print("Current Configuration:")
    print(f"  API Key: {'*' * (len(API_KEY) - 4) + API_KEY[-4:]}")
    print(f"  Tier: {TIER}")
    print(f"  Regions: {REGION_PLAYERS} (players), {REGION_MATCHES} (matches)")
    print(f"  Storage: {'BigQuery + JSONL' if BIGQUERY_AVAILABLE else 'JSONL only'}")
    print(f"  Output file: {OUTPUT_FILE}")
    print(f"  API batch size: {BATCH_SIZE}, Workers: {MAX_WORKERS}")
    
    if TEST_MODE:
        print(f"  Mode: TEST MODE - Using structure.json data")
    elif INIT_TRACKER_ONLY:
        print(f"  Mode: Initialize tracker only")
    else:
        readable_time = datetime.fromtimestamp(START_TIMESTAMP).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  Mode: Period-based collection from {readable_time}")
    print()
    
    # Special mode: Test BigQuery integration
    if TEST_MODE:
        print("🧪 TEST MODE: Testing BigQuery integration with structure.json...")
        test_bigquery_integration()
        exit(0)
    
    # Special mode: Initialize global tracker
    if INIT_TRACKER_ONLY:
        print("Initializing global match tracker from existing data...")
        matches_added = initialize_global_tracker_from_existing_data()
        if matches_added > 0:
            print(f"\nSuccessfully initialized global tracker with {matches_added} matches")
        else:
            print("\nNo new matches added to tracker")
        exit(0)
    
    # Run period-based collection
    print(f"Starting period-based collection from timestamp {START_TIMESTAMP}")
    readable_time = datetime.fromtimestamp(START_TIMESTAMP).strftime('%Y-%m-%d %H:%M:%S')
    print(f"Start time: {readable_time}")
    
    # Use configured batch size
    effective_batch_size = BATCH_SIZE
    
    success = collect_period_match_data(
        api_key=API_KEY,
        start_timestamp=START_TIMESTAMP,
        tier=TIER,
        region_matches=REGION_MATCHES,
        region_players=REGION_PLAYERS,
        output_file=OUTPUT_FILE,
        batch_size=effective_batch_size,
        max_workers=MAX_WORKERS
    )
    
    print("\n" + "="*60)
    if success:
        print("DATA COLLECTION COMPLETE")
    else:
        print("DATA COLLECTION FAILED")
    print("="*60)
    # Clean slate - ready for BigQuery statistics
    
    print("Files generated:")
    if os.path.exists(OUTPUT_FILE):
        print(f"  - {OUTPUT_FILE}: Match data")
    if os.path.exists('global_matches_downloaded.json'):
        print("  - global_matches_downloaded.json: Global match tracking")
    
    print("\nNext steps:")
    print("  - Run clustering: python clustering.py")
    print("  - Run querying: python querying.py")
    print("  - Add BigQuery integration for cloud storage")
    print("\nTo change settings, edit the configuration variables at the top of this file.")