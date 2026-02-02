"""
Riot Games TFT API Functions

This module contains all Riot Games API interaction code for TFT data collection.
Includes rate limiting, API client, and player/match lookup functions.
"""

import requests
import time
import threading
from collections import deque


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
