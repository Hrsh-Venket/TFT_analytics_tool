#!/usr/bin/env python3
"""
TFT Analytics PostgreSQL Querying System

Production-ready PostgreSQL-based querying for TFT match data.
Uses JSONB columns for nested unit/trait data with jsonb_array_elements for filtering.
"""

import logging
import json
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

from psycopg2.extras import RealDictCursor

from tft_analytics.postgres import get_connection, put_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_MODE = os.getenv('TFT_TEST_MODE', 'false').lower() == 'true'


@dataclass
class DatabaseQueryFilter:
    """Represents a database query filter with condition and parameters."""
    condition: str
    params: Dict[str, Any]


class TFTQuery:
    """
    PostgreSQL-based TFT composition query builder.

    Provides flexible querying capabilities with method chaining, logical operations,
    and comprehensive statistical analysis. Uses JSONB columns for nested array data.

    Example Usage:
        stats = TFTQuery().add_unit('Jinx').get_stats()
        high_elo = TFTQuery().add_unit('Jinx').add_trait('Sniper', min_tier=3).get_stats()
        versatile = TFTQuery().add_unit('Jinx').or_(TFTQuery().add_unit('Aphelios')).get_stats()
    """

    def __init__(self):
        if TEST_MODE:
            logger.info("TFTQuery running in TEST MODE")

        self._filters: List[DatabaseQueryFilter] = []

    def add_unit(self, unit_id: str, must_have: bool = True) -> 'TFTQuery':
        """
        Add filter for presence/absence of a specific unit.

        Args:
            unit_id: Unit name (e.g., 'Jinx', 'Aphelios')
            must_have: True to require unit presence, False to require absence
        """
        if must_have:
            condition = """
                EXISTS (
                    SELECT 1 FROM jsonb_array_elements(units) AS unit
                    WHERE unit->>'name' = %(unit_id)s
                )
            """
        else:
            condition = """
                NOT EXISTS (
                    SELECT 1 FROM jsonb_array_elements(units) AS unit
                    WHERE unit->>'name' = %(unit_id)s
                )
            """

        self._filters.append(DatabaseQueryFilter(condition, {"unit_id": unit_id}))
        return self

    def add_unit_count(self, unit_id: str, count: int) -> 'TFTQuery':
        """Add filter for exact unit count of a specific unit type."""
        condition = """
            (SELECT COUNT(*)
             FROM jsonb_array_elements(units) AS unit
             WHERE unit->>'name' = %(unit_id)s) = %(count)s
        """

        self._filters.append(DatabaseQueryFilter(condition, {"unit_id": unit_id, "count": count}))
        return self

    def add_unit_star_level(self, unit_id: str, min_star: int = 1, max_star: int = 3) -> 'TFTQuery':
        """Add filter for unit star level range."""
        condition = """
            EXISTS (
                SELECT 1 FROM jsonb_array_elements(units) AS unit
                WHERE unit->>'name' = %(unit_id)s
                AND (unit->>'tier')::int >= %(min_star)s
                AND (unit->>'tier')::int <= %(max_star)s
            )
        """

        self._filters.append(DatabaseQueryFilter(condition, {
            "unit_id": unit_id,
            "min_star": min_star,
            "max_star": max_star
        }))
        return self

    def add_item_on_unit(self, unit_id: str, item_id: str) -> 'TFTQuery':
        """
        Add filter for specific item on specific unit.

        Args:
            unit_id: Unit name
            item_id: Item name (e.g., 'Infinity_Edge')
        """
        condition = """
            EXISTS (
                SELECT 1 FROM jsonb_array_elements(units) AS unit
                CROSS JOIN jsonb_array_elements_text(unit->'item_names') AS item
                WHERE unit->>'name' = %(unit_id)s
                AND item = %(item_id)s
            )
        """

        self._filters.append(DatabaseQueryFilter(condition, {"unit_id": unit_id, "item_id": item_id}))
        return self

    def add_unit_item_count(self, unit_id: str, min_count: int = 0, max_count: int = 3) -> 'TFTQuery':
        """Add filter for number of items on a specific unit."""
        condition = """
            EXISTS (
                SELECT 1 FROM jsonb_array_elements(units) AS unit
                WHERE unit->>'name' = %(unit_id)s
                AND jsonb_array_length(unit->'item_names') >= %(min_count)s
                AND jsonb_array_length(unit->'item_names') <= %(max_count)s
            )
        """

        self._filters.append(DatabaseQueryFilter(condition, {
            "unit_id": unit_id,
            "min_count": min_count,
            "max_count": max_count
        }))
        return self

    def add_trait(self, trait_name: str, min_tier: int = 1, max_tier: int = 4) -> 'TFTQuery':
        """Add filter for trait activation level."""
        condition = """
            EXISTS (
                SELECT 1 FROM jsonb_array_elements(traits) AS trait
                WHERE trait->>'name' = %(trait_name)s
                AND (trait->>'tier_current')::int >= %(min_tier)s
                AND (trait->>'tier_current')::int <= %(max_tier)s
            )
        """

        self._filters.append(DatabaseQueryFilter(condition, {
            "trait_name": trait_name,
            "min_tier": min_tier,
            "max_tier": max_tier
        }))
        return self

    def add_player_level(self, min_level: int = 1, max_level: int = 10) -> 'TFTQuery':
        """Add filter for player level range."""
        condition = "level >= %(min_level)s AND level <= %(max_level)s"
        self._filters.append(DatabaseQueryFilter(condition, {"min_level": min_level, "max_level": max_level}))
        return self

    def add_last_round(self, min_round: int = 1, max_round: int = 50) -> 'TFTQuery':
        """Add filter for last round survived range."""
        condition = "last_round >= %(min_round)s AND last_round <= %(max_round)s"
        self._filters.append(DatabaseQueryFilter(condition, {"min_round": min_round, "max_round": max_round}))
        return self

    def add_placement_range(self, min_placement: int = 1, max_placement: int = 8) -> 'TFTQuery':
        """Add filter for placement range."""
        condition = "placement >= %(min_placement)s AND placement <= %(max_placement)s"
        self._filters.append(DatabaseQueryFilter(condition, {"min_placement": min_placement, "max_placement": max_placement}))
        return self

    def add_set_filter(self, set_number: int) -> 'TFTQuery':
        """Add filter for specific TFT set."""
        condition = "tft_set_number = %(set_number)s"
        self._filters.append(DatabaseQueryFilter(condition, {"set_number": set_number}))
        return self

    def add_patch_filter(self, patch_version: str) -> 'TFTQuery':
        """Add filter for specific patch version."""
        condition = "game_version LIKE %(patch_pattern)s"
        self._filters.append(DatabaseQueryFilter(condition, {"patch_pattern": f"Version {patch_version}%"}))
        return self

    def add_custom_filter(self, condition: str, params: Optional[Dict[str, Any]] = None) -> 'TFTQuery':
        """Add a custom SQL filter condition with %(param_name)s placeholders."""
        self._filters.append(DatabaseQueryFilter(condition, params or {}))
        return self

    def or_(self, *other_queries: 'TFTQuery') -> 'TFTQuery':
        """Combine this query with other queries using OR logic."""
        if not other_queries:
            return self

        new_query = TFTQuery()

        if self._filters or any(hasattr(q, '_filters') and q._filters for q in other_queries):
            all_conditions = []
            all_params = {}

            if self._filters:
                current_condition_parts = []
                for f in self._filters:
                    current_condition_parts.append(f.condition)
                    all_params.update(f.params)
                if current_condition_parts:
                    all_conditions.append(f"({' AND '.join(current_condition_parts)})")

            for i, other_query in enumerate(other_queries):
                if hasattr(other_query, '_filters') and other_query._filters:
                    other_condition_parts = []
                    for j, f in enumerate(other_query._filters):
                        renamed_condition = f.condition
                        for key, value in f.params.items():
                            new_key = f"{key}_or_{i}_{j}"
                            all_params[new_key] = value
                            renamed_condition = renamed_condition.replace(
                                f"%({key})s", f"%({new_key})s"
                            )
                        other_condition_parts.append(renamed_condition)
                    if other_condition_parts:
                        all_conditions.append(f"({' AND '.join(other_condition_parts)})")

            if all_conditions:
                combined_condition = ' OR '.join(all_conditions)
                new_query._filters.append(DatabaseQueryFilter(combined_condition, all_params))

        return new_query

    def not_(self, other_query: Optional['TFTQuery'] = None) -> 'TFTQuery':
        """
        Apply NOT logic to this query or to another query.

        Usage:
            TFTQuery().not_(TFTQuery().add_unit('Jinx')) = NOT Jinx
            TFTQuery().add_trait('Vanguard').not_(TFTQuery().add_unit('Jinx')) = Vanguard AND NOT Jinx
        """
        new_query = TFTQuery()

        if other_query is None:
            if self._filters:
                current_conditions = []
                all_params = {}
                for f in self._filters:
                    current_conditions.append(f.condition)
                    all_params.update(f.params)
                combined_condition = f"NOT ({' AND '.join(current_conditions)})"
                new_query._filters.append(DatabaseQueryFilter(combined_condition, all_params))
        else:
            all_params = {}

            current_conditions = []
            for f in self._filters:
                current_conditions.append(f.condition)
                all_params.update(f.params)

            if hasattr(other_query, '_filters') and other_query._filters:
                other_conditions = []
                for f in other_query._filters:
                    renamed_condition = f.condition
                    for key, value in f.params.items():
                        if key in all_params:
                            new_key = f"{key}_not_{id(f)}"
                            all_params[new_key] = value
                            renamed_condition = renamed_condition.replace(
                                f"%({key})s", f"%({new_key})s"
                            )
                        else:
                            all_params[key] = value
                    other_conditions.append(renamed_condition)

                if current_conditions and other_conditions:
                    combined_condition = f"({' AND '.join(current_conditions)}) AND NOT ({' AND '.join(other_conditions)})"
                elif current_conditions:
                    combined_condition = ' AND '.join(current_conditions)
                elif other_conditions:
                    combined_condition = f"NOT ({' AND '.join(other_conditions)})"
                else:
                    combined_condition = "TRUE"

                new_query._filters.append(DatabaseQueryFilter(combined_condition, all_params))

        return new_query

    def xor(self, other_query: 'TFTQuery') -> 'TFTQuery':
        """Combine this query with another using XOR logic (exactly one condition true)."""
        new_query = TFTQuery()

        if hasattr(other_query, '_filters'):
            all_params = {}

            current_conditions_a1 = []
            for i, f in enumerate(self._filters):
                renamed_condition = f.condition
                for key, value in f.params.items():
                    new_key = f"{key}_a1_{i}"
                    all_params[new_key] = value
                    renamed_condition = renamed_condition.replace(f"%({key})s", f"%({new_key})s")
                current_conditions_a1.append(renamed_condition)

            other_conditions_b1 = []
            for i, f in enumerate(other_query._filters):
                renamed_condition = f.condition
                for key, value in f.params.items():
                    new_key = f"{key}_b1_{i}"
                    all_params[new_key] = value
                    renamed_condition = renamed_condition.replace(f"%({key})s", f"%({new_key})s")
                other_conditions_b1.append(renamed_condition)

            current_conditions_a2 = []
            for i, f in enumerate(self._filters):
                renamed_condition = f.condition
                for key, value in f.params.items():
                    new_key = f"{key}_a2_{i}"
                    all_params[new_key] = value
                    renamed_condition = renamed_condition.replace(f"%({key})s", f"%({new_key})s")
                current_conditions_a2.append(renamed_condition)

            other_conditions_b2 = []
            for i, f in enumerate(other_query._filters):
                renamed_condition = f.condition
                for key, value in f.params.items():
                    new_key = f"{key}_b2_{i}"
                    all_params[new_key] = value
                    renamed_condition = renamed_condition.replace(f"%({key})s", f"%({new_key})s")
                other_conditions_b2.append(renamed_condition)

            if current_conditions_a1 and other_conditions_b1:
                current_clause_a1 = ' AND '.join(current_conditions_a1)
                other_clause_b1 = ' AND '.join(other_conditions_b1)
                current_clause_a2 = ' AND '.join(current_conditions_a2)
                other_clause_b2 = ' AND '.join(other_conditions_b2)

                xor_condition = (
                    f"(({current_clause_a1}) AND NOT ({other_clause_b1})) OR "
                    f"(NOT ({current_clause_a2}) AND ({other_clause_b2}))"
                )
                new_query._filters.append(DatabaseQueryFilter(xor_condition, all_params))

        return new_query

    def _build_sql_query(self, limit: Optional[int] = None) -> tuple[str, Dict[str, Any]]:
        """Build the complete PostgreSQL query with all filters."""
        base_query = """
            SELECT
                match_id,
                puuid,
                riot_id_game_name,
                riot_id_tagline,
                placement,
                level,
                last_round,
                players_eliminated,
                total_damage_to_players,
                gold_left,
                units,
                traits,
                companion,
                missions,
                game_datetime,
                game_creation,
                game_version,
                tft_set_number,
                tft_set_core_name,
                tft_game_type
            FROM match_participants
            WHERE 1=1
        """

        all_params = {}

        for filter_obj in self._filters:
            base_query += f" AND ({filter_obj.condition})"
            all_params.update(filter_obj.params)

        base_query += " ORDER BY placement ASC, game_datetime DESC"

        if limit:
            base_query += f" LIMIT {limit}"

        return base_query, all_params

    def execute(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Execute the query and return matching participants."""
        if TEST_MODE:
            return self._execute_test_mode(limit)

        conn = get_connection()
        try:
            query, params = self._build_sql_query(limit)

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                participants = [dict(row) for row in cur.fetchall()]

            logger.info(f"Query returned {len(participants)} participants")
            return participants

        except Exception as e:
            logger.error(f"Query failed: {e}")
            if TEST_MODE:
                return self._execute_test_mode(limit)
            raise
        finally:
            put_connection(conn)

    def _execute_test_mode(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Execute query in test mode with sample data."""
        logger.info("Executing query in test mode with sample data")

        sample_data = [
            {
                "match_id": "NA1_TEST_001",
                "puuid": "test_player_1",
                "riot_id_game_name": "TestPlayer",
                "riot_id_tagline": "NA1",
                "placement": 1,
                "level": 9,
                "last_round": 35,
                "players_eliminated": 3,
                "total_damage_to_players": 12500,
                "gold_left": 15,
                "units": [
                    {"character_id": "Jinx", "tier": 3, "item_names": ["Infinity Edge", "Last Whisper"]},
                    {"character_id": "Aphelios", "tier": 2, "item_names": ["Guinsoo's Rageblade"]}
                ],
                "traits": [
                    {"name": "Sniper", "tier_current": 4, "num_units": 2},
                    {"name": "Star Guardian", "tier_current": 3, "num_units": 3}
                ],
                "game_datetime": datetime.now(),
                "tft_set_number": 14
            },
            {
                "match_id": "NA1_TEST_002",
                "puuid": "test_player_2",
                "riot_id_game_name": "TestPlayer2",
                "riot_id_tagline": "NA1",
                "placement": 4,
                "level": 8,
                "last_round": 28,
                "players_eliminated": 1,
                "total_damage_to_players": 8500,
                "gold_left": 5,
                "units": [
                    {"character_id": "Jinx", "tier": 2, "item_names": ["Rapid Firecannon"]},
                    {"character_id": "Ezreal", "tier": 3, "item_names": ["Shojin", "Blue Buff"]}
                ],
                "traits": [
                    {"name": "Sniper", "tier_current": 2, "num_units": 2},
                    {"name": "Chrono", "tier_current": 2, "num_units": 2}
                ],
                "game_datetime": datetime.now(),
                "tft_set_number": 14
            }
        ]

        if limit:
            sample_data = sample_data[:limit]

        return sample_data

    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Execute the query and return statistical summary."""
        participants = self.execute()
        if not participants:
            return None

        count = len(participants)
        avg_place = sum(p['placement'] for p in participants) / count
        winrate = (sum(1 for p in participants if p['placement'] == 1) / count) * 100
        top4_rate = (sum(1 for p in participants if p['placement'] <= 4) / count) * 100

        return {
            'play_count': count,
            'avg_placement': round(avg_place, 2),
            'winrate': round(winrate, 2),
            'top4_rate': round(top4_rate, 2)
        }


TFTQueryBigQuery = TFTQuery


def test_connection() -> Dict[str, Any]:
    """Test PostgreSQL connection and system status."""
    try:
        if TEST_MODE:
            return {
                'success': True,
                'message': 'Test mode - PostgreSQL simulation active',
                'test_results_count': 2,
            }

        query = TFTQuery()
        test_results = query.add_custom_filter("random() < 0.001").execute(limit=1)

        return {
            'success': True,
            'message': 'PostgreSQL connection successful',
            'test_results_count': len(test_results),
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': 'PostgreSQL connection failed'
        }


def main():
    """Main function for testing the querying system."""
    print("TFT Analytics PostgreSQL Querying System")
    print("=" * 50)

    print("Testing PostgreSQL connection...")
    connection_result = test_connection()

    if connection_result['success']:
        print(f"[SUCCESS] {connection_result['message']}")
    else:
        print(f"[ERROR] {connection_result['message']}")
        if not TEST_MODE:
            print("   Enable test mode with: export TFT_TEST_MODE=true")
            return

    print("\nTesting query functionality...")

    print("\n1. Basic query test...")
    try:
        query = TFTQuery()
        stats = query.add_custom_filter("placement <= 4").get_stats()
        if stats:
            print(f"[SUCCESS] Top 4 players stats:")
            print(f"   Play count: {stats['play_count']}")
            print(f"   Average placement: {stats['avg_placement']}")
            print(f"   Win rate: {stats['winrate']}%")
            print(f"   Top 4 rate: {stats['top4_rate']}%")
        else:
            print("[INFO] No results found")
    except Exception as e:
        print(f"[ERROR] Basic query failed: {e}")

    print("\n2. Unit query test...")
    try:
        jinx_stats = TFTQuery().add_unit('Jinx').get_stats()
        if jinx_stats and jinx_stats['play_count'] > 0:
            print(f"[SUCCESS] Jinx query: {jinx_stats['play_count']} games, "
                  f"avg placement {jinx_stats['avg_placement']}")
        else:
            print("[INFO] No Jinx results found")
    except Exception as e:
        print(f"[ERROR] Unit query failed: {e}")

    print("\n3. OR query test...")
    try:
        or_stats = TFTQuery().add_unit('Jinx').or_(TFTQuery().add_unit('Aphelios')).get_stats()
        if or_stats:
            print(f"[SUCCESS] Jinx OR Aphelios: {or_stats['play_count']} games")
        else:
            print("[INFO] No OR query results found")
    except Exception as e:
        print(f"[ERROR] OR query failed: {e}")

    print("\n" + "=" * 50)
    print("[DONE] Query system test completed!")


if __name__ == "__main__":
    main()
