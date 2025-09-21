#!/usr/bin/env python3
"""
TFT Analytics BigQuery Querying System

Production-ready BigQuery-based querying for TFT match data.
Designed for Firebase webapp integration with comprehensive query capabilities.
"""

import logging
import json
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test for BigQuery availability
HAS_BIGQUERY = False
try:
    from google.cloud import bigquery
    from google.cloud.exceptions import NotFound
    HAS_BIGQUERY = True
    logger.info("BigQuery dependencies available")
except ImportError:
    logger.warning("BigQuery dependencies not available - install google-cloud-bigquery")
    bigquery = None

# Test Mode Configuration
TEST_MODE = os.getenv('TFT_TEST_MODE', 'false').lower() == 'true'

@dataclass
class DatabaseQueryFilter:
    """Represents a database query filter with condition and parameters."""
    condition: str
    params: Dict[str, Any]

class TFTQuery:
    """
    BigQuery-based TFT composition query builder for Firebase webapp integration.
    
    Provides flexible querying capabilities with method chaining, logical operations,
    and comprehensive statistical analysis. Compatible with denormalized BigQuery schema
    containing match and participant data with nested STRUCT arrays.
    
    Example Usage:
        # Basic unit query
        stats = TFTQuery().add_unit('Jinx').get_stats()
        
        # Complex logical query
        high_elo = TFTQuery().add_unit('Jinx').add_trait('Sniper', min_tier=3).get_stats()
        
        # OR logic
        versatile = TFTQuery().add_unit('Jinx').or_(TFTQuery().add_unit('Aphelios')).get_stats()
        
        # Get statistics for Jinx compositions
        stats = TFTQuery().add_unit('Jinx').get_stats()
    """
    
    def __init__(self, project_id: Optional[str] = None, dataset_id: str = 'tft_analytics'):
        """
        Initialize the BigQuery TFT query builder.
        
        Args:
            project_id: GCP project ID (auto-detected if None)
            dataset_id: BigQuery dataset ID
        """
        if not HAS_BIGQUERY and not TEST_MODE:
            raise ImportError("BigQuery dependencies not available. Install google-cloud-bigquery or enable TEST_MODE.")
        
        if HAS_BIGQUERY:
            self.client = bigquery.Client(project=project_id)
            self.project_id = project_id or self.client.project
        else:
            self.client = None
            self.project_id = project_id or "test-project"
            
        self.dataset_id = dataset_id
        self.table_id = f"{self.project_id}.{self.dataset_id}.match_participants"
        
        # Query state
        self._filters: List[DatabaseQueryFilter] = []
        
        if TEST_MODE:
            logger.info("🧪 TFT Query running in TEST MODE")
    
    def add_unit(self, unit_id: str, must_have: bool = True) -> 'TFTQuery':
        """
        Add filter for presence/absence of a specific unit.
        
        Args:
            unit_id: Unit character ID (e.g., 'Jinx', 'Aphelios')
            must_have: True to require unit presence, False to require absence
            
        Returns:
            Self for method chaining
        """
        if must_have:
            condition = """
                EXISTS (
                    SELECT 1 FROM UNNEST(units) AS unit
                    WHERE unit.character_id = @unit_id
                )
            """
        else:
            condition = """
                NOT EXISTS (
                    SELECT 1 FROM UNNEST(units) AS unit
                    WHERE unit.character_id = @unit_id
                )
            """
        
        self._filters.append(DatabaseQueryFilter(condition, {"unit_id": unit_id}))
        return self
    
    def add_unit_count(self, unit_id: str, count: int) -> 'TFTQuery':
        """
        Add filter for exact unit count of a specific unit type.
        
        Args:
            unit_id: Unit character ID
            count: Exact number of this unit required
            
        Returns:
            Self for method chaining
        """
        condition = """
            (SELECT COUNT(*) 
             FROM UNNEST(units) AS unit
             WHERE unit.character_id = @unit_id) = @count
        """
        
        self._filters.append(DatabaseQueryFilter(condition, {"unit_id": unit_id, "count": count}))
        return self
    
    def add_unit_star_level(self, unit_id: str, min_star: int = 1, max_star: int = 3) -> 'TFTQuery':
        """
        Add filter for unit star level range.
        
        Args:
            unit_id: Unit character ID
            min_star: Minimum star level (1-3)
            max_star: Maximum star level (1-3)
            
        Returns:
            Self for method chaining
        """
        condition = """
            EXISTS (
                SELECT 1 FROM UNNEST(units) AS unit
                WHERE unit.character_id = @unit_id
                AND unit.tier >= @min_star
                AND unit.tier <= @max_star
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
            unit_id: Unit character ID
            item_id: Item name (e.g., 'InfinityEdge', 'GuinsoosRageblade')

        Returns:
            Self for method chaining
        """
        # Clean up item name - remove any TFT prefix if present
        clean_item_id = item_id.replace('TFT_Item_', '').replace('TFTItem_', '').replace('TFT_', '')

        # Handle both exact matches and prefix matches
        # The actual format in database is TFT_Item_InfinityEdge
        condition = """
            EXISTS (
                SELECT 1 FROM UNNEST(units) AS unit
                CROSS JOIN UNNEST(unit.item_names) AS item
                WHERE unit.character_id = @unit_id
                AND (item = @item_id
                     OR item = CONCAT('TFT_Item_', @item_id)
                     OR item = CONCAT('TFT_', @item_id)
                     OR item = CONCAT('TFTItem_', @item_id)
                     OR REGEXP_REPLACE(item, r'^TFT[0-9]*_Item_', '') = @item_id)
            )
        """

        self._filters.append(DatabaseQueryFilter(condition, {"unit_id": unit_id, "item_id": clean_item_id}))
        return self
    
    def add_unit_item_count(self, unit_id: str, min_count: int = 0, max_count: int = 3) -> 'TFTQuery':
        """
        Add filter for number of items on a specific unit.
        
        Args:
            unit_id: Unit character ID
            min_count: Minimum number of items
            max_count: Maximum number of items
            
        Returns:
            Self for method chaining
        """
        condition = """
            EXISTS (
                SELECT 1 FROM UNNEST(units) AS unit
                WHERE unit.character_id = @unit_id
                AND ARRAY_LENGTH(unit.item_names) >= @min_count
                AND ARRAY_LENGTH(unit.item_names) <= @max_count
            )
        """
        
        self._filters.append(DatabaseQueryFilter(condition, {
            "unit_id": unit_id,
            "min_count": min_count,
            "max_count": max_count
        }))
        return self
    
    def add_trait(self, trait_name: str, min_tier: int = 1, max_tier: int = 4) -> 'TFTQuery':
        """
        Add filter for trait activation level.
        
        Args:
            trait_name: Trait name (e.g., 'Vanguard', 'Star Guardian', 'Sniper')
            min_tier: Minimum trait tier
            max_tier: Maximum trait tier
            
        Returns:
            Self for method chaining
        """
        condition = """
            EXISTS (
                SELECT 1 FROM UNNEST(traits) AS trait
                WHERE trait.name = @trait_name
                AND trait.tier_current >= @min_tier
                AND trait.tier_current <= @max_tier
            )
        """
        
        self._filters.append(DatabaseQueryFilter(condition, {
            "trait_name": trait_name,
            "min_tier": min_tier,
            "max_tier": max_tier
        }))
        return self
    
    def add_player_level(self, min_level: int = 1, max_level: int = 10) -> 'TFTQuery':
        """
        Add filter for player level range.
        
        Args:
            min_level: Minimum player level
            max_level: Maximum player level
            
        Returns:
            Self for method chaining
        """
        condition = "level >= @min_level AND level <= @max_level"
        self._filters.append(DatabaseQueryFilter(condition, {"min_level": min_level, "max_level": max_level}))
        return self
    
    def add_last_round(self, min_round: int = 1, max_round: int = 50) -> 'TFTQuery':
        """
        Add filter for last round survived range.
        
        Args:
            min_round: Minimum last round
            max_round: Maximum last round
            
        Returns:
            Self for method chaining
        """
        condition = "last_round >= @min_round AND last_round <= @max_round"
        self._filters.append(DatabaseQueryFilter(condition, {"min_round": min_round, "max_round": max_round}))
        return self
    
    def add_placement_range(self, min_placement: int = 1, max_placement: int = 8) -> 'TFTQuery':
        """
        Add filter for placement range.
        
        Args:
            min_placement: Minimum placement (1 = 1st place)
            max_placement: Maximum placement (8 = 8th place)
            
        Returns:
            Self for method chaining
        """
        condition = "placement >= @min_placement AND placement <= @max_placement"
        self._filters.append(DatabaseQueryFilter(condition, {"min_placement": min_placement, "max_placement": max_placement}))
        return self
    
    def add_set_filter(self, set_number: int) -> 'TFTQuery':
        """
        Add filter for specific TFT set.
        
        Args:
            set_number: TFT set number (e.g., 14 for Set 14)
            
        Returns:
            Self for method chaining
        """
        condition = "tft_set_number = @set_number"
        self._filters.append(DatabaseQueryFilter(condition, {"set_number": set_number}))
        return self
    
    def add_patch_filter(self, patch_version: str) -> 'TFTQuery':
        """
        Add filter for specific patch version.
        
        Args:
            patch_version: Patch version string (e.g., '14.23')
            
        Returns:
            Self for method chaining
        """
        condition = "game_version LIKE @patch_pattern"
        self._filters.append(DatabaseQueryFilter(condition, {"patch_pattern": f"Version {patch_version}%"}))
        return self
    
    def add_custom_filter(self, condition: str, params: Optional[Dict[str, Any]] = None) -> 'TFTQuery':
        """
        Add a custom BigQuery SQL filter condition.
        
        Args:
            condition: SQL WHERE condition with parameter placeholders (@param_name)
            params: Parameters for the condition
            
        Returns:
            Self for method chaining
        """
        self._filters.append(DatabaseQueryFilter(condition, params or {}))
        return self
    
    def or_(self, *other_queries: 'TFTQuery') -> 'TFTQuery':
        """
        Combine this query with other queries using OR logic.
        
        Args:
            other_queries: Other TFTQuery instances to combine with OR
            
        Returns:
            New TFTQuery instance with combined filters
        """
        if not other_queries:
            return self
        
        # Create new query instance
        new_query = TFTQuery(project_id=self.project_id, dataset_id=self.dataset_id)
        
        # Combine all filters using OR logic
        if self._filters or any(hasattr(q, '_filters') and q._filters for q in other_queries):
            all_conditions = []
            all_params = {}
            
            # Add current query conditions
            if self._filters:
                current_condition_parts = []
                for f in self._filters:
                    current_condition_parts.append(f.condition)
                    all_params.update(f.params)
                if current_condition_parts:
                    all_conditions.append(f"({' AND '.join(current_condition_parts)})")
            
            # Add other query conditions with parameter renaming to avoid conflicts
            for i, other_query in enumerate(other_queries):
                if hasattr(other_query, '_filters') and other_query._filters:
                    other_condition_parts = []
                    for j, f in enumerate(other_query._filters):
                        renamed_condition = f.condition
                        for key, value in f.params.items():
                            new_key = f"{key}_or_{i}_{j}"
                            all_params[new_key] = value
                            renamed_condition = renamed_condition.replace(f"@{key}", f"@{new_key}")
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
        
        Args:
            other_query: Optional other query to negate. If None, negates this query.
            
        Returns:
            New TFTQuery instance with NOT logic applied
            
        Usage: 
            - TFTQuery().not_(TFTQuery().add_unit('Jinx')) = NOT Jinx
            - TFTQuery().add_trait('Vanguard').not_(TFTQuery().add_unit('Jinx')) = Vanguard AND NOT Jinx
        """
        new_query = TFTQuery(project_id=self.project_id, dataset_id=self.dataset_id)
        
        if other_query is None:
            # NOT this query
            if self._filters:
                current_conditions = []
                all_params = {}
                for f in self._filters:
                    current_conditions.append(f.condition)
                    all_params.update(f.params)
                combined_condition = f"NOT ({' AND '.join(current_conditions)})"
                new_query._filters.append(DatabaseQueryFilter(combined_condition, all_params))
        else:
            # This query AND NOT other_query
            all_params = {}
            
            # Add current query conditions
            current_conditions = []
            for f in self._filters:
                current_conditions.append(f.condition)
                all_params.update(f.params)
            
            # Add NOT other_query conditions
            if hasattr(other_query, '_filters') and other_query._filters:
                other_conditions = []
                for f in other_query._filters:
                    # Rename conflicting parameters
                    renamed_condition = f.condition
                    for key, value in f.params.items():
                        if key in all_params:
                            new_key = f"{key}_not_{id(f)}"
                            all_params[new_key] = value
                            renamed_condition = renamed_condition.replace(f"@{key}", f"@{new_key}")
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
                    combined_condition = "TRUE"  # Always true if no conditions
                
                new_query._filters.append(DatabaseQueryFilter(combined_condition, all_params))
        
        return new_query
    
    def xor(self, other_query: 'TFTQuery') -> 'TFTQuery':
        """
        Combine this query with another query using XOR logic (exactly one condition true).
        
        Args:
            other_query: Other TFTQuery instance for XOR logic
            
        Returns:
            New TFTQuery instance with XOR logic applied
            
        Usage: 
            TFTQuery().add_unit('Jinx').xor(TFTQuery().add_unit('Aphelios'))
        """
        new_query = TFTQuery(project_id=self.project_id, dataset_id=self.dataset_id)
        
        if hasattr(other_query, '_filters'):
            all_params = {}
            
            # XOR: (A AND NOT B) OR (NOT A AND B)
            # Each condition appears twice, so we need unique parameters for each occurrence
            
            # First occurrence - get current query conditions (A)
            current_conditions_a1 = []
            for i, f in enumerate(self._filters):
                renamed_condition = f.condition
                for key, value in f.params.items():
                    new_key = f"{key}_a1_{i}"
                    all_params[new_key] = value
                    renamed_condition = renamed_condition.replace(f"@{key}", f"@{new_key}")
                current_conditions_a1.append(renamed_condition)
            
            # First occurrence - get other query conditions (B)
            other_conditions_b1 = []
            for i, f in enumerate(other_query._filters):
                renamed_condition = f.condition
                for key, value in f.params.items():
                    new_key = f"{key}_b1_{i}"
                    all_params[new_key] = value
                    renamed_condition = renamed_condition.replace(f"@{key}", f"@{new_key}")
                other_conditions_b1.append(renamed_condition)
            
            # Second occurrence - get current query conditions (for NOT A)
            current_conditions_a2 = []
            for i, f in enumerate(self._filters):
                renamed_condition = f.condition
                for key, value in f.params.items():
                    new_key = f"{key}_a2_{i}"
                    all_params[new_key] = value
                    renamed_condition = renamed_condition.replace(f"@{key}", f"@{new_key}")
                current_conditions_a2.append(renamed_condition)
            
            # Second occurrence - get other query conditions (B)
            other_conditions_b2 = []
            for i, f in enumerate(other_query._filters):
                renamed_condition = f.condition
                for key, value in f.params.items():
                    new_key = f"{key}_b2_{i}"
                    all_params[new_key] = value
                    renamed_condition = renamed_condition.replace(f"@{key}", f"@{new_key}")
                other_conditions_b2.append(renamed_condition)
            
            # XOR: (A AND NOT B) OR (NOT A AND B)
            if current_conditions_a1 and other_conditions_b1:
                current_clause_a1 = ' AND '.join(current_conditions_a1)
                other_clause_b1 = ' AND '.join(other_conditions_b1)
                current_clause_a2 = ' AND '.join(current_conditions_a2)
                other_clause_b2 = ' AND '.join(other_conditions_b2)
                
                xor_condition = f"(({current_clause_a1}) AND NOT ({other_clause_b1})) OR (NOT ({current_clause_a2}) AND ({other_clause_b2}))"
                new_query._filters.append(DatabaseQueryFilter(xor_condition, all_params))
        
        return new_query
    
    def _build_sql_query(self, limit: Optional[int] = None) -> tuple[str, Dict[str, Any]]:
        """
        Build the complete BigQuery SQL query with all filters.
        
        Args:
            limit: Optional limit on number of results
            
        Returns:
            Tuple of (SQL query string, parameters dict)
        """
        base_query = f"""
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
            FROM `{self.table_id}`
            WHERE 1=1
        """
        
        all_params = {}
        
        # Add all filters
        for filter_obj in self._filters:
            base_query += f" AND ({filter_obj.condition})"
            all_params.update(filter_obj.params)
        
        # Add ordering
        base_query += " ORDER BY placement ASC, game_datetime DESC"
        
        # Add limit if specified
        if limit:
            base_query += f" LIMIT {limit}"
        
        return base_query, all_params
    
    def execute(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Execute the query and return matching participants.
        
        Args:
            limit: Optional limit on number of results
            
        Returns:
            List of participant dictionaries
        """
        if TEST_MODE:
            return self._execute_test_mode(limit)
        
        try:
            query, params = self._build_sql_query(limit)
            
            # Configure query job with parameters
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        key, 
                        "STRING" if isinstance(value, str) else "FLOAT64" if isinstance(value, float) else "INTEGER", 
                        value
                    )
                    for key, value in params.items()
                ]
            )
            
            # Execute query
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            
            # Convert to list of dictionaries
            participants = []
            for row in results:
                participant = dict(row)
                participants.append(participant)
            
            logger.info(f"Query executed successfully, returned {len(participants)} participants")
            return participants
            
        except Exception as e:
            logger.error(f"BigQuery query failed: {e}")
            if TEST_MODE:
                return self._execute_test_mode(limit)
            raise
    
    def _execute_test_mode(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Execute query in test mode with sample data."""
        logger.info("🧪 Executing query in test mode with sample data")
        
        # Return sample data that matches the expected structure
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
        
        # Apply limit if specified
        if limit:
            sample_data = sample_data[:limit]
        
        return sample_data
    
    def get_stats(self) -> Optional[Dict[str, Any]]:
        """
        Execute the query and return statistical summary.
        
        Returns:
            Dictionary with play_count, avg_placement, winrate, top4_rate
        """
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
    

# Compatibility aliases for existing code
TFTQueryBigQuery = TFTQuery

def test_connection() -> Dict[str, Any]:
    """
    Test BigQuery connection and system status.
    
    Returns:
        Dictionary with connection status and basic information
    """
    try:
        if not HAS_BIGQUERY:
            if TEST_MODE:
                # In test mode, simulate successful connection
                return {
                    'success': True,
                    'message': 'Test mode - BigQuery simulation active',
                    'test_results_count': 2,
                    'project_id': 'test-project',
                    'dataset_id': 'tft_analytics',
                    'table_id': 'test-project.tft_analytics.match_participants'
                }
            else:
                return {
                    'success': False,
                    'error': 'BigQuery dependencies not available',
                    'message': 'Install google-cloud-bigquery'
                }
        
        query = TFTQuery()
        test_results = query.add_custom_filter("RAND() < 0.001").execute(limit=1)  # Tiny sample
        
        return {
            'success': True,
            'message': 'BigQuery connection successful',
            'test_results_count': len(test_results),
            'project_id': query.project_id,
            'dataset_id': query.dataset_id,
            'table_id': query.table_id
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': 'BigQuery connection failed'
        }

def main():
    """Main function for testing the querying system."""
    print("TFT Analytics BigQuery Querying System")
    print("=" * 50)
    
    # Test connection
    print("Testing BigQuery connection...")
    connection_result = test_connection()
    
    if connection_result['success']:
        print("[SUCCESS] BigQuery connection successful")
        print(f"   Project: {connection_result['project_id']}")
        print(f"   Dataset: {connection_result['dataset_id']}")
        print(f"   Test query returned: {connection_result['test_results_count']} results")
    else:
        print("[ERROR] BigQuery connection failed")
        print(f"   Error: {connection_result['error']}")
        # Don't return - continue with test mode if available
        if not TEST_MODE:
            print("   Enable test mode with: set TFT_TEST_MODE=true")
            return
    
    print("\nTesting query functionality...")
    
    # Test basic query
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
    
    # Test unit query
    print("\n2. Unit query test...")
    try:
        unit_query = TFTQuery().add_unit('Jinx')
        jinx_stats = unit_query.get_stats()
        if jinx_stats and jinx_stats['play_count'] > 0:
            print(f"[SUCCESS] Jinx query successful:")
            print(f"   Play count: {jinx_stats['play_count']}")
            print(f"   Average placement: {jinx_stats['avg_placement']}")
            print(f"   Win rate: {jinx_stats['winrate']}%")
        else:
            print("[INFO] No Jinx results found - this is normal if using different unit names")
    except Exception as e:
        print(f"[ERROR] Unit query failed: {e}")
    
    # Test logical operations
    print("\n3. Logical operations test...")
    try:
        or_query = TFTQuery().add_unit('Jinx').or_(TFTQuery().add_unit('Aphelios'))
        or_stats = or_query.get_stats()
        if or_stats:
            print(f"[SUCCESS] OR query (Jinx OR Aphelios) successful:")
            print(f"   Play count: {or_stats['play_count']}")
            print(f"   Average placement: {or_stats['avg_placement']}")
        else:
            print("[INFO] No OR query results found")
    except Exception as e:
        print(f"[ERROR] OR query failed: {e}")
    
    print("\n" + "=" * 50)
    print("[SUCCESS] Query system test completed!")
    print("\nExample usage for Firebase webapp:")
    print("```python")
    print("from querying import TFTQuery")
    print("")
    print("# Get Jinx statistics")
    print("stats = TFTQuery().add_unit('Jinx').get_stats()")
    print("")
    print("# Get high-level Aphelios statistics")
    print("stats = TFTQuery().add_unit('Aphelios').add_player_level(min_level=8).get_stats()")
    print("")
    print("# Complex logical query")
    print("versatile = TFTQuery().add_unit('Jinx').or_(TFTQuery().add_unit('Aphelios')).add_trait('Sniper', min_tier=2)")
    print("results = versatile.get_stats()")
    print("```")

if __name__ == "__main__":
    main()