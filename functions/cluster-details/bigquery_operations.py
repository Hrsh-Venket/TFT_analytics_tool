"""
BigQuery Operations Module for TFT Analytics

This module handles all BigQuery interactions for storing TFT match data.
Replaces PostgreSQL functionality with BigQuery's columnar storage and JSON operations.
"""

from google.cloud import bigquery
from google.cloud.exceptions import NotFound, Conflict
import pandas as pd
from datetime import datetime
import json
from name_mapper import map_match_data, get_mapper

class BigQueryDataImporter:
    """
    BigQuery data importer for TFT match data.
    Handles table creation, data insertion, and querying operations.
    """
    
    def __init__(self, project_id=None, dataset_id='tft_analytics', cluster_dataset_id='tft_clusters'):
        self.client = bigquery.Client(project=project_id)  # Auto-auth on GCP VM
        self.dataset_id = dataset_id
        self.cluster_dataset_id = cluster_dataset_id
        self.project_id = project_id or self.client.project

        # Table references for BigQuery operations (match data)
        self.matches_table = f"{self.project_id}.{self.dataset_id}.matches"
        self.participants_table = f"{self.project_id}.{self.dataset_id}.participants"

        # Cluster table references (separate dataset)
        self.main_clusters_table = f"{self.project_id}.{self.cluster_dataset_id}.main_clusters"
        self.sub_clusters_table = f"{self.project_id}.{self.cluster_dataset_id}.sub_clusters"
        
        # Ensure dataset exists
        self.create_dataset_if_not_exists()
        
        # Initialize name mapper and log mapping stats
        mapper = get_mapper()
        mapping_stats = mapper.get_mapping_stats()
        print(f"📝 Name mappings loaded: {mapping_stats['units']} units, {mapping_stats['traits']} traits, {mapping_stats['items']} items")
        
    def create_dataset_if_not_exists(self):
        """Create BigQuery dataset if it doesn't exist"""
        dataset_ref = self.client.dataset(self.dataset_id)
        try:
            self.client.get_dataset(dataset_ref)
            print(f"Dataset {self.dataset_id} already exists")
        except NotFound:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"  # Always Free tier location
            self.client.create_dataset(dataset)
            print(f"Created dataset {self.dataset_id}")
            
            # Create tables after dataset creation
            success, message = self.create_tables()
            if not success:
                print(f"Warning: {message}")
    
    def create_tables(self):
          """Create BigQuery tables with proper schema for TFT data"""

          # Define the comprehensive schema for match participants table
          match_participants_schema = [
              # Match-level data
              bigquery.SchemaField("match_id", "STRING", mode="REQUIRED"),
              bigquery.SchemaField("game_datetime", "TIMESTAMP", mode="REQUIRED"),  # Partition field
              bigquery.SchemaField("game_creation", "TIMESTAMP", mode="NULLABLE"),
              bigquery.SchemaField("game_length", "FLOAT64", mode="NULLABLE"),
              bigquery.SchemaField("game_version", "STRING", mode="NULLABLE"),
              bigquery.SchemaField("game_id", "INTEGER", mode="NULLABLE"),
              bigquery.SchemaField("queue_id", "INTEGER", mode="NULLABLE"),
              bigquery.SchemaField("tft_set_number", "INTEGER", mode="NULLABLE"),
              bigquery.SchemaField("tft_set_core_name", "STRING", mode="NULLABLE"),
              bigquery.SchemaField("tft_game_type", "STRING", mode="NULLABLE"),
              bigquery.SchemaField("end_of_game_result", "STRING", mode="NULLABLE"),
              bigquery.SchemaField("map_id", "INTEGER", mode="NULLABLE"),

              # Participant-level data
              bigquery.SchemaField("puuid", "STRING", mode="REQUIRED"),
              bigquery.SchemaField("riot_id_game_name", "STRING", mode="NULLABLE"),
              bigquery.SchemaField("riot_id_tagline", "STRING", mode="NULLABLE"),
              bigquery.SchemaField("placement", "INTEGER", mode="NULLABLE"),
              bigquery.SchemaField("level", "INTEGER", mode="NULLABLE"),
              bigquery.SchemaField("last_round", "INTEGER", mode="NULLABLE"),
              bigquery.SchemaField("players_eliminated", "INTEGER", mode="NULLABLE"),
              bigquery.SchemaField("total_damage_to_players", "INTEGER", mode="NULLABLE"),
              bigquery.SchemaField("gold_left", "INTEGER", mode="NULLABLE"),
              bigquery.SchemaField("time_eliminated", "FLOAT64", mode="NULLABLE"),
              bigquery.SchemaField("win", "BOOLEAN", mode="NULLABLE"),

              # Complex nested data as STRUCT arrays
              bigquery.SchemaField("units", "RECORD", mode="REPEATED", fields=[
                  bigquery.SchemaField("character_id", "STRING", mode="NULLABLE"),
                  bigquery.SchemaField("tier", "INTEGER", mode="NULLABLE"),
                  bigquery.SchemaField("rarity", "INTEGER", mode="NULLABLE"),
                  bigquery.SchemaField("name", "STRING", mode="NULLABLE"),
                  bigquery.SchemaField("item_names", "STRING", mode="REPEATED")
              ]),

              bigquery.SchemaField("traits", "RECORD", mode="REPEATED", fields=[
                  bigquery.SchemaField("name", "STRING", mode="NULLABLE"),
                  bigquery.SchemaField("num_units", "INTEGER", mode="NULLABLE"),
                  bigquery.SchemaField("style", "INTEGER", mode="NULLABLE"),
                  bigquery.SchemaField("tier_current", "INTEGER", mode="NULLABLE"),
                  bigquery.SchemaField("tier_total", "INTEGER", mode="NULLABLE")
              ]),

              # fairly useless data but keeping it in case I want to display stats for this. Could be cute
              bigquery.SchemaField("companion", "RECORD", mode="NULLABLE", fields=[
                  bigquery.SchemaField("content_id", "STRING", mode="NULLABLE"),
                  bigquery.SchemaField("item_id", "INTEGER", mode="NULLABLE"),
                  bigquery.SchemaField("skin_id", "INTEGER", mode="NULLABLE"),
                  bigquery.SchemaField("species", "STRING", mode="NULLABLE")
              ]),

              bigquery.SchemaField("missions", "RECORD", mode="NULLABLE", fields=[
                  bigquery.SchemaField("player_score2", "INTEGER", mode="NULLABLE")
              ]),

              # Collection metadata
              bigquery.SchemaField("collection_start_timestamp", "INTEGER", mode="NULLABLE"),
              bigquery.SchemaField("collection_timestamp", "INTEGER", mode="NULLABLE"),
              bigquery.SchemaField("collection_datetime", "TIMESTAMP", mode="NULLABLE")
          ]

          # Create the table with partitioning and clustering
          table_id = f"{self.project_id}.{self.dataset_id}.match_participants"
          table = bigquery.Table(table_id, schema=match_participants_schema)

          # Set up time partitioning by game_datetime
          table.time_partitioning = bigquery.TimePartitioning(
              type_=bigquery.TimePartitioningType.DAY,
              field="game_datetime"
          )

          # Set up clustering for optimal query performance
          table.clustering_fields = ["tft_set_number", "placement", "level"]

          try:
              table = self.client.create_table(table, exists_ok=True)
              print(f"✓ Created partitioned table: {table.table_id}")
              print(f"  - Partitioned by: DATE(game_datetime)")
              print(f"  - Clustered by: tft_set_number, placement, level")
              print(f"  - Schema: {len(match_participants_schema)} fields with nested STRUCT arrays")
              return True, f"Successfully created table {table.table_id}"

          except Exception as e:
              print(f"✗ Error creating table: {e}")
              return False, f"Error creating table: {e}"
        
    def check_match_exists(self, match_id):
        """Check if match already exists in BigQuery"""
        
        query = f"""
        SELECT COUNT(*) as count
        FROM `{self.project_id}.{self.dataset_id}.match_participants`
        WHERE match_id = @match_id
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
              bigquery.ScalarQueryParameter("match_id", "STRING", match_id)
            ]
        )
        try:
            result = self.client.query(query, job_config=job_config).result()
            count = next(result).count
            return count > 0
        except Exception as e:
            print(f"Error checking match existence: {e}")
            return False
    
    def insert_match_data(self, match_data):
        """Insert TFT match data into BigQuery tables"""
        
        try:
            # Apply name mappings to clean unit, trait, and item names
            mapped_match_data = map_match_data(match_data)
            
            #extract match metadata (from mapped data)
            metadata = mapped_match_data['metadata']
            info = mapped_match_data['info']
            collection_info = mapped_match_data.get('collection_info', {})
            
            # convert game_datetime from milliseconds to ISO format string for BigQuery
            game_datetime = datetime.fromtimestamp(info['game_datetime'] / 1000).isoformat()
            game_creation = datetime.fromtimestamp(info.get('gameCreation', info['game_datetime']) / 1000).isoformat()
            
            # Prepare participant rows (one row per participant)
            participant_rows = []

            for participant in info['participants']:
                # Convert units to BigQuery STRUCT format
                units_struct = []
                for unit in participant.get('units', []):
                    # Use mapped item_names (already mapped by map_match_data above)
                    # The name mapper converts itemNames -> item_names and applies mappings
                    units_struct.append({
                        'character_id': unit.get('character_id', ''),
                        'tier': unit.get('tier', None),
                        'rarity': unit.get('rarity', None),
                        'name': unit.get('name', ''),
                        'item_names': unit.get('item_names', [])  # Use mapped item_names, not raw itemNames
                    })

                # Convert traits to BigQuery STRUCT format
                traits_struct = []
                for trait in participant.get('traits', []):
                    traits_struct.append({
                        'name': trait.get('name', ''),
                        'num_units': trait.get('num_units', None),
                        'style': trait.get('style', None),
                        'tier_current': trait.get('tier_current', None),
                        'tier_total': trait.get('tier_total', None)
                    })

                # Convert companion to STRUCT (or None)
                companion_struct = None
                if 'companion' in participant:
                    companion_struct = {
                        'content_id': participant['companion'].get('content_ID', ''),
                        'item_id': participant['companion'].get('item_ID', None),
                        'skin_id': participant['companion'].get('skin_ID', None),
                        'species': participant['companion'].get('species', '')
                    }

                # Convert missions to STRUCT (or None)
                missions_struct = None
                if 'missions' in participant:
                    missions_struct = {
                        'player_score2': participant['missions'].get('PlayerScore2', None)
                    }

                # Create participant row with all match + participant data
                participant_row = {
                    # Match-level data (duplicated for each participant)
                    'match_id': metadata['match_id'],
                    'game_datetime': game_datetime,
                    'game_creation': game_creation,
                    'game_length': info.get('game_length', None),
                    'game_version': info.get('game_version', ''),
                    'game_id': info.get('gameId', None),
                    'queue_id': info.get('queueId', None),
                    'tft_set_number': info.get('tft_set_number', None),
                    'tft_set_core_name': info.get('tft_set_core_name', ''),
                    'tft_game_type': info.get('tft_game_type', ''),
                    'end_of_game_result': info.get('endOfGameResult', ''),
                    'map_id': info.get('mapId', None),

                    # Participant-level data
                    'puuid': participant.get('puuid', ''),
                    'riot_id_game_name': participant.get('riotIdGameName', ''),
                    'riot_id_tagline': participant.get('riotIdTagline', ''),
                    'placement': participant.get('placement', None),
                    'level': participant.get('level', None),
                    'last_round': participant.get('last_round', None),
                    'players_eliminated': participant.get('players_eliminated', None),
                    'total_damage_to_players': participant.get('total_damage_to_players', None),
                    'gold_left': participant.get('gold_left', None),
                    'time_eliminated': participant.get('time_eliminated', None),
                    'win': participant.get('win', None),

                    # Complex nested data
                    'units': units_struct,
                    'traits': traits_struct,
                    'companion': companion_struct,
                    'missions': missions_struct,

                    # Collection metadata
                    'collection_start_timestamp': collection_info.get('start_timestamp', None),
                    'collection_timestamp': collection_info.get('collection_timestamp', None),
                    'collection_datetime': datetime.now().isoformat()
                }

                participant_rows.append(participant_row)

            # Insert all participant rows for this match
            table = self.client.get_table(f"{self.project_id}.{self.dataset_id}.match_participants")
            errors = self.client.insert_rows_json(table, participant_rows)

            if errors:
                return False, f"Insert errors: {errors}"

            return True, f"Successfully inserted {len(participant_rows)} participants for match {metadata['match_id']}"

        except Exception as e:
              return False, f"Insert error: {str(e)}"
    
    def get_match_count(self):
        """Get total number of matches in BigQuery"""
        query = f"SELECT COUNT(*) as count FROM `{self.matches_table}`"
        try:
            result = self.client.query(query).result()
            return next(result).count
        except:
            return 0

# Helper functions for compatibility with existing code
def test_bigquery_connection(project_id=None):
    """Test BigQuery connection and dataset creation"""
    try:
        importer = BigQueryDataImporter(project_id=project_id)
        return {
            'success': True,
            'message': 'BigQuery connection successful',
            'project_id': importer.project_id,
            'dataset_id': importer.dataset_id
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def get_bigquery_stats(project_id=None):
    """Get BigQuery database statistics"""
    try:
        importer = BigQueryDataImporter(project_id=project_id)
        matches = importer.get_match_count()
        
        # Get participant count
        query = f"SELECT COUNT(*) as count FROM `{importer.participants_table}`"
        result = importer.client.query(query).result()
        participants = next(result).count
        
        return {
            'matches': matches,
            'participants': participants,
            'units': participants * 7,  # Approximate
            'traits': participants * 3   # Approximate  
        }
    except Exception as e:
        return {'error': str(e)}