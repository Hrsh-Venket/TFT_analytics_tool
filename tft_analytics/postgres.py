"""
PostgreSQL Operations Module for TFT Analytics

PostgreSQL + JSONB storage for TFT match data.
Flat columns for filterable scalars, JSONB for nested arrays (units, traits, companion, missions).
"""

import os
import json
import logging
from datetime import datetime

import psycopg2
from psycopg2.extras import Json, RealDictCursor, execute_values
from psycopg2.pool import ThreadedConnectionPool

from tft_analytics.mapper import map_match_data, get_mapper

logger = logging.getLogger(__name__)


def get_db_config():
    """Read database config from environment variables."""
    return {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': int(os.getenv('POSTGRES_PORT', '5432')),
        'dbname': os.getenv('POSTGRES_DB', 'tft_analytics'),
        'user': os.getenv('POSTGRES_USER', 'tft'),
        'password': os.getenv('POSTGRES_PASSWORD', ''),
    }


# Module-level connection pool (initialized lazily)
_pool = None


def get_pool():
    """Get or create the module-level connection pool."""
    global _pool
    if _pool is None:
        config = get_db_config()
        _pool = ThreadedConnectionPool(minconn=1, maxconn=10, **config)
    return _pool


def get_connection():
    """Get a connection from the pool."""
    return get_pool().getconn()


def put_connection(conn):
    """Return a connection to the pool."""
    get_pool().putconn(conn)


def ensure_tables():
    """Create all tables/indexes if they don't exist. Safe to call on app startup."""
    importer = PostgresDataImporter()
    importer.close()


class PostgresDataImporter:
    """
    PostgreSQL data importer for TFT match data.
    PostgreSQL data importer for TFT match data.
    """

    def __init__(self):
        self.conn = get_connection()
        self.conn.autocommit = False

        self.create_tables()

        mapper = get_mapper()
        mapping_stats = mapper.get_mapping_stats()
        print(f"Name mappings loaded: {mapping_stats['units']} units, "
              f"{mapping_stats['traits']} traits, {mapping_stats['items']} items")

    def create_tables(self):
        """Create PostgreSQL tables if they don't exist."""
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS match_participants (
                    match_id            TEXT NOT NULL,
                    game_datetime       TIMESTAMPTZ NOT NULL,
                    game_creation       TIMESTAMPTZ,
                    game_length         DOUBLE PRECISION,
                    game_version        TEXT,
                    game_id             BIGINT,
                    queue_id            INTEGER,
                    tft_set_number      INTEGER,
                    tft_set_core_name   TEXT,
                    tft_game_type       TEXT,
                    end_of_game_result  TEXT,
                    map_id              INTEGER,

                    puuid               TEXT NOT NULL,
                    riot_id_game_name   TEXT,
                    riot_id_tagline     TEXT,
                    placement           INTEGER,
                    level               INTEGER,
                    last_round          INTEGER,
                    players_eliminated  INTEGER,
                    total_damage_to_players INTEGER,
                    gold_left           INTEGER,
                    time_eliminated     DOUBLE PRECISION,
                    win                 BOOLEAN,

                    units               JSONB NOT NULL DEFAULT '[]',
                    traits              JSONB NOT NULL DEFAULT '[]',
                    companion           JSONB,
                    missions            JSONB,

                    collection_start_timestamp BIGINT,
                    collection_timestamp BIGINT,
                    collection_datetime TIMESTAMPTZ,

                    PRIMARY KEY (match_id, puuid)
                );

                CREATE INDEX IF NOT EXISTS idx_mp_set_number
                    ON match_participants (tft_set_number);
                CREATE INDEX IF NOT EXISTS idx_mp_placement
                    ON match_participants (placement);
                CREATE INDEX IF NOT EXISTS idx_mp_game_datetime
                    ON match_participants (game_datetime);
                CREATE INDEX IF NOT EXISTS idx_mp_level
                    ON match_participants (level);
                CREATE INDEX IF NOT EXISTS idx_mp_game_version
                    ON match_participants USING btree (game_version text_pattern_ops);
                CREATE INDEX IF NOT EXISTS idx_mp_units
                    ON match_participants USING GIN (units jsonb_path_ops);
                CREATE INDEX IF NOT EXISTS idx_mp_traits
                    ON match_participants USING GIN (traits jsonb_path_ops);

                CREATE TABLE IF NOT EXISTS main_clusters (
                    id                  INTEGER PRIMARY KEY,
                    size                INTEGER NOT NULL,
                    avg_placement       DOUBLE PRECISION,
                    winrate             DOUBLE PRECISION,
                    top4_rate           DOUBLE PRECISION,
                    common_carries      TEXT[],
                    top_units_display   TEXT,
                    sub_cluster_ids     INTEGER[],
                    analysis_date       TIMESTAMPTZ
                );

                CREATE TABLE IF NOT EXISTS sub_clusters (
                    id                  INTEGER PRIMARY KEY,
                    main_cluster_id     INTEGER REFERENCES main_clusters(id),
                    carry_set           TEXT[],
                    size                INTEGER NOT NULL,
                    avg_placement       DOUBLE PRECISION,
                    winrate             DOUBLE PRECISION,
                    top4_rate           DOUBLE PRECISION,
                    analysis_date       TIMESTAMPTZ
                );
            """)
            self.conn.commit()
            print("Tables ready")

    def check_match_exists(self, match_id):
        """Check if match already exists in the database."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM match_participants WHERE match_id = %s LIMIT 1",
                    (match_id,)
                )
                return cur.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking match existence: {e}")
            return False

    def insert_match_data(self, match_data):
        """Insert TFT match data into PostgreSQL."""
        try:
            mapped_match_data = map_match_data(match_data)

            metadata = mapped_match_data['metadata']
            info = mapped_match_data['info']
            collection_info = mapped_match_data.get('collection_info', {})

            game_datetime = datetime.fromtimestamp(info['game_datetime'] / 1000).isoformat()
            game_creation = datetime.fromtimestamp(
                info.get('gameCreation', info['game_datetime']) / 1000
            ).isoformat()

            rows = []
            for participant in info['participants']:
                units_struct = []
                for unit in participant.get('units', []):
                    units_struct.append({
                        'character_id': unit.get('character_id', ''),
                        'tier': unit.get('tier', None),
                        'rarity': unit.get('rarity', None),
                        'name': unit.get('name', ''),
                        'item_names': unit.get('item_names', [])
                    })

                traits_struct = []
                for trait in participant.get('traits', []):
                    traits_struct.append({
                        'name': trait.get('name', ''),
                        'num_units': trait.get('num_units', None),
                        'style': trait.get('style', None),
                        'tier_current': trait.get('tier_current', None),
                        'tier_total': trait.get('tier_total', None)
                    })

                companion_struct = None
                if 'companion' in participant:
                    companion_struct = {
                        'content_id': participant['companion'].get('content_ID', ''),
                        'item_id': participant['companion'].get('item_ID', None),
                        'skin_id': participant['companion'].get('skin_ID', None),
                        'species': participant['companion'].get('species', '')
                    }

                missions_struct = None
                if 'missions' in participant:
                    missions_struct = {
                        'player_score2': participant['missions'].get('PlayerScore2', None)
                    }

                rows.append((
                    metadata['match_id'],
                    game_datetime,
                    game_creation,
                    info.get('game_length', None),
                    info.get('game_version', ''),
                    info.get('gameId', None),
                    info.get('queueId', None),
                    info.get('tft_set_number', None),
                    info.get('tft_set_core_name', ''),
                    info.get('tft_game_type', ''),
                    info.get('endOfGameResult', ''),
                    info.get('mapId', None),
                    participant.get('puuid', ''),
                    participant.get('riotIdGameName', ''),
                    participant.get('riotIdTagline', ''),
                    participant.get('placement', None),
                    participant.get('level', None),
                    participant.get('last_round', None),
                    participant.get('players_eliminated', None),
                    participant.get('total_damage_to_players', None),
                    participant.get('gold_left', None),
                    participant.get('time_eliminated', None),
                    participant.get('win', None),
                    Json(units_struct),
                    Json(traits_struct),
                    Json(companion_struct) if companion_struct else None,
                    Json(missions_struct) if missions_struct else None,
                    collection_info.get('start_timestamp', None),
                    collection_info.get('collection_timestamp', None),
                    datetime.now().isoformat(),
                ))

            with self.conn.cursor() as cur:
                execute_values(cur, """
                    INSERT INTO match_participants (
                        match_id, game_datetime, game_creation, game_length,
                        game_version, game_id, queue_id, tft_set_number,
                        tft_set_core_name, tft_game_type, end_of_game_result, map_id,
                        puuid, riot_id_game_name, riot_id_tagline,
                        placement, level, last_round, players_eliminated,
                        total_damage_to_players, gold_left, time_eliminated, win,
                        units, traits, companion, missions,
                        collection_start_timestamp, collection_timestamp, collection_datetime
                    ) VALUES %s
                    ON CONFLICT (match_id, puuid) DO NOTHING
                """, rows)
                self.conn.commit()

            return True, f"Inserted {len(rows)} participants for match {metadata['match_id']}"

        except Exception as e:
            self.conn.rollback()
            return False, f"Insert error: {str(e)}"

    def get_match_count(self):
        """Get total number of distinct matches."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT COUNT(DISTINCT match_id) FROM match_participants")
                return cur.fetchone()[0]
        except Exception:
            return 0

    def close(self):
        """Return connection to pool."""
        if self.conn:
            put_connection(self.conn)
            self.conn = None


def test_postgres_connection():
    """Test PostgreSQL connection and table creation."""
    try:
        importer = PostgresDataImporter()
        importer.close()
        return {'success': True, 'message': 'PostgreSQL connection successful'}
    except Exception as e:
        return {'success': False, 'error': str(e)}
