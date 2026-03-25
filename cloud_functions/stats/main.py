"""
Cloud Function: GET /api/stats
Returns database statistics for the TFT analytics dashboard
"""

import logging

from utils import handle_cors_preflight, json_response, error_response, get_bigquery_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_ID = 'tft_analytics'
TABLE_NAME = 'match_participants'


def get_stats(request):
    """
    Cloud Function entry point for /api/stats
    Returns match count, participant count, and summary statistics
    """
    cors_response = handle_cors_preflight(request)
    if cors_response:
        return cors_response

    try:
        client = get_bigquery_client()
        project_id = client.project
        table_id = f"{project_id}.{DATASET_ID}.{TABLE_NAME}"

        query = f"""
        SELECT
            COUNT(DISTINCT match_id) as matches,
            COUNT(*) as participants,
            AVG(placement) as avg_placement,
            AVG(level) as avg_level,
            COUNT(DISTINCT match_id) / NULLIF(COUNT(DISTINCT puuid), 0) as avg_matches_per_player,
            MAX(game_datetime) as last_updated
        FROM `{table_id}`
        """

        result = client.query(query).result()
        row = next(result)

        stats = {
            'matches': row.matches or 0,
            'participants': row.participants or 0,
            'avg_placement': round(float(row.avg_placement), 2) if row.avg_placement else 0,
            'avg_level': round(float(row.avg_level), 1) if row.avg_level else 0,
            'avg_players_per_match': 8,
            'last_updated': row.last_updated.isoformat() if row.last_updated else None
        }

        logger.info(f"Stats: {stats['matches']} matches, {stats['participants']} participants")
        return json_response(stats)

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return error_response(f"Failed to get stats: {str(e)}")
