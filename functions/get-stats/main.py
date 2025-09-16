"""
Cloud Function: GET /api/stats
Returns BigQuery dataset statistics for TFT analytics dashboard
"""

import sys
import os

from utils import handle_cors_preflight, json_response, error_response, get_bigquery_client
from bigquery_operations import BigQueryDataImporter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_stats(request):
    """
    Cloud Function entry point for /api/stats
    Returns database statistics including match count, participant count, last updated
    """

    # Handle CORS preflight
    cors_response = handle_cors_preflight(request)
    if cors_response:
        return cors_response

    try:
        # Initialize BigQuery connection
        logger.info("Initializing BigQuery connection for stats")
        importer = BigQueryDataImporter()

        # Get basic statistics from BigQuery
        client = get_bigquery_client()

        # Query statistics from the combined match_participants table
        stats_query = f"""
        SELECT
            COUNT(DISTINCT match_id) as total_matches,
            COUNT(*) as total_participants,
            COUNT(DISTINCT game_version) as versions_covered,
            AVG(level) as avg_level,
            AVG(placement) as avg_placement,
            MIN(game_datetime) as earliest_match,
            MAX(game_datetime) as latest_match
        FROM `{importer.match_participants_table}`
        """

        logger.info("Executing BigQuery statistics query")
        result = client.query(stats_query).result()

        # Process results
        stats_row = next(result)

        stats = {
            'matches': int(stats_row.total_matches),
            'participants': int(stats_row.total_participants),
            'avg_players_per_match': round(stats_row.total_participants / max(stats_row.total_matches, 1), 1),
            'versions_covered': int(stats_row.versions_covered),
            'avg_level': round(float(stats_row.avg_level), 1) if stats_row.avg_level else 0,
            'avg_placement': round(float(stats_row.avg_placement), 1) if stats_row.avg_placement else 0,
            'earliest_match': stats_row.earliest_match.isoformat() if stats_row.earliest_match else None,
            'latest_match': stats_row.latest_match.isoformat() if stats_row.latest_match else None,
            'last_updated': stats_row.latest_match.isoformat() if stats_row.latest_match else None
        }

        logger.info(f"Stats query successful: {stats['matches']} matches, {stats['participants']} participants")
        return json_response(stats)

    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return error_response(f"Failed to get statistics: {str(e)}")