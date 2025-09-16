"""
Cloud Function: GET /api/stats - Standalone version
Returns BigQuery dataset statistics for TFT analytics dashboard
"""

import json
from datetime import datetime
from typing import Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Embedded utility functions
def cors_headers(origin: str = '*') -> Dict[str, str]:
    """Standard CORS headers for Firebase frontend"""
    return {
        'Access-Control-Allow-Origin': origin,
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        'Access-Control-Max-Age': '3600'
    }

def handle_cors_preflight(request):
    """Handle CORS preflight requests"""
    if request.method == 'OPTIONS':
        headers = cors_headers()
        return ('', 204, headers)
    return None

def json_response(data: Any, status: int = 200) -> tuple:
    """Create JSON response with CORS headers"""
    headers = cors_headers()
    headers['Content-Type'] = 'application/json'
    return (json.dumps(data), status, headers)

def error_response(message: str, status: int = 500) -> tuple:
    """Create error response with CORS headers"""
    return json_response({'error': message}, status)

def get_bigquery_client():
    """Get BigQuery client with error handling"""
    try:
        from google.cloud import bigquery
        return bigquery.Client()
    except Exception as e:
        raise Exception(f"Failed to initialize BigQuery client: {str(e)}")

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
        client = get_bigquery_client()

        # Use your existing dataset configuration
        project_id = client.project
        dataset_id = 'tft_analytics'
        matches_table = f"{project_id}.{dataset_id}.matches"
        participants_table = f"{project_id}.{dataset_id}.participants"

        # Query match statistics
        matches_query = f"""
        SELECT
            COUNT(*) as total_matches,
            COUNT(DISTINCT game_version) as versions_covered,
            MIN(game_datetime) as earliest_match,
            MAX(game_datetime) as latest_match
        FROM `{matches_table}`
        """

        # Query participant statistics
        participants_query = f"""
        SELECT
            COUNT(*) as total_participants,
            AVG(level) as avg_level,
            AVG(placement) as avg_placement
        FROM `{participants_table}`
        """

        logger.info("Executing BigQuery statistics queries")
        matches_result = client.query(matches_query).result()
        participants_result = client.query(participants_query).result()

        # Process results
        match_stats = next(matches_result)
        participant_stats = next(participants_result)

        stats = {
            'matches': int(match_stats.total_matches),
            'participants': int(participant_stats.total_participants),
            'avg_players_per_match': round(participant_stats.total_participants / max(match_stats.total_matches, 1), 1),
            'versions_covered': int(match_stats.versions_covered),
            'avg_level': round(float(participant_stats.avg_level), 1) if participant_stats.avg_level else 0,
            'avg_placement': round(float(participant_stats.avg_placement), 1) if participant_stats.avg_placement else 0,
            'earliest_match': match_stats.earliest_match.isoformat() if match_stats.earliest_match else None,
            'latest_match': match_stats.latest_match.isoformat() if match_stats.latest_match else None,
            'last_updated': match_stats.latest_match.isoformat() if match_stats.latest_match else None
        }

        logger.info(f"Stats query successful: {stats['matches']} matches, {stats['participants']} participants")
        return json_response(stats)

    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return error_response(f"Failed to get statistics: {str(e)}")