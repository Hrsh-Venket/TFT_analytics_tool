"""
Cloud Function: GET /api/stats - Minimal version without pandas
"""

import json
import logging
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_stats(request):
    """Minimal stats function without pandas dependencies"""

    # Handle CORS
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    headers = {'Access-Control-Allow-Origin': '*'}

    try:
        # Initialize BigQuery client
        client = bigquery.Client()
        project_id = client.project
        dataset_id = 'tft_analytics'

        # Simple count query
        query = f"""
        SELECT
            COUNT(*) as total_matches
        FROM `{project_id}.{dataset_id}.matches`
        LIMIT 1
        """

        result = client.query(query).result()
        row = next(result)

        stats = {
            'matches': int(row.total_matches),
            'status': 'success',
            'message': 'Minimal stats function working'
        }

        logger.info(f"Stats query successful: {stats['matches']} matches")

        response = json.dumps(stats)
        return (response, 200, headers)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        error_response = json.dumps({'error': str(e), 'status': 'failed'})
        return (error_response, 500, headers)