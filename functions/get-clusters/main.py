"""
Cloud Function: GET /api/clusters
Returns available TFT composition clusters from BigQuery analysis
"""

import sys
import os

from utils import handle_cors_preflight, json_response, error_response, get_bigquery_client
from bigquery_operations import BigQueryDataImporter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_clusters(request):
    """
    Cloud Function entry point for /api/clusters
    Returns list of available clusters with metadata
    """

    # Handle CORS preflight
    cors_response = handle_cors_preflight(request)
    if cors_response:
        return cors_response

    try:
        # Initialize BigQuery connection
        logger.info("Initializing BigQuery connection for clusters")
        importer = BigQueryDataImporter()
        client = get_bigquery_client()

        # Check if clusters table exists, if not return empty list
        try:
            clusters_table = f"{importer.project_id}.{importer.dataset_id}.main_clusters"

            # Query main clusters with metadata
            clusters_query = f"""
            SELECT
                id,
                size,
                avg_placement,
                winrate,
                top4_rate,
                common_carries,
                top_units_display,
                analysis_date
            FROM `{clusters_table}`
            ORDER BY size DESC
            LIMIT 50
            """

            logger.info("Executing clusters query")
            results = client.query(clusters_query).result()

            clusters = []
            for row in results:
                cluster = {
                    'id': int(row.id),
                    'name': row.top_units_display or f"Cluster {row.id}",
                    'size': int(row.size),
                    'avg_placement': round(float(row.avg_placement), 2) if row.avg_placement else 0,
                    'winrate': round(float(row.winrate), 1) if row.winrate else 0,
                    'top4_rate': round(float(row.top4_rate), 1) if row.top4_rate else 0,
                    'carries': row.common_carries if row.common_carries else [],
                    'analysis_date': row.analysis_date.isoformat() if row.analysis_date else None
                }
                clusters.append(cluster)

            logger.info(f"Found {len(clusters)} clusters")
            return json_response({
                'clusters': clusters,
                'total_clusters': len(clusters)
            })

        except Exception as table_error:
            # If clusters table doesn't exist, return empty list with helpful message
            logger.warning(f"Clusters table not found: {str(table_error)}")
            return json_response({
                'clusters': [],
                'total_clusters': 0,
                'message': 'No clustering analysis available. Run clustering analysis to generate clusters.',
                'status': 'no_clusters'
            })

    except Exception as e:
        logger.error(f"Error getting clusters: {str(e)}")
        return error_response(f"Failed to get clusters: {str(e)}")