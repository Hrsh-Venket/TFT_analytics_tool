"""
Cloud Function: GET /api/clusters/{id}
Returns detailed information about a specific cluster
"""

import sys
import os

from utils import handle_cors_preflight, json_response, error_response, get_bigquery_client
from bigquery_operations import BigQueryDataImporter
from google.cloud import bigquery
import logging
from urllib.parse import parse_qs

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cluster_details(request):
    """
    Cloud Function entry point for /api/clusters/{id}
    Returns detailed cluster information including compositions and sub-clusters
    """

    # Handle CORS preflight
    cors_response = handle_cors_preflight(request)
    if cors_response:
        return cors_response

    try:
        # Extract cluster ID from query parameters or path
        cluster_id = None

        # Try query parameters first
        if hasattr(request, 'args') and 'id' in request.args:
            cluster_id = request.args.get('id')
        elif hasattr(request, 'args') and 'cluster_id' in request.args:
            cluster_id = request.args.get('cluster_id')
        else:
            # Parse from URL path if available
            path_parts = request.path.split('/')
            if len(path_parts) >= 3 and path_parts[-2] == 'clusters':
                cluster_id = path_parts[-1]

        if not cluster_id:
            return error_response("Missing cluster ID parameter")

        try:
            cluster_id = int(cluster_id)
        except ValueError:
            return error_response("Invalid cluster ID format")

        # Initialize BigQuery connection
        logger.info(f"Getting details for cluster {cluster_id}")
        importer = BigQueryDataImporter()
        client = get_bigquery_client()

        # Get main cluster details
        main_cluster_query = f"""
        SELECT
            id,
            size,
            avg_placement,
            winrate,
            top4_rate,
            common_carries,
            top_units_display,
            analysis_date,
            sub_cluster_ids
        FROM `{importer.project_id}.{importer.dataset_id}.main_clusters`
        WHERE id = @cluster_id
        """

        job_config = client.query_job_config()
        job_config.query_parameters = [
            bigquery.ScalarQueryParameter("cluster_id", "INT64", cluster_id)
        ]

        main_result = client.query(main_cluster_query, job_config=job_config).result()
        main_cluster = next(main_result, None)

        if not main_cluster:
            return error_response(f"Cluster {cluster_id} not found", 404)

        # Get sub-cluster details if available
        sub_clusters = []
        if main_cluster.sub_cluster_ids:
            sub_cluster_ids = main_cluster.sub_cluster_ids
            if sub_cluster_ids:
                sub_cluster_query = f"""
                SELECT
                    id,
                    carry_set,
                    size,
                    avg_placement,
                    winrate,
                    top4_rate
                FROM `{importer.project_id}.{importer.dataset_id}.sub_clusters`
                WHERE id IN UNNEST(@sub_cluster_ids)
                ORDER BY size DESC
                """

                sub_job_config = client.query_job_config()
                sub_job_config.query_parameters = [
                    bigquery.ArrayQueryParameter("sub_cluster_ids", "INT64", sub_cluster_ids)
                ]

                sub_results = client.query(sub_cluster_query, job_config=sub_job_config).result()

                for row in sub_results:
                    sub_cluster = {
                        'id': int(row.id),
                        'carry_set': list(row.carry_set) if row.carry_set else [],
                        'size': int(row.size),
                        'avg_placement': round(float(row.avg_placement), 2) if row.avg_placement else 0,
                        'winrate': round(float(row.winrate), 1) if row.winrate else 0,
                        'top4_rate': round(float(row.top4_rate), 1) if row.top4_rate else 0
                    }
                    sub_clusters.append(sub_cluster)

        # Format main cluster response
        cluster_details = {
            'id': int(main_cluster.id),
            'name': main_cluster.top_units_display or f"Cluster {main_cluster.id}",
            'size': int(main_cluster.size),
            'avg_placement': round(float(main_cluster.avg_placement), 2) if main_cluster.avg_placement else 0,
            'winrate': round(float(main_cluster.winrate), 1) if main_cluster.winrate else 0,
            'top4_rate': round(float(main_cluster.top4_rate), 1) if main_cluster.top4_rate else 0,
            'carries': main_cluster.common_carries if main_cluster.common_carries else [],
            'analysis_date': main_cluster.analysis_date.isoformat() if main_cluster.analysis_date else None,
            'sub_clusters': sub_clusters,
            'sub_cluster_count': len(sub_clusters)
        }

        logger.info(f"Found cluster {cluster_id} with {len(sub_clusters)} sub-clusters")
        return json_response(cluster_details)

    except Exception as e:
        logger.error(f"Error getting cluster details: {str(e)}")
        return error_response(f"Failed to get cluster details: {str(e)}")