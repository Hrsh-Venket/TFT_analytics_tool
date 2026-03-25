"""
Cloud Function: GET /api/cluster-details
Returns detailed information about a specific cluster
"""

import logging

from utils import handle_cors_preflight, json_response, error_response, get_bigquery_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_ID = 'tft_analytics'


def get_cluster_details(request):
    """
    Cloud Function entry point for /api/cluster-details
    Returns detailed data for a specific cluster by ID
    """
    cors_response = handle_cors_preflight(request)
    if cors_response:
        return cors_response

    try:
        cluster_id = request.args.get('id')
        if not cluster_id:
            return error_response("Missing 'id' parameter")

        client = get_bigquery_client()
        project_id = client.project
        main_table = f"{project_id}.{DATASET_ID}.main_clusters"
        sub_table = f"{project_id}.{DATASET_ID}.sub_clusters"

        # Get main cluster info
        main_query = f"""
        SELECT *
        FROM `{main_table}`
        WHERE id = @cluster_id
        """

        from google.cloud import bigquery as bq
        job_config = bq.QueryJobConfig(
            query_parameters=[
                bq.ScalarQueryParameter("cluster_id", "INTEGER", int(cluster_id))
            ]
        )

        result = client.query(main_query, job_config=job_config).result()
        rows = list(result)

        if not rows:
            return error_response(f"Cluster {cluster_id} not found", 404)

        row = rows[0]
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

        # Get sub-clusters for this main cluster
        sub_query = f"""
        SELECT *
        FROM `{sub_table}`
        WHERE main_cluster_id = @cluster_id
        ORDER BY size DESC
        """

        sub_result = client.query(sub_query, job_config=job_config).result()
        sub_clusters = []
        for sub_row in sub_result:
            sub_clusters.append({
                'id': int(sub_row.id),
                'size': int(sub_row.size),
                'avg_placement': round(float(sub_row.avg_placement), 2) if sub_row.avg_placement else 0,
                'winrate': round(float(sub_row.winrate), 1) if sub_row.winrate else 0,
                'carries': sub_row.carries if hasattr(sub_row, 'carries') and sub_row.carries else [],
            })

        cluster['sub_clusters'] = sub_clusters

        logger.info(f"Cluster {cluster_id}: {len(sub_clusters)} sub-clusters")
        return json_response(cluster)

    except Exception as e:
        if 'Not found' in str(e):
            return json_response({
                'error': 'Clusters table not found. Run clustering analysis first.',
                'status': 'no_clusters'
            })
        logger.error(f"Error getting cluster details: {e}")
        return error_response(f"Failed to get cluster details: {str(e)}")
