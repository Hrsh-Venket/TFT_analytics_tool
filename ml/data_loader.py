"""
BigQuery data loader for TFT ML pipeline.
Fetches match data and converts to match dictionaries.
"""

from google.cloud import bigquery
from typing import List, Dict, Any, Optional


def load_matches_from_bigquery(
    project_id: Optional[str] = None,
    dataset_id: str = 'tft_analytics',
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load match data from BigQuery.

    Args:
        project_id: GCP project ID (auto-detected if None)
        dataset_id: BigQuery dataset ID
        limit: Optional limit on number of matches

    Returns:
        List of match dictionaries, each with 'match_id' and 'participants'
    """
    client = bigquery.Client(project=project_id)
    project_id = project_id or client.project

    # Build query
    if limit:
        match_limit_query = f"""
        WITH sampled_matches AS (
            SELECT DISTINCT match_id
            FROM `{project_id}.{dataset_id}.match_participants`
            ORDER BY RAND()
            LIMIT {limit}
        )
        """
        join_clause = "INNER JOIN sampled_matches sm ON p.match_id = sm.match_id"
    else:
        match_limit_query = ""
        join_clause = ""

    query = f"""
    {match_limit_query}
    SELECT
        p.match_id,
        p.placement,
        p.level,
        p.units,
        p.traits
    FROM `{project_id}.{dataset_id}.match_participants` p
    {join_clause}
    ORDER BY p.match_id, p.placement
    """

    print(f"Loading matches from BigQuery...")
    print(f"Project: {project_id}, Dataset: {dataset_id}")
    if limit:
        print(f"Limit: {limit} matches")

    # Execute query
    query_job = client.query(query)
    results = query_job.result()

    # Group by match_id
    matches_dict = {}
    for row in results:
        match_id = row.match_id

        if match_id not in matches_dict:
            matches_dict[match_id] = {
                'match_id': match_id,
                'participants': []
            }

        # Convert units
        units = []
        if row.units:
            for unit in row.units:
                units.append({
                    'name': unit['name'],
                    'tier': unit['tier'],
                    'items': list(unit['item_names']) if unit.get('item_names') else []
                })

        # Convert traits
        traits = []
        if row.traits:
            for trait in row.traits:
                # Only include active traits (style > 0)
                if trait.get('style', 0) > 0:
                    traits.append({
                        'name': trait['name'],
                        'tier_current': trait['tier_current']
                    })

        # Add participant
        participant = {
            'placement': row.placement,
            'level': row.level,
            'units': units,
            'traits': traits
        }

        matches_dict[match_id]['participants'].append(participant)

    # Convert to list and filter to only complete matches (8 players)
    matches = []
    for match in matches_dict.values():
        if len(match['participants']) == 8:
            matches.append(match)
        else:
            print(f"Warning: Match {match['match_id']} has {len(match['participants'])} participants, skipping")

    print(f"✓ Loaded {len(matches)} complete matches")
    return matches
