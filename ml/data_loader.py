"""
BigQuery data loader for TFT ML pipeline.
Fetches match data and converts to match dictionaries.

Supports streaming/batch processing to avoid memory issues with large datasets.
"""

from google.cloud import bigquery
from typing import List, Dict, Any, Optional, Iterator
import time


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

    # Only use ORDER BY when limiting (for deterministic sampling)
    # Skip ORDER BY for full data loads (much faster, data will be shuffled anyway)
    order_by_clause = "ORDER BY p.match_id, p.placement" if limit else ""

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
    {order_by_clause}
    """

    print(f"Loading matches from BigQuery...")
    print(f"Project: {project_id}, Dataset: {dataset_id}")
    if limit:
        print(f"Limit: {limit} matches")
    else:
        print(f"Loading ALL data (skipping ORDER BY for performance)")

    # Execute query
    query_job = client.query(query)
    results = query_job.result()

    print(f"Query complete. Processing rows...")

    # Group by match_id
    matches_dict = {}
    row_count = 0
    for row in results:
        row_count += 1
        if row_count % 10000 == 0:
            print(f"  Processed {row_count:,} rows, {len(matches_dict):,} matches so far...")

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


def stream_matches_from_bigquery(
    project_id: Optional[str] = None,
    dataset_id: str = 'tft_analytics',
    limit: Optional[int] = None,
    batch_size: int = 500
) -> Iterator[List[Dict[str, Any]]]:
    """
    Stream match data from BigQuery in batches to avoid memory issues.

    Yields batches of complete matches (8 players each).
    This is memory-efficient for large datasets.

    Args:
        project_id: GCP project ID (auto-detected if None)
        dataset_id: BigQuery dataset ID
        limit: Optional limit on number of matches
        batch_size: Number of matches per batch

    Yields:
        Batches of match dictionaries (list of dicts)
    """
    client = bigquery.Client(project=project_id)
    project_id = project_id or client.project

    # Build query (same as load_matches_from_bigquery)
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

    # Skip ORDER BY for full data loads (faster)
    order_by_clause = "ORDER BY p.match_id, p.placement" if limit else ""

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
    {order_by_clause}
    """

    print(f"Streaming matches from BigQuery (batch_size={batch_size})...")
    print(f"Project: {project_id}, Dataset: {dataset_id}")
    if limit:
        print(f"Limit: {limit} matches")
    else:
        print(f"Streaming ALL data (skipping ORDER BY for performance)")

    # Execute query
    print(f"Executing BigQuery query...")
    start_time = time.time()
    query_job = client.query(query)
    results = query_job.result()
    query_time = time.time() - start_time
    print(f"✓ Query complete in {query_time:.1f}s. Streaming rows...")

    # Stream and group by match_id
    matches_dict = {}
    row_count = 0
    matches_yielded = 0
    incomplete_count = 0
    last_log_time = time.time()

    for row in results:
        row_count += 1

        # Log progress every 3 seconds
        current_time = time.time()
        if current_time - last_log_time >= 3:
            print(f"  Streamed {row_count:,} rows → {matches_yielded:,} complete matches (batch buffer: {len(matches_dict)})")
            last_log_time = current_time

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

        # Check if we have complete matches to yield
        complete_matches = []
        incomplete_match_ids = []

        for mid, match in list(matches_dict.items()):
            if len(match['participants']) == 8:
                complete_matches.append(match)
                del matches_dict[mid]
            elif len(match['participants']) > 8:
                # This shouldn't happen, but handle it
                incomplete_count += 1
                del matches_dict[mid]

        # Yield batch when we have enough complete matches
        if len(complete_matches) >= batch_size:
            matches_yielded += len(complete_matches)
            yield complete_matches
            complete_matches = []

    # Yield any remaining complete matches
    final_batch = []
    for match in matches_dict.values():
        if len(match['participants']) == 8:
            final_batch.append(match)
        else:
            incomplete_count += 1

    if final_batch:
        matches_yielded += len(final_batch)
        yield final_batch

    total_time = time.time() - start_time
    print(f"✓ Streamed {row_count:,} rows → {matches_yielded:,} complete matches in {total_time:.1f}s")
    if incomplete_count > 0:
        print(f"  Skipped {incomplete_count} incomplete matches")
