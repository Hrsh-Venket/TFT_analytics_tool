"""
Shared utilities for TFT Analytics Cloud Functions.
Provides CORS handling, JSON responses, and BigQuery client.
"""

import json
from google.cloud import bigquery

# Singleton BigQuery client
_bq_client = None

CORS_HEADERS = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
}


def get_bigquery_client():
    """Get or create a singleton BigQuery client."""
    global _bq_client
    if _bq_client is None:
        _bq_client = bigquery.Client()
    return _bq_client


def handle_cors_preflight(request):
    """Handle CORS preflight OPTIONS request. Returns response or None."""
    if request.method == 'OPTIONS':
        return ('', 204, CORS_HEADERS)
    return None


def json_response(data, status_code=200):
    """Return a JSON response with CORS headers."""
    return (
        json.dumps(data, default=str),
        status_code,
        {**CORS_HEADERS, 'Content-Type': 'application/json'},
    )


def error_response(message, status_code=400):
    """Return a JSON error response with CORS headers."""
    return json_response({'error': message, 'success': False}, status_code)
