"""
Common utilities for Cloud Functions
Shared BigQuery operations and CORS handling
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional

def cors_headers(origin: str = '*') -> Dict[str, str]:
    """Standard CORS headers for Firebase frontend"""
    return {
        'Access-Control-Allow-Origin': origin,
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization, hx-current-url, hx-target, hx-trigger, hx-request',
        'Access-Control-Max-Age': '3600'
    }

def handle_cors_preflight(request) -> Optional[tuple]:
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

def format_datetime(dt: datetime) -> str:
    """Format datetime for JSON serialization"""
    return dt.isoformat() if dt else None