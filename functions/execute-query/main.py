"""
Cloud Function: POST /api/query
Executes TFT queries using the existing querying.py system
"""

import sys
import os

from utils import handle_cors_preflight, json_response, error_response, get_bigquery_client
from querying import TFTQuery
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def execute_query(request):
    """
    Cloud Function entry point for /api/query
    Executes TFT queries and returns formatted results
    """

    # Handle CORS preflight
    cors_response = handle_cors_preflight(request)
    if cors_response:
        return cors_response

    if request.method != 'POST':
        return error_response("Method not allowed", 405)

    try:
        # Parse request data
        request_data = request.get_json(force=True) if request.is_json else {}

        if not request_data:
            # Try to parse form data
            request_data = request.form.to_dict()

        query_text = request_data.get('query', '').strip()
        query_type = request_data.get('type', 'auto')

        if not query_text:
            return error_response("Missing 'query' parameter")

        logger.info(f"Executing query: {query_text[:100]}...")

        # Execute the query using existing TFTQuery system
        try:
            # Simple eval approach for demo - in production you'd want a proper parser
            if query_text.startswith('TFTQuery()') or query_text.startswith('SimpleTFTQuery()'):
                # Replace SimpleTFTQuery with TFTQuery for compatibility
                normalized_query = query_text.replace('SimpleTFTQuery()', 'TFTQuery()')

                # Safe execution environment
                safe_globals = {
                    'TFTQuery': TFTQuery,
                    '__builtins__': {}
                }

                result = eval(normalized_query, safe_globals, {})

                # Determine result type and format appropriately
                if isinstance(result, dict):
                    # Statistics result
                    formatted_result = {
                        'type': 'stats',
                        'data': result,
                        'query': query_text
                    }
                elif isinstance(result, list):
                    # Participants result
                    formatted_result = {
                        'type': 'participants',
                        'data': result,
                        'count': len(result),
                        'query': query_text
                    }
                else:
                    # Unknown result type
                    formatted_result = {
                        'type': 'unknown',
                        'data': str(result),
                        'query': query_text
                    }

                logger.info(f"Query executed successfully, type: {formatted_result['type']}")
                return json_response(formatted_result)

            else:
                return error_response("Invalid query format. Use TFTQuery() or SimpleTFTQuery() syntax.")

        except Exception as query_error:
            logger.error(f"Query execution failed: {str(query_error)}")
            return error_response(f"Query execution failed: {str(query_error)}")

    except Exception as e:
        logger.error(f"Error processing query request: {str(e)}")
        return error_response(f"Failed to process query: {str(e)}")