"""
Cloud Function: POST /api/query
Executes TFT queries using the existing querying.py system
"""

import sys
import os
import re
import ast
import json
import logging
from typing import Dict, Any, Optional, Tuple

from utils import handle_cors_preflight, json_response, error_response, get_bigquery_client
from querying import TFTQuery

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_tft_query(query_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse a TFTQuery string into executable method calls.

    Args:
        query_text: String containing TFTQuery method chain

    Returns:
        Parsed result or None if invalid
    """
    try:
        # Normalize multi-line queries - join lines and clean whitespace
        normalized = ' '.join(query_text.split())

        # Replace SimpleTFTQuery with TFTQuery for compatibility
        normalized = normalized.replace('SimpleTFTQuery()', 'TFTQuery()')

        # Check if it's a valid TFTQuery
        if not ('TFTQuery()' in normalized):
            return None

        # Extract the method chain
        pattern = r'TFTQuery\(\)((?:\.\w+\([^)]*\))*)'
        match = re.match(pattern, normalized)

        if not match:
            return None

        method_chain = match.group(1)

        # Parse individual method calls
        method_pattern = r'\.(\w+)\(([^)]*)\)'
        methods = re.findall(method_pattern, method_chain)

        # Build and execute the query
        query = TFTQuery()

        for method_name, args_str in methods:
            # Parse arguments safely
            args = []
            kwargs = {}

            if args_str.strip():
                # Split by comma, but respect quotes
                arg_parts = []
                current = []
                in_quotes = False
                quote_char = None

                for char in args_str + ',':
                    if char in ('"', "'") and not in_quotes:
                        in_quotes = True
                        quote_char = char
                        current.append(char)
                    elif char == quote_char and in_quotes:
                        in_quotes = False
                        quote_char = None
                        current.append(char)
                    elif char == ',' and not in_quotes:
                        if current:
                            arg_parts.append(''.join(current).strip())
                            current = []
                    else:
                        current.append(char)

                # Parse each argument
                for arg in arg_parts:
                    if not arg:
                        continue

                    if '=' in arg and not (arg.startswith('"') or arg.startswith("'")):
                        # Keyword argument
                        key, value = arg.split('=', 1)
                        key = key.strip()
                        value = value.strip()

                        # Parse value
                        try:
                            # Try to evaluate as literal
                            kwargs[key] = ast.literal_eval(value)
                        except:
                            # Treat as string
                            kwargs[key] = value.strip('"\'')
                    else:
                        # Positional argument
                        try:
                            # Try to evaluate as literal
                            args.append(ast.literal_eval(arg))
                        except:
                            # Treat as string
                            args.append(arg.strip('"\''))

            # Apply method to query
            if hasattr(query, method_name):
                method = getattr(query, method_name)

                # Handle special methods
                if method_name == 'or_':
                    # or_ expects other TFTQuery objects - for now skip
                    continue
                elif method_name == 'not_':
                    # not_ expects other TFTQuery object - for now skip
                    continue
                elif method_name == 'xor':
                    # xor expects other TFTQuery object - for now skip
                    continue
                elif method_name == 'get_stats':
                    # Terminal method - execute and return
                    return method(*args, **kwargs)
                elif method_name == 'get_participants':
                    # Deprecated - convert to get_stats
                    return query.get_stats()
                elif method_name == 'execute':
                    # Terminal method - don't expose raw data
                    return query.get_stats()
                else:
                    # Chain method
                    query = method(*args, **kwargs)
            else:
                logger.warning(f"Unknown method: {method_name}")

        # If no terminal method was called, call get_stats by default
        return query.get_stats()

    except Exception as e:
        logger.error(f"Query parsing failed: {str(e)}")
        return None

def format_query_result(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Format query result for display with pretty table formatting.

    Args:
        result: Query result dictionary or None

    Returns:
        Formatted response dictionary
    """
    if result is None:
        return {
            'success': False,
            'message': 'No data found matching the query criteria',
            'formatted_output': 'Not found in the data'
        }

    # Format as a pretty table
    headers = ['Metric', 'Value']
    rows = []

    if 'play_count' in result:
        rows.append(['Play Count', str(result['play_count'])])
    if 'avg_placement' in result:
        rows.append(['Average Placement', f"{result['avg_placement']:.2f}"])
    if 'winrate' in result:
        rows.append(['Win Rate', f"{result['winrate']:.2f}%"])
    if 'top4_rate' in result:
        rows.append(['Top 4 Rate', f"{result['top4_rate']:.2f}%"])

    # Create simple ASCII table
    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    table_lines = []

    # Header
    header_line = ' | '.join(h.ljust(w) for h, w in zip(headers, col_widths))
    table_lines.append(header_line)
    table_lines.append('-' * len(header_line))

    # Rows
    for row in rows:
        row_line = ' | '.join(val.ljust(w) for val, w in zip(row, col_widths))
        table_lines.append(row_line)

    formatted_output = '\n'.join(table_lines)

    return {
        'success': True,
        'data': result,
        'formatted_output': formatted_output,
        'table': {
            'headers': headers,
            'rows': rows
        }
    }

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

        if not query_text:
            return error_response("Missing 'query' parameter")

        logger.info(f"Executing query: {query_text[:100]}...")

        # Parse and execute the query using proper parser
        result = parse_tft_query(query_text)

        # Format the result
        formatted_response = format_query_result(result)

        # Add original query to response
        formatted_response['query'] = query_text

        if formatted_response['success']:
            logger.info(f"Query executed successfully")
            return json_response(formatted_response)
        else:
            logger.info(f"Query returned no results")
            return json_response(formatted_response)

    except Exception as e:
        logger.error(f"Error processing query request: {str(e)}")
        return error_response(f"Failed to process query: {str(e)}")