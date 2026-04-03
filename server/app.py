"""
TFT Analytics Flask Server

Self-hosted Flask app serving the TFT Analytics API and frontend.
Serves the static frontend and provides the 4 API endpoints.
"""

import os
import re
import ast
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from psycopg2.extras import RealDictCursor

from tft_analytics.postgres import get_connection, put_connection
from tft_analytics.query import TFTQuery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Static files directory
STATIC_DIR = os.environ.get(
    'STATIC_DIR',
    os.path.join(os.path.dirname(__file__), '..', 'static')
)

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='')


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(STATIC_DIR, path)


# ---------------------------------------------------------------------------
# CORS (needed for dev when frontend runs on a different port)
# ---------------------------------------------------------------------------

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


# ---------------------------------------------------------------------------
# GET /api/stats
# ---------------------------------------------------------------------------

@app.route('/api/stats')
def api_stats():
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    COUNT(DISTINCT match_id) AS matches,
                    COUNT(*)                 AS participants,
                    AVG(placement)           AS avg_placement,
                    AVG(level)               AS avg_level,
                    MAX(game_datetime)       AS last_updated
                FROM match_participants
            """)
            row = cur.fetchone()

        stats = {
            'matches': row['matches'] or 0,
            'participants': row['participants'] or 0,
            'avg_placement': round(float(row['avg_placement']), 2) if row['avg_placement'] else 0,
            'avg_level': round(float(row['avg_level']), 1) if row['avg_level'] else 0,
            'avg_players_per_match': 8,
            'last_updated': row['last_updated'].isoformat() if row['last_updated'] else None,
        }
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e), 'success': False}), 500
    finally:
        put_connection(conn)


# ---------------------------------------------------------------------------
# GET /api/clusters
# ---------------------------------------------------------------------------

@app.route('/api/clusters')
def api_clusters():
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, size, avg_placement, winrate, top4_rate,
                       common_carries, top_units_display, analysis_date
                FROM main_clusters
                ORDER BY size DESC
                LIMIT 50
            """)
            rows = cur.fetchall()

        clusters = []
        for row in rows:
            clusters.append({
                'id': row['id'],
                'name': row['top_units_display'] or f"Cluster {row['id']}",
                'size': row['size'],
                'avg_placement': round(float(row['avg_placement']), 2) if row['avg_placement'] else 0,
                'winrate': round(float(row['winrate']), 1) if row['winrate'] else 0,
                'top4_rate': round(float(row['top4_rate']), 1) if row['top4_rate'] else 0,
                'carries': row['common_carries'] or [],
                'analysis_date': row['analysis_date'].isoformat() if row['analysis_date'] else None,
            })

        return jsonify({'clusters': clusters, 'total_clusters': len(clusters)})

    except Exception as e:
        error_str = str(e)
        if 'does not exist' in error_str or 'relation' in error_str:
            return jsonify({
                'clusters': [],
                'total_clusters': 0,
                'message': 'No clustering analysis available. Run clustering analysis to generate clusters.',
                'status': 'no_clusters',
            })
        logger.error(f"Error getting clusters: {e}")
        return jsonify({'error': error_str, 'success': False}), 500
    finally:
        put_connection(conn)


# ---------------------------------------------------------------------------
# GET /api/cluster-details?id=N
# ---------------------------------------------------------------------------

@app.route('/api/cluster-details')
def api_cluster_details():
    cluster_id = request.args.get('id')
    if not cluster_id:
        return jsonify({'error': "Missing 'id' parameter", 'success': False}), 400

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM main_clusters WHERE id = %s", (int(cluster_id),))
            row = cur.fetchone()

        if not row:
            return jsonify({'error': f"Cluster {cluster_id} not found", 'success': False}), 404

        cluster = {
            'id': row['id'],
            'name': row['top_units_display'] or f"Cluster {row['id']}",
            'size': row['size'],
            'avg_placement': round(float(row['avg_placement']), 2) if row['avg_placement'] else 0,
            'winrate': round(float(row['winrate']), 1) if row['winrate'] else 0,
            'top4_rate': round(float(row['top4_rate']), 1) if row['top4_rate'] else 0,
            'carries': row['common_carries'] or [],
            'analysis_date': row['analysis_date'].isoformat() if row['analysis_date'] else None,
        }

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM sub_clusters WHERE main_cluster_id = %s ORDER BY size DESC",
                (int(cluster_id),)
            )
            sub_rows = cur.fetchall()

        cluster['sub_clusters'] = [
            {
                'id': sr['id'],
                'size': sr['size'],
                'avg_placement': round(float(sr['avg_placement']), 2) if sr['avg_placement'] else 0,
                'winrate': round(float(sr['winrate']), 1) if sr['winrate'] else 0,
                'carries': sr.get('carry_set') or [],
            }
            for sr in sub_rows
        ]

        return jsonify(cluster)

    except Exception as e:
        error_str = str(e)
        if 'does not exist' in error_str or 'relation' in error_str:
            return jsonify({
                'error': 'Clusters table not found. Run clustering analysis first.',
                'status': 'no_clusters',
            })
        logger.error(f"Error getting cluster details: {e}")
        return jsonify({'error': error_str, 'success': False}), 500
    finally:
        put_connection(conn)


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------

def parse_tft_query(query_text: str) -> Optional[Dict[str, Any]]:
    """Parse a TFTQuery method-chain string and execute it."""
    try:
        normalized = ' '.join(query_text.split())
        normalized = normalized.replace('SimpleTFTQuery()', 'TFTQuery()')

        if 'TFTQuery()' not in normalized:
            return None

        pattern = r'TFTQuery\(\)((?:\.\w+\([^)]*\))*)'
        match = re.match(pattern, normalized)
        if not match:
            return None

        method_chain = match.group(1)
        method_pattern = r'\.(\w+)\(([^)]*)\)'
        methods = re.findall(method_pattern, method_chain)

        query = TFTQuery()

        for method_name, args_str in methods:
            args = []
            kwargs = {}

            if args_str.strip():
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

                for arg in arg_parts:
                    if not arg:
                        continue
                    if '=' in arg and not (arg.startswith('"') or arg.startswith("'")):
                        key, value = arg.split('=', 1)
                        try:
                            kwargs[key.strip()] = ast.literal_eval(value.strip())
                        except Exception:
                            kwargs[key.strip()] = value.strip().strip('"\'')
                    else:
                        try:
                            args.append(ast.literal_eval(arg))
                        except Exception:
                            args.append(arg.strip('"\''))

            if not hasattr(query, method_name):
                logger.warning(f"Unknown method: {method_name}")
                continue

            if method_name in ('or_', 'not_', 'xor'):
                continue
            elif method_name == 'get_stats':
                return getattr(query, method_name)(*args, **kwargs)
            elif method_name in ('get_participants', 'execute'):
                return query.get_stats()
            else:
                query = getattr(query, method_name)(*args, **kwargs)

        return query.get_stats()
    except Exception as e:
        logger.error(f"Query parsing failed: {e}")
        return None


def format_query_result(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Format query result for the frontend."""
    if result is None:
        return {
            'success': False,
            'message': 'No data found matching the query criteria',
            'formatted_output': 'Not found in the data',
        }

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

    if rows:
        col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
        lines = [' | '.join(h.ljust(w) for h, w in zip(headers, col_widths))]
        lines.append('-' * len(lines[0]))
        for row in rows:
            lines.append(' | '.join(val.ljust(w) for val, w in zip(row, col_widths)))
        formatted_output = '\n'.join(lines)
    else:
        formatted_output = ''

    return {
        'success': True,
        'data': result,
        'formatted_output': formatted_output,
        'table': {'headers': headers, 'rows': rows},
    }


@app.route('/api/query', methods=['POST'])
def api_query():
    try:
        request_data = request.get_json(force=True) if request.is_json else {}
        if not request_data:
            request_data = request.form.to_dict()

        query_text = request_data.get('query', '').strip()
        if not query_text:
            return jsonify({'error': "Missing 'query' parameter", 'success': False}), 400

        logger.info(f"Executing query: {query_text[:100]}...")

        result = parse_tft_query(query_text)
        formatted = format_query_result(result)
        formatted['query'] = query_text

        return jsonify(formatted)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({'error': f"Failed to process query: {e}", 'success': False}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
