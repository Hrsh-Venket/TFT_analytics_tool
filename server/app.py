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

import requests
from flask import Flask, request, jsonify, send_from_directory
from psycopg2.extras import RealDictCursor

from tft_analytics.postgres import get_connection, put_connection, ensure_tables
from tft_analytics.query import TFTQuery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Static files directory
STATIC_DIR = os.environ.get(
    'STATIC_DIR',
    os.path.join(os.path.dirname(__file__), '..', 'static')
)

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='')

try:
    ensure_tables()
except Exception as e:
    logger.warning(f"Could not ensure tables on startup: {e}")


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
        error_str = str(e)
        if 'does not exist' in error_str or 'relation' in error_str:
            return jsonify({
                'matches': 0,
                'participants': 0,
                'avg_placement': 0,
                'avg_level': 0,
                'avg_players_per_match': 8,
                'last_updated': None,
                'status': 'no_data',
            })
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': error_str, 'success': False}), 500
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


# ---------------------------------------------------------------------------
# POST /api/nl-to-query  — natural language → TFTQuery via OpenRouter
# ---------------------------------------------------------------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

NL_SYSTEM_PROMPT = """You translate a user's natural-language question about Teamfight Tactics match data into a single TFTQuery method-chain expression.

TFTQuery is a Python builder. Always start with `TFTQuery()` and always end with `.get_stats()`. Chain filter methods in between. Output one line of code, no imports, no comments, no backticks.

Filter methods (all return self, so they chain):
- add_unit(unit_id: str, must_have: bool = True)
- add_unit_count(unit_id: str, count: int)
- add_unit_star_level(unit_id: str, min_star: int = 1, max_star: int = 3)
- add_item_on_unit(unit_id: str, item_id: str)
- add_unit_item_count(unit_id: str, min_count: int = 0, max_count: int = 3)
- add_trait(trait_name: str, min_tier: int = 1, max_tier: int = 4)
- add_player_level(min_level: int = 1, max_level: int = 10)
- add_last_round(min_round: int = 1, max_round: int = 50)
- add_placement_range(min_placement: int = 1, max_placement: int = 8)
- add_set_filter(set_number: int)
- add_patch_filter(patch_version: str)
- or_(other: TFTQuery)   # logical OR with a sibling TFTQuery
- not_(other: TFTQuery)  # exclude matches of sibling TFTQuery
- xor(other: TFTQuery)   # exclusive OR with a sibling TFTQuery

Unit, trait, and item identifiers use PascalCase/snake_case exactly as shown in the examples (e.g. 'Jinx', 'Kaisa', 'Aatrox', 'Anima', 'Sniper', 'Bastion', 'Last_Whisper'). Do not prefix them with the set number.

Reference examples — the exact syntax and style you must follow:
- TFTQuery().add_unit('Jinx').get_stats()
- TFTQuery().add_trait('Anima', min_tier=2).get_stats()
- TFTQuery().add_item_on_unit('Jinx', 'Last_Whisper').get_stats()
- TFTQuery().add_unit_star_level('Jinx', min_star=3).get_stats()
- TFTQuery().add_unit('Jinx').add_trait('Sniper', min_tier=2).add_player_level(min_level=9).get_stats()
- TFTQuery().add_trait('Bastion', min_tier=2).not_(TFTQuery().add_unit('Aatrox')).get_stats()
- TFTQuery().add_unit('Kaisa').or_(TFTQuery().add_unit('Jinx')).get_stats()

Respond with ONLY a JSON object, no prose, no markdown fences. Both fields are REQUIRED and neither may be empty:
{"query": "<TFTQuery()...get_stats()>", "description": "<one sentence rephrasing the user's request>"}

The `description` must be a non-empty plain-English sentence, present tense, that mirrors every filter in the query so the user can verify their intent was understood.

Worked example — user asks: "how does Jinx do at 3 stars"
{"query": "TFTQuery().add_unit_star_level('Jinx', min_star=3).get_stats()", "description": "Stats for games containing a 3-star Jinx."}

Worked example — user asks: "Kaisa or Jinx winrate"
{"query": "TFTQuery().add_unit('Kaisa').or_(TFTQuery().add_unit('Jinx')).get_stats()", "description": "Stats for games containing either Kaisa or Jinx."}
"""


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from a model response."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError(f"Could not parse JSON from model response: {text[:200]}")


@app.route('/api/nl-to-query', methods=['POST'])
def api_nl_to_query():
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        return jsonify({
            'error': 'OPENROUTER_API_KEY is not set on the server',
            'success': False,
        }), 500

    try:
        request_data = request.get_json(force=True) if request.is_json else {}
        nl_query = (request_data or {}).get('query', '').strip()
        if not nl_query:
            return jsonify({'error': "Missing 'query' parameter", 'success': False}), 400

        model = os.environ.get('OPENROUTER_MODEL') or DEFAULT_OPENROUTER_MODEL
        payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': NL_SYSTEM_PROMPT},
                {'role': 'user', 'content': nl_query},
            ],
            'response_format': {'type': 'json_object'},
            'temperature': 0,
        }
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }

        logger.info(f"NL→Query via {model}: {nl_query[:120]}")
        resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=60)
        if not resp.ok:
            logger.error(f"OpenRouter error {resp.status_code}: {resp.text[:500]}")
            return jsonify({
                'error': f"OpenRouter returned HTTP {resp.status_code}",
                'detail': resp.text[:500],
                'success': False,
            }), 502

        content = resp.json()['choices'][0]['message']['content']
        logger.info(f"NL→Query raw model output: {content[:500]}")
        parsed = _extract_json_object(content)
        query_text = (parsed.get('query') or '').strip()
        description = (parsed.get('description') or '').strip()
        # Free models occasionally corrupt the `description` key. If it's
        # missing but a second string field exists, use that as the description.
        if not description:
            for key, val in parsed.items():
                if key == 'query' or not isinstance(val, str):
                    continue
                val = val.strip()
                if val and 'TFTQuery()' not in val:
                    description = val
                    break

        if not query_text or 'TFTQuery()' not in query_text:
            return jsonify({
                'error': 'Model did not return a valid TFTQuery expression',
                'raw': content,
                'success': False,
            }), 502

        return jsonify({
            'success': True,
            'query': query_text,
            'description': description or '(model returned no description)',
            'model': model,
        })
    except Exception as e:
        logger.error(f"NL→Query failed: {e}")
        return jsonify({'error': str(e), 'success': False}), 500


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
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)
