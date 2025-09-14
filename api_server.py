#!/usr/bin/env python3
"""
TFT Analytics REST API Server

Production-ready Flask API server that wraps the BigQuery-based TFT analytics system.
Designed to serve the Firebase webapp with comprehensive TFT composition analysis.

Endpoints:
- GET /api/health - Health check
- GET /api/stats - System statistics  
- POST /api/query - Execute TFT queries
- GET /api/clusters - Get clustering analysis
- GET /api/clusters/{id} - Get specific cluster details
"""

import os
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

from flask import Flask, request, jsonify, g
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix

# Import your existing TFT analytics modules
try:
    from querying import TFTQuery, test_connection as test_query_connection
    from clustering import run_clustering_analysis, TFTClusteringEngine, test_connection as test_clustering_connection
    from bigquery_operations import test_bigquery_connection, get_bigquery_stats
except ImportError as e:
    logging.error(f"Failed to import TFT analytics modules: {e}")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configure CORS for Firebase webapp
CORS(app, origins=[
    "https://*.firebaseapp.com",
    "https://*.web.app", 
    "http://localhost:3000",  # Development
    "http://localhost:5173",  # Vite dev server
])

# Handle proxy headers if behind nginx
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Global cache for expensive operations
_cache = {
    'clustering_results': None,
    'clustering_timestamp': None,
    'system_stats': None,
    'stats_timestamp': None
}

# Cache TTL in seconds
CLUSTERING_CACHE_TTL = 3600  # 1 hour
STATS_CACHE_TTL = 300  # 5 minutes


@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {e}")
    logger.error(traceback.format_exc())
    
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'An unexpected error occurred. Please try again later.',
        'timestamp': datetime.utcnow().isoformat()
    }), 500


@app.errorhandler(400)
def handle_bad_request(e):
    """Handle bad request errors."""
    return jsonify({
        'success': False,
        'error': 'Bad request',
        'message': 'Invalid request format or parameters',
        'timestamp': datetime.utcnow().isoformat()
    }), 400


@app.errorhandler(404) 
def handle_not_found(e):
    """Handle not found errors."""
    return jsonify({
        'success': False,
        'error': 'Not found',
        'message': 'The requested resource was not found',
        'timestamp': datetime.utcnow().isoformat()
    }), 404


def is_cache_valid(timestamp: Optional[datetime], ttl: int) -> bool:
    """Check if cached data is still valid."""
    if timestamp is None:
        return False
    return datetime.utcnow() - timestamp < timedelta(seconds=ttl)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        # Test BigQuery connection
        bigquery_status = test_bigquery_connection()
        
        # Test query system
        query_status = test_query_connection()
        
        # Test clustering system  
        clustering_status = test_clustering_connection()
        
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'services': {
                'bigquery': {
                    'status': 'up' if bigquery_status.get('success', False) else 'down',
                    'message': bigquery_status.get('message', 'Unknown')
                },
                'querying': {
                    'status': 'up' if query_status.get('success', False) else 'down', 
                    'message': query_status.get('message', 'Unknown')
                },
                'clustering': {
                    'status': 'up' if clustering_status.get('success', False) else 'down',
                    'message': clustering_status.get('message', 'Unknown')
                }
            },
            'version': '1.0.0',
            'environment': os.getenv('ENVIRONMENT', 'development')
        }
        
        # Overall health based on critical services
        all_services_up = all(
            service['status'] == 'up' 
            for service in health_status['services'].values()
        )
        
        if not all_services_up:
            health_status['status'] = 'degraded'
            return jsonify(health_status), 503
            
        return jsonify(health_status), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 503


@app.route('/api/stats', methods=['GET'])
def get_system_stats():
    """Get system and database statistics."""
    try:
        # Check cache first
        if (is_cache_valid(_cache['stats_timestamp'], STATS_CACHE_TTL) 
            and _cache['system_stats'] is not None):
            logger.info("Returning cached system stats")
            return jsonify(_cache['system_stats'])
        
        # Get fresh stats
        logger.info("Fetching fresh system stats")
        bigquery_stats = get_bigquery_stats()
        
        stats_response = {
            'success': True,
            'timestamp': datetime.utcnow().isoformat(),
            'database': {
                'type': 'BigQuery',
                'status': 'connected' if not bigquery_stats.get('error') else 'error',
                'statistics': bigquery_stats if not bigquery_stats.get('error') else None,
                'error': bigquery_stats.get('error')
            },
            'system': {
                'cache_status': 'active',
                'clustering_cache_age': (
                    int((datetime.utcnow() - _cache['clustering_timestamp']).total_seconds())
                    if _cache['clustering_timestamp'] else None
                ),
                'api_version': '1.0.0'
            }
        }
        
        # Cache the results
        _cache['system_stats'] = stats_response
        _cache['stats_timestamp'] = datetime.utcnow()
        
        return jsonify(stats_response)
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve system statistics',
            'message': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500


@app.route('/api/query', methods=['POST'])
def execute_query():
    """Execute TFT composition queries."""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json'
            }), 400
        
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'Request body is required'
            }), 400
        
        # Parse query parameters
        query_type = data.get('type', 'stats')  # 'stats' or 'participants'
        filters = data.get('filters', {})
        limit = data.get('limit', 1000)
        
        # Validate limit
        if limit > 10000:
            limit = 10000
        elif limit < 1:
            limit = 100
        
        # Build TFT query
        logger.info(f"Building TFT query with filters: {filters}")
        query = TFTQuery()
        
        # Apply filters from request
        if 'unit' in filters:
            query = query.add_unit(filters['unit'], must_have=filters.get('unit_must_have', True))
        
        if 'trait' in filters:
            min_tier = filters.get('trait_min_tier', 1)
            max_tier = filters.get('trait_max_tier', 4)
            query = query.add_trait(filters['trait'], min_tier=min_tier, max_tier=max_tier)
        
        if 'player_level' in filters:
            min_level = filters['player_level'].get('min', 1)
            max_level = filters['player_level'].get('max', 10)
            query = query.add_player_level(min_level=min_level, max_level=max_level)
        
        if 'placement' in filters:
            min_place = filters['placement'].get('min', 1)
            max_place = filters['placement'].get('max', 8)
            query = query.add_placement_range(min_placement=min_place, max_placement=max_place)
        
        if 'set_number' in filters:
            query = query.add_set_filter(filters['set_number'])
        
        if 'patch' in filters:
            query = query.add_patch_filter(filters['patch'])
        
        if 'custom_sql' in filters:
            query = query.add_custom_filter(filters['custom_sql'], filters.get('custom_params', {}))
        
        # Handle logical operations
        if 'logical_operation' in filters:
            operation = filters['logical_operation']
            other_filters = filters.get('other_query_filters', {})
            
            if operation == 'OR' and other_filters:
                other_query = TFTQuery()
                # Apply other_filters to other_query (simplified for demo)
                if 'unit' in other_filters:
                    other_query = other_query.add_unit(other_filters['unit'])
                query = query.or_(other_query)
                
            elif operation == 'NOT' and other_filters:
                other_query = TFTQuery()
                if 'unit' in other_filters:
                    other_query = other_query.add_unit(other_filters['unit'])
                query = query.not_(other_query)
        
        # Execute query based on type
        start_time = datetime.utcnow()
        
        if query_type == 'stats':
            results = query.get_stats()
            result_type = 'statistics'
        elif query_type == 'participants':
            results = query.get_participants(limit=limit)
            result_type = 'participants'
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid query type. Must be "stats" or "participants"'
            }), 400
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Prepare response
        response = {
            'success': True,
            'timestamp': datetime.utcnow().isoformat(),
            'query': {
                'type': query_type,
                'filters': filters,
                'limit': limit if query_type == 'participants' else None
            },
            'results': {
                'type': result_type,
                'data': results,
                'count': len(results) if isinstance(results, list) else (1 if results else 0)
            },
            'performance': {
                'execution_time_seconds': round(execution_time, 3),
                'cached': False
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return jsonify({
            'success': False,
            'error': 'Query execution failed',
            'message': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500


@app.route('/api/clusters', methods=['GET'])
def get_clusters():
    """Get clustering analysis results."""
    try:
        # Parse query parameters
        cluster_type = request.args.get('type', 'main')  # 'main' or 'sub'
        limit = int(request.args.get('limit', 10))
        filters = {}
        
        # Parse filters from query string
        if request.args.get('set_number'):
            filters['set_number'] = int(request.args.get('set_number'))
        
        if request.args.get('date_from'):
            filters['date_from'] = request.args.get('date_from')
        
        if request.args.get('date_to'):
            filters['date_to'] = request.args.get('date_to')
        
        # Validate limit
        if limit > 50:
            limit = 50
        elif limit < 1:
            limit = 10
        
        # Check cache first (only for unfiltered main clusters)
        cache_key = f"clustering_{cluster_type}_{limit}"
        if (not filters and cluster_type == 'main' and 
            is_cache_valid(_cache['clustering_timestamp'], CLUSTERING_CACHE_TTL) and 
            _cache['clustering_results'] is not None):
            logger.info("Returning cached clustering results")
            cached_response = _cache['clustering_results'].copy()
            cached_response['results']['cached'] = True
            return jsonify(cached_response)
        
        # Run fresh clustering analysis
        logger.info(f"Running clustering analysis with filters: {filters}")
        start_time = datetime.utcnow()
        
        analysis_results = run_clustering_analysis(
            filters=filters,
            limit=5000  # Analyze more data for better clusters
        )
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Get requested cluster data
        if cluster_type == 'main':
            clusters = analysis_results['main_clusters'][:limit]
        else:
            clusters = analysis_results['sub_clusters'][:limit]
        
        response = {
            'success': True,
            'timestamp': datetime.utcnow().isoformat(),
            'query': {
                'type': cluster_type,
                'filters': filters,
                'limit': limit
            },
            'results': {
                'clusters': clusters,
                'count': len(clusters),
                'total_available': len(analysis_results.get(f'{cluster_type}_clusters', [])),
                'statistics': analysis_results['statistics'],
                'cached': False
            },
            'performance': {
                'execution_time_seconds': round(execution_time, 3)
            }
        }
        
        # Cache main cluster results (without filters)
        if not filters and cluster_type == 'main':
            _cache['clustering_results'] = response
            _cache['clustering_timestamp'] = datetime.utcnow()
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error getting clusters: {e}")
        return jsonify({
            'success': False,
            'error': 'Clustering analysis failed',
            'message': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500


@app.route('/api/clusters/<int:cluster_id>', methods=['GET'])
def get_cluster_details(cluster_id):
    """Get detailed information about a specific cluster."""
    try:
        cluster_type = request.args.get('type', 'main')
        
        # For detailed cluster info, we need to run the clustering engine
        logger.info(f"Getting details for {cluster_type} cluster {cluster_id}")
        
        engine = TFTClusteringEngine()
        engine.load_compositions_from_bigquery(limit=3000)
        engine.create_sub_clusters()
        engine.create_main_clusters()
        
        # Get cluster details
        details = engine.get_cluster_details(cluster_id, cluster_type)
        
        if details is None:
            return jsonify({
                'success': False,
                'error': 'Cluster not found',
                'message': f'{cluster_type.title()} cluster {cluster_id} does not exist',
                'timestamp': datetime.utcnow().isoformat()
            }), 404
        
        response = {
            'success': True,
            'timestamp': datetime.utcnow().isoformat(),
            'query': {
                'cluster_id': cluster_id,
                'type': cluster_type
            },
            'results': {
                'cluster': details
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error getting cluster details: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve cluster details',
            'message': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500


@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear API cache (admin endpoint)."""
    try:
        # Simple authentication check (you should implement proper auth)
        api_key = request.headers.get('X-API-Key')
        expected_key = os.getenv('ADMIN_API_KEY', 'admin-key-change-in-production')
        
        if api_key != expected_key:
            return jsonify({
                'success': False,
                'error': 'Unauthorized',
                'message': 'Invalid API key'
            }), 401
        
        # Clear all cache
        global _cache
        _cache = {
            'clustering_results': None,
            'clustering_timestamp': None,
            'system_stats': None,
            'stats_timestamp': None
        }
        
        logger.info("API cache cleared")
        
        return jsonify({
            'success': True,
            'message': 'Cache cleared successfully',
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to clear cache',
            'message': str(e)
        }), 500


@app.route('/api/info', methods=['GET'])
def get_api_info():
    """Get API information and documentation."""
    return jsonify({
        'name': 'TFT Analytics API',
        'version': '1.0.0',
        'description': 'REST API for TFT composition analysis and clustering',
        'endpoints': {
            'GET /api/health': 'Health check and service status',
            'GET /api/stats': 'System and database statistics',
            'POST /api/query': 'Execute TFT composition queries',
            'GET /api/clusters': 'Get clustering analysis results',
            'GET /api/clusters/{id}': 'Get specific cluster details',
            'GET /api/info': 'This endpoint',
            'POST /api/cache/clear': 'Clear API cache (requires API key)'
        },
        'documentation': {
            'query_filters': {
                'unit': 'Filter by unit name (e.g., "Jinx")',
                'trait': 'Filter by trait name (e.g., "Sniper")', 
                'player_level': 'Filter by player level range {min: 8, max: 10}',
                'placement': 'Filter by placement range {min: 1, max: 4}',
                'set_number': 'Filter by TFT set number (e.g., 14)',
                'patch': 'Filter by patch version (e.g., "14.23")',
                'custom_sql': 'Custom SQL WHERE condition'
            },
            'examples': {
                'jinx_query': {
                    'type': 'stats',
                    'filters': {'unit': 'Jinx'}
                },
                'high_level_sniper': {
                    'type': 'stats', 
                    'filters': {
                        'trait': 'Sniper',
                        'trait_min_tier': 2,
                        'player_level': {'min': 8}
                    }
                }
            }
        },
        'timestamp': datetime.utcnow().isoformat()
    })


if __name__ == '__main__':
    # Configuration
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 8080))
    DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting TFT Analytics API Server")
    logger.info(f"Host: {HOST}, Port: {PORT}, Debug: {DEBUG}")
    
    # Start the Flask app
    app.run(
        host=HOST,
        port=PORT,
        debug=DEBUG,
        threaded=True
    )