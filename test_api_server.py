#!/usr/bin/env python3
"""
Test script for the TFT Analytics API Server

Tests all API endpoints to ensure they're working correctly.
"""

import os
import requests
import json
import time
from typing import Dict, Any

# Force test mode for the API server
os.environ['TFT_TEST_MODE'] = 'true'

# API base URL (adjust for your setup)
BASE_URL = "http://localhost:8080"

def test_endpoint(method: str, endpoint: str, data: Dict[str, Any] = None, expected_status: int = 200) -> Dict[str, Any]:
    """Test a single API endpoint."""
    url = f"{BASE_URL}{endpoint}"
    print(f"\n{'='*50}")
    print(f"Testing {method} {endpoint}")
    print(f"{'='*50}")
    
    try:
        start_time = time.time()
        
        if method == 'GET':
            response = requests.get(url, timeout=30)
        elif method == 'POST':
            response = requests.post(url, json=data, timeout=30)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        execution_time = time.time() - start_time
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {execution_time:.3f} seconds")
        
        try:
            response_json = response.json()
            print(f"Response Preview: {json.dumps(response_json, indent=2)[:500]}...")
        except:
            print(f"Response Text: {response.text[:200]}...")
        
        if response.status_code == expected_status:
            print("✅ Test PASSED")
            return response.json() if response.content else {}
        else:
            print(f"❌ Test FAILED - Expected {expected_status}, got {response.status_code}")
            return {}
            
    except Exception as e:
        print(f"❌ Test FAILED - Exception: {e}")
        return {}

def run_comprehensive_api_tests():
    """Run comprehensive tests of all API endpoints."""
    print("TFT Analytics API Server Test Suite")
    print("="*60)
    
    # Test 1: Health Check
    health_response = test_endpoint('GET', '/api/health')
    
    # Test 2: API Info
    info_response = test_endpoint('GET', '/api/info')
    
    # Test 3: System Stats
    stats_response = test_endpoint('GET', '/api/stats')
    
    # Test 4: Basic Unit Query (stats)
    basic_query = {
        'type': 'stats',
        'filters': {
            'unit': 'Jinx'
        }
    }
    query_response = test_endpoint('POST', '/api/query', basic_query)
    
    # Test 5: Complex Query (participants)
    complex_query = {
        'type': 'participants',
        'filters': {
            'trait': 'Sniper',
            'trait_min_tier': 2,
            'player_level': {
                'min': 8,
                'max': 10
            }
        },
        'limit': 50
    }
    complex_response = test_endpoint('POST', '/api/query', complex_query)
    
    # Test 6: Logical OR Query
    or_query = {
        'type': 'stats',
        'filters': {
            'unit': 'Jinx',
            'logical_operation': 'OR',
            'other_query_filters': {
                'unit': 'Aphelios'
            }
        }
    }
    or_response = test_endpoint('POST', '/api/query', or_query)
    
    # Test 7: Clustering Analysis (main clusters)
    clusters_response = test_endpoint('GET', '/api/clusters?type=main&limit=5')
    
    # Test 8: Sub-clusters
    sub_clusters_response = test_endpoint('GET', '/api/clusters?type=sub&limit=10')
    
    # Test 9: Cluster Details (if clusters exist)
    if clusters_response.get('results', {}).get('clusters'):
        cluster_id = clusters_response['results']['clusters'][0]['id']
        cluster_details_response = test_endpoint('GET', f'/api/clusters/{cluster_id}?type=main')
    
    # Test 10: Error Handling - Invalid query
    invalid_query = {
        'type': 'invalid_type',
        'filters': {}
    }
    test_endpoint('POST', '/api/query', invalid_query, expected_status=400)
    
    # Test 11: Error Handling - Missing cluster
    test_endpoint('GET', '/api/clusters/999?type=main', expected_status=404)
    
    print(f"\n{'='*60}")
    print("🎉 API Test Suite Completed!")
    print("="*60)
    
    print("\nNext Steps:")
    print("1. If tests passed, your API server is ready for Firebase integration")
    print("2. Deploy to your GCP VM with: python api_server.py")
    print("3. Configure firewall rules to allow external traffic on port 8080")
    print("4. Set up nginx reverse proxy for SSL/domain routing")
    print("5. Start building your Firebase webapp frontend")

if __name__ == "__main__":
    print("Starting API Server Test...")
    print("Note: Make sure api_server.py is running on localhost:8080")
    print("You can start it with: python api_server.py")
    
    input("Press Enter when the API server is running...")
    
    run_comprehensive_api_tests()