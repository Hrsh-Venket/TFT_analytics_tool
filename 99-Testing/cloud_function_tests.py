#!/usr/bin/env python3
"""
Test script for Cloud Functions deployment
Verifies that all functions can be imported and basic functionality works
"""

import sys
import os
import json
from unittest.mock import Mock

# Add shared modules to path
sys.path.insert(0, 'functions/shared')

def test_shared_modules():
    """Test that all shared modules can be imported"""
    print("Testing shared module imports...")

    try:
        from utils import cors_headers, json_response, error_response
        print("OK utils.py imported successfully")

        from bigquery_operations import BigQueryDataImporter
        print("OK bigquery_operations.py imported successfully")

        from querying import TFTQuery
        print("OK querying.py imported successfully")

        from clustering import MainCluster, SubCluster
        print("OK clustering.py imported successfully")

    except ImportError as e:
        print(f"ERROR Import error: {e}")
        return False

    return True

def test_function_imports():
    """Test that all Cloud Function main modules can be imported"""
    print("\n🧪 Testing Cloud Function imports...")

    functions = [
        'functions/get-stats/main.py',
        'functions/get-clusters/main.py',
        'functions/execute-query/main.py',
        'functions/cluster-details/main.py'
    ]

    success = True

    for func_path in functions:
        try:
            func_dir = os.path.dirname(func_path)
            sys.path.insert(0, func_dir)

            # Import the main module
            spec = __import__('main')
            func_name = os.path.basename(func_dir).replace('-', '_')

            if hasattr(spec, 'get_stats'):
                print("✅ get-stats function imported")
            elif hasattr(spec, 'get_clusters'):
                print("✅ get-clusters function imported")
            elif hasattr(spec, 'execute_query'):
                print("✅ execute-query function imported")
            elif hasattr(spec, 'cluster_details'):
                print("✅ cluster-details function imported")
            else:
                print(f"⚠️  {func_path} imported but function not found")

            sys.path.pop(0)  # Remove from path

        except ImportError as e:
            print(f"❌ Failed to import {func_path}: {e}")
            success = False
        except Exception as e:
            print(f"❌ Error testing {func_path}: {e}")
            success = False

    return success

def test_utils():
    """Test utility functions"""
    print("\n🧪 Testing utility functions...")

    try:
        from utils import cors_headers, json_response, error_response, handle_cors_preflight

        # Test CORS headers
        headers = cors_headers()
        assert 'Access-Control-Allow-Origin' in headers
        print("✅ CORS headers working")

        # Test JSON response
        response = json_response({'test': 'data'})
        assert isinstance(response, tuple)
        assert len(response) == 3
        print("✅ JSON response working")

        # Test error response
        error_resp = error_response('Test error')
        assert isinstance(error_resp, tuple)
        print("✅ Error response working")

        # Test CORS preflight with mock request
        mock_request = Mock()
        mock_request.method = 'OPTIONS'
        preflight = handle_cors_preflight(mock_request)
        assert preflight is not None
        print("✅ CORS preflight working")

    except Exception as e:
        print(f"❌ Utility function test failed: {e}")
        return False

    return True

def test_query_syntax():
    """Test TFTQuery syntax parsing"""
    print("\n🧪 Testing query syntax...")

    try:
        from querying import TFTQuery

        # Test basic query construction (without execution)
        query = TFTQuery()
        query = query.add_unit('Jinx')
        print("✅ Basic query construction working")

        # Test query string evaluation (safe environment)
        query_text = "TFTQuery().add_unit('Jinx')"
        safe_globals = {
            'TFTQuery': TFTQuery,
            '__builtins__': {}
        }

        result = eval(query_text, safe_globals, {})
        assert isinstance(result, TFTQuery)
        print("✅ Query string evaluation working")

    except Exception as e:
        print(f"❌ Query syntax test failed: {e}")
        return False

    return True

def check_deployment_files():
    """Check that deployment files exist"""
    print("\n🧪 Checking deployment files...")

    required_files = [
        'deploy-functions.sh',
        'delete-functions.sh',
        'firebase/firebase.json',
        'firebase/public/index.html',
        'firebase/public/js/api.js',
        'firebase/public/js/app.js'
    ]

    missing_files = []

    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path} exists")

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False

    return True

def main():
    """Run all tests"""
    print("🚀 TFT Analytics Cloud Functions Test Suite")
    print("=" * 50)

    tests = [
        ("Shared Modules", test_shared_modules),
        ("Function Imports", test_function_imports),
        ("Utility Functions", test_utils),
        ("Query Syntax", test_query_syntax),
        ("Deployment Files", check_deployment_files)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")

    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Ready for deployment.")
        print("\n📋 Next steps:")
        print("1. Update PROJECT_ID in firebase/public/js/api.js")
        print("2. Run: chmod +x deploy-functions.sh (on VM)")
        print("3. Run: ./deploy-functions.sh (on VM)")
        print("4. Deploy Firebase: firebase deploy --only hosting")
    else:
        print("⚠️  Some tests failed. Fix issues before deployment.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)