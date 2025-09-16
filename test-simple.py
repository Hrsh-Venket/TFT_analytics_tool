#!/usr/bin/env python3
"""
Simple test script for Cloud Functions - Windows compatible
"""

import sys
import os

# Add shared modules to path
sys.path.insert(0, 'functions/shared')

def test_imports():
    """Test basic imports"""
    print("Testing imports...")

    try:
        from utils import cors_headers, json_response
        print("OK: utils.py")

        from bigquery_operations import BigQueryDataImporter
        print("OK: bigquery_operations.py")

        from querying import TFTQuery
        print("OK: querying.py")

        return True
    except ImportError as e:
        print(f"ERROR: {e}")
        return False

def test_files():
    """Test required files exist"""
    print("\nTesting files...")

    files = [
        'functions/get-stats/main.py',
        'functions/get-clusters/main.py',
        'functions/execute-query/main.py',
        'functions/cluster-details/main.py',
        'deploy-functions.sh',
        'firebase/public/index.html'
    ]

    all_exist = True
    for f in files:
        if os.path.exists(f):
            print(f"OK: {f}")
        else:
            print(f"MISSING: {f}")
            all_exist = False

    return all_exist

def test_query():
    """Test query construction"""
    print("\nTesting query...")

    try:
        from querying import TFTQuery

        query = TFTQuery()
        query = query.add_unit('Jinx')
        print("OK: Query construction works")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    print("TFT Analytics Test Suite")
    print("=" * 40)

    tests = [test_imports, test_files, test_query]
    passed = 0

    for test in tests:
        if test():
            passed += 1

    print(f"\nResults: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("SUCCESS: Ready for deployment")
        print("\nNext steps:")
        print("1. Update PROJECT_ID in firebase/public/js/api.js")
        print("2. On VM: chmod +x deploy-functions.sh")
        print("3. On VM: ./deploy-functions.sh")
    else:
        print("FAILED: Fix errors before deployment")

if __name__ == "__main__":
    main()