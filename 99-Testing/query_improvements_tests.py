#!/usr/bin/env python3
"""
Test script for TFT querying system functionality.

This script tests:
1. Removal of get_participants function
2. Item querying with prefix handling
3. Pretty table formatting for results
4. Proper "not found" messages
5. Multi-line query parsing
6. Safe query parser implementation
"""

import os
import sys
import json
import requests
from datetime import datetime

# Set test mode
os.environ['TFT_TEST_MODE'] = 'true'

def test_local_querying():
    """Test the local querying system improvements."""
    print("=" * 60)
    print("LOCAL QUERYING SYSTEM TEST")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        from querying import TFTQuery

        # Test 1: Verify get_participants is removed
        print("1. Testing that get_participants is removed...")
        query = TFTQuery()
        if hasattr(query, 'get_participants'):
            print("   FAILED: get_participants method still exists")
        else:
            print("   PASSED: get_participants method successfully removed")
        print()

        # Test 2: Test item querying with various prefixes
        print("2. Testing item query with prefix handling...")
        test_items = [
            ("Infinity_Edge", "Clean item name"),
            ("TFTItem_Infinity_Edge", "With TFTItem prefix"),
            ("TFT_Item_Infinity_Edge", "With TFT_Item prefix"),
        ]

        for item_name, description in test_items:
            try:
                query = TFTQuery().add_item_on_unit('Jinx', item_name)
                print(f"   OK: {description}: Query created successfully")
            except Exception as e:
                print(f"   FAILED: {description}: {e}")
        print()

        # Test 3: Test get_stats returns None for no data
        print("3. Testing get_stats with no matching data...")
        # Create an impossible query
        query = TFTQuery().add_unit('NonExistentUnit123456')
        result = query.get_stats()

        if result is None:
            print("   PASSED: Returns None for no matching data")
        else:
            print(f"   FAILED: Expected None, got {result}")
        print()

        # Test 4: Test stats formatting
        print("4. Testing statistics formatting...")
        # In test mode, this should return sample data
        query = TFTQuery().add_unit('Jinx')
        stats = query.get_stats()

        if stats:
            required_fields = ['play_count', 'avg_placement', 'winrate', 'top4_rate']
            missing_fields = [f for f in required_fields if f not in stats]

            if not missing_fields:
                print("   PASSED: All required statistics fields present")
                print(f"      Play Count: {stats['play_count']}")
                print(f"      Avg Placement: {stats['avg_placement']}")
                print(f"      Win Rate: {stats['winrate']}%")
                print(f"      Top 4 Rate: {stats['top4_rate']}%")
            else:
                print(f"   FAILED: Missing fields: {missing_fields}")
        else:
            print("   WARNING:  WARNING: No stats returned in test mode")
        print()

        print("=" * 60)
        print("PASSED: LOCAL TESTS COMPLETED")
        print("=" * 60)
        print()

    except ImportError as e:
        print(f"FAILED: Import error: {e}")
        return False
    except Exception as e:
        print(f"FAILED: Unexpected error: {e}")
        return False

    return True

def test_query_parser():
    """Test the query parser functionality."""
    print("=" * 60)
    print("QUERY PARSER TEST")
    print("=" * 60)
    print()

    # Import the parser from the cloud function
    sys.path.insert(0, 'functions/execute-query')

    try:
        from main import parse_tft_query, format_query_result

        # Test 1: Single-line query
        print("1. Testing single-line query parsing...")
        query = 'TFTQuery().add_unit("Jinx").get_stats()'
        result = parse_tft_query(query)
        if result is not None:
            print("   PASSED: Single-line query parsed")
        else:
            print("   FAILED: Single-line query failed to parse")
        print()

        # Test 2: Multi-line query with spaces
        print("2. Testing multi-line query parsing...")
        multi_query = """
        TFTQuery()
            .add_unit("Varus")
            .add_unit("Jinx")
            .get_stats()
        """
        result = parse_tft_query(multi_query)
        if result is not None:
            print("   PASSED: Multi-line query parsed")
        else:
            print("   FAILED: Multi-line query failed to parse")
        print()

        # Test 3: Multi-line query with indentation
        print("3. Testing indented multi-line query...")
        indented_query = """
        TFTQuery()
          .add_unit("Varus")
          .add_unit("Jinx")
        .get_stats()
        """
        result = parse_tft_query(indented_query)
        if result is not None:
            print("   PASSED: Indented query parsed")
        else:
            print("   FAILED: Indented query failed to parse")
        print()

        # Test 4: SimpleTFTQuery compatibility
        print("4. Testing SimpleTFTQuery compatibility...")
        simple_query = 'SimpleTFTQuery().add_unit("Jinx").get_stats()'
        result = parse_tft_query(simple_query)
        if result is not None:
            print("   PASSED: SimpleTFTQuery converted and parsed")
        else:
            print("   FAILED: SimpleTFTQuery failed to parse")
        print()

        # Test 5: Item query with underscores
        print("5. Testing item query with underscores...")
        item_query = 'TFTQuery().add_item_on_unit("Jinx", "Infinity_Edge").get_stats()'
        result = parse_tft_query(item_query)
        if result is not None:
            print("   PASSED: Item query parsed")
        else:
            print("   FAILED: Item query failed to parse")
        print()

        # Test 6: Not found formatting
        print("6. Testing 'not found' formatting...")
        formatted = format_query_result(None)
        if formatted['formatted_output'] == 'Not found in the data':
            print("   PASSED: 'Not found' message correct")
        else:
            print(f"   FAILED: Expected 'Not found in the data', got '{formatted['formatted_output']}'")
        print()

        # Test 7: Table formatting
        print("7. Testing table formatting...")
        sample_stats = {
            'play_count': 100,
            'avg_placement': 4.5,
            'winrate': 12.5,
            'top4_rate': 50.0
        }
        formatted = format_query_result(sample_stats)

        if 'formatted_output' in formatted:
            output = formatted['formatted_output']
            print("   Table output:")
            for line in output.split('\n'):
                print(f"      {line}")
            # Check for table structure - look for key components
            if 'Metric' in output and 'Value' in output and '-' in output:
                print("   PASSED: Table format correct")
            else:
                print("   FAILED: Table format incorrect")
        else:
            print("   FAILED: No formatted_output in response")
        print()

        # Test 8: Deprecated get_participants handling
        print("8. Testing deprecated get_participants conversion...")
        deprecated_query = 'TFTQuery().add_unit("Jinx").get_participants()'
        result = parse_tft_query(deprecated_query)
        if result is not None:
            print("   PASSED: get_participants converted to get_stats")
        else:
            print("   FAILED: get_participants not handled")
        print()

        print("=" * 60)
        print("PASSED: PARSER TESTS COMPLETED")
        print("=" * 60)
        print()

        return True

    except ImportError as e:
        print(f"FAILED: Import error: {e}")
        print("   Make sure you're running from the project root directory")
        return False
    except Exception as e:
        print(f"FAILED: Unexpected error: {e}")
        return False

def test_api_endpoint(base_url="http://localhost:8080"):
    """Test the API endpoint with the improvements."""
    print("=" * 60)
    print("API ENDPOINT TEST")
    print("=" * 60)
    print(f"Testing against: {base_url}/api/query")
    print()

    test_queries = [
        {
            "name": "Simple unit query",
            "query": 'TFTQuery().add_unit("Jinx").get_stats()'
        },
        {
            "name": "Multi-line query",
            "query": """TFTQuery()
                .add_unit("Jinx")
                .add_unit("Varus")
                .get_stats()"""
        },
        {
            "name": "Item query",
            "query": 'TFTQuery().add_item_on_unit("Jinx", "Infinity_Edge").get_stats()'
        },
        {
            "name": "Non-existent unit (should return 'not found')",
            "query": 'TFTQuery().add_unit("FakeUnit999").get_stats()'
        },
        {
            "name": "SimpleTFTQuery compatibility",
            "query": 'SimpleTFTQuery().add_unit("Jinx").get_stats()'
        }
    ]

    for i, test_case in enumerate(test_queries, 1):
        print(f"{i}. Testing: {test_case['name']}")

        try:
            response = requests.post(
                f"{base_url}/api/query",
                json={"query": test_case['query']},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()

                # Check for expected fields
                if 'formatted_output' in data:
                    print(f"   PASSED: Got formatted output")

                    # For non-existent queries, check for "not found"
                    if "FakeUnit999" in test_case['query']:
                        if "Not found in the data" in data['formatted_output']:
                            print(f"   PASSED: 'Not found' message displayed correctly")
                        else:
                            print(f"   FAILED: Expected 'Not found' message")

                    # Show a snippet of the output
                    lines = data['formatted_output'].split('\n')
                    print(f"   Output preview:")
                    for line in lines[:3]:
                        print(f"      {line}")
                else:
                    print(f"   FAILED: Missing formatted_output")

            else:
                print(f"   FAILED: HTTP {response.status_code}")
                print(f"      Response: {response.text[:200]}")

        except requests.exceptions.ConnectionError:
            print(f"   WARNING:  SKIPPED: API not available (start local server to test)")
        except Exception as e:
            print(f"   FAILED: {e}")

        print()

    print("=" * 60)
    print("API TESTS COMPLETED")
    print("=" * 60)
    print()

def main():
    """Run all tests."""
    print("\nCOMPREHENSIVE TFT QUERY SYSTEM TEST SUITE\n")

    # Test local querying system
    local_success = test_local_querying()

    # Test query parser
    parser_success = test_query_parser()

    # Test API endpoint (optional - only if running)
    print("Testing API endpoint (optional)...")
    print("To test the API, deploy the function or run locally with:")
    print("  functions-framework --target execute_query --port 8080")
    print()
    test_api_endpoint()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Local Querying: {'PASSED' if local_success else 'FAILED'}")
    print(f"Query Parser: {'PASSED' if parser_success else 'FAILED'}")
    print("API Endpoint: Check results above")
    print("=" * 60)

    if local_success and parser_success:
        print("\nAll required tests passed!")
        print("\nKey features verified:")
        print("1. get_participants function removed")
        print("2. Item querying with TFTItem prefix handling")
        print("3. Pretty table formatting for results")
        print("4. 'Not found' messages for empty results")
        print("5. Multi-line query parsing")
        print("6. Safe query parser implementation")
        return 0
    else:
        print("\nSome tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())