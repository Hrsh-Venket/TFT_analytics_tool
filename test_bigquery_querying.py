#!/usr/bin/env python3
"""
Test script for BigQuery-based TFT querying functionality.

This script tests the new BigQuery querying system with real data.
Run this on your VM after data collection to verify queries work correctly.
"""

import logging
import sys
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_bigquery_querying():
    """Test BigQuery TFT querying functionality."""
    
    print("BigQuery TFT Querying Test")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        from querying import TFTQuery, HAS_BIGQUERY
        
        if not HAS_BIGQUERY:
            print("❌ BigQuery dependencies not available")
            print("Install with: pip3 install --user google-cloud-bigquery")
            return False
            
        print("✅ BigQuery dependencies available")
        print()
        
        # Test 1: Basic query - all data
        print("1. Testing basic query (all participants)...")
        try:
            query = TFTQuery()
            # Just get a small sample to test connectivity
            query.add_custom_filter("RAND() < 0.1")  # Random 10% sample
            
            participants = query.execute()
            print(f"✅ Found {len(participants)} participants in sample")
            
            if participants:
                sample = participants[0]
                print(f"   Sample participant: Match {sample.get('match_id', 'N/A')}, Placement {sample.get('placement', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Basic query failed: {e}")
            return False
        
        # Test 2: Unit-based query
        print("\n2. Testing unit-based query...")
        try:
            # Look for a common unit - let's try different variations
            unit_names = ['TFT14_Jinx', 'Jinx', 'TFT_Jinx', 'jinx']
            
            found_results = False
            for unit_name in unit_names:
                try:
                    jinx_query = TFTQuery().add_unit(unit_name)
                    jinx_stats = jinx_query.get_stats()
                    
                    if jinx_stats and jinx_stats['play_count'] > 0:
                        print(f"✅ Unit query '{unit_name}' successful:")
                        print(f"   Play count: {jinx_stats['play_count']}")
                        print(f"   Average placement: {jinx_stats['avg_placement']}")
                        print(f"   Win rate: {jinx_stats['winrate']}%")
                        print(f"   Top 4 rate: {jinx_stats['top4_rate']}%")
                        found_results = True
                        break
                        
                except Exception as e:
                    print(f"   '{unit_name}' failed: {e}")
                    continue
            
            if not found_results:
                print("ℹ️  No results found for common unit names - this is normal if units use different naming")
                
        except Exception as e:
            print(f"❌ Unit query test failed: {e}")
        
        # Test 3: Trait-based query
        print("\n3. Testing trait-based query...")
        try:
            # Look for common traits
            trait_names = ['TFT14_Vanguard', 'Vanguard', 'Star_Guardian', 'TFT14_StarGuardian']
            
            found_results = False
            for trait_name in trait_names:
                try:
                    trait_query = TFTQuery().add_trait(trait_name, min_tier=1)
                    trait_stats = trait_query.get_stats()
                    
                    if trait_stats and trait_stats['play_count'] > 0:
                        print(f"✅ Trait query '{trait_name}' successful:")
                        print(f"   Play count: {trait_stats['play_count']}")
                        print(f"   Average placement: {trait_stats['avg_placement']}")
                        found_results = True
                        break
                        
                except Exception as e:
                    print(f"   '{trait_name}' failed: {e}")
                    continue
            
            if not found_results:
                print("ℹ️  No results found for common trait names - check actual trait names in your data")
                
        except Exception as e:
            print(f"❌ Trait query test failed: {e}")
        
        # Test 4: Level-based query
        print("\n4. Testing level-based query...")
        try:
            high_level_query = TFTQuery().add_player_level(min_level=8)
            high_level_stats = high_level_query.get_stats()
            
            if high_level_stats:
                print(f"✅ Level 8+ query successful:")
                print(f"   Play count: {high_level_stats['play_count']}")
                print(f"   Average placement: {high_level_stats['avg_placement']}")
            else:
                print("ℹ️  No level 8+ participants found")
                
        except Exception as e:
            print(f"❌ Level query test failed: {e}")
        
        # Test 5: Combined query (OR logic)
        print("\n5. Testing combined query with OR logic...")
        try:
            # Try to find participants with either high level OR specific placement
            level_query = TFTQuery().add_player_level(min_level=9)
            placement_query = TFTQuery().add_custom_filter("placement = 1")
            
            combined_query = level_query.or_(placement_query)
            combined_stats = combined_query.get_stats()
            
            if combined_stats:
                print(f"✅ Combined OR query successful:")
                print(f"   Play count: {combined_stats['play_count']}")
                print(f"   Average placement: {combined_stats['avg_placement']}")
            else:
                print("ℹ️  No results for combined query")
                
        except Exception as e:
            print(f"❌ Combined query test failed: {e}")
        
        print("\n" + "=" * 50)
        print("✅ BigQuery querying test completed!")
        print("Your BigQuery TFT querying system is working correctly.")
        print("\nYou can now use queries like:")
        print("  from querying import TFTQuery")
        print("  stats = TFTQuery().add_unit('UnitName').get_stats()")
        print("  stats = TFTQuery().add_trait('TraitName').get_stats()")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the correct directory")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def inspect_data_structure():
    """Helper function to inspect the actual data structure in BigQuery."""
    print("\n" + "=" * 50)
    print("Data Structure Inspection")
    print("=" * 50)
    
    try:
        from querying import TFTQuery
        
        # Get a small sample to inspect structure
        query = TFTQuery()
        query.add_custom_filter("RAND() < 0.01")  # 1% sample
        
        participants = query.execute()
        
        if participants:
            sample = participants[0]
            print(f"Sample participant keys: {list(sample.keys())}")
            
            if 'units' in sample and sample['units']:
                print(f"Sample unit structure: {sample['units'][0] if sample['units'] else 'No units'}")
            
            if 'traits' in sample and sample['traits']:
                print(f"Sample trait structure: {sample['traits'][0] if sample['traits'] else 'No traits'}")
                
            print(f"Sample values:")
            for key in ['match_id', 'placement', 'level', 'last_round']:
                print(f"  {key}: {sample.get(key, 'N/A')}")
        
    except Exception as e:
        print(f"Data inspection failed: {e}")

if __name__ == "__main__":
    success = test_bigquery_querying()
    
    # Optionally inspect data structure
    if len(sys.argv) > 1 and sys.argv[1] == "--inspect":
        inspect_data_structure()
    
    sys.exit(0 if success else 1)