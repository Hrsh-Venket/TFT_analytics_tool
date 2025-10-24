#!/usr/bin/env python3
"""
TFT Item Mapping & Querying Tests

This script tests:
1. Name mapping functionality (items, units, traits)
2. Item mapping fixes (itemNames -> item_names transformation)
3. BigQuery item querying with actual data
"""

from google.cloud import bigquery

print("=" * 60)
print("TFT ITEM MAPPING & QUERYING TESTS")
print("=" * 60)
print()

try:
    client = bigquery.Client()

    # ====================================================================
    # NAME MAPPING FUNCTIONALITY TEST
    # ====================================================================

    print("=" * 60)
    print("TEST: NAME MAPPING FUNCTIONALITY")
    print("=" * 60)
    
    try:
        from name_mapper import TFTNameMapper, map_match_data
        
        # Test the name mapper directly
        mapper = TFTNameMapper()
        
        print(f"Mapper loaded with {len(mapper.items_mapping)} item mappings")
        
        # Test specific mapping
        test_item = "TFT_Item_InfinityEdge"
        mapped_item = mapper.map_item_name(test_item)
        print(f"Direct mapping test: '{test_item}' -> '{mapped_item}'")
        
        # Check if this specific mapping exists
        if test_item in mapper.items_mapping:
            print(f"✓ Mapping exists: {test_item} -> {mapper.items_mapping[test_item]}")
        else:
            print(f"✗ No mapping found for {test_item}")
            print("Available mappings (first 10):")
            for i, (key, value) in enumerate(list(mapper.items_mapping.items())[:10]):
                print(f"  {key} -> {value}")
        
        # Test a complete unit data structure
        print("\nTesting unit data mapping...")
        test_unit = {
            'character_id': 'TFT14_Jinx',
            'itemNames': ['TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper'],
            'tier': 3
        }
        
        mapped_unit = mapper.map_unit_data(test_unit)
        print(f"Original unit: {test_unit}")
        print(f"Mapped unit: {mapped_unit}")
        
        # Test with item_names field (snake_case)
        test_unit_snake = {
            'character_id': 'TFT14_Jinx',
            'item_names': ['TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper'],
            'tier': 3
        }
        
        mapped_unit_snake = mapper.map_unit_data(test_unit_snake)
        print(f"Original unit (snake_case): {test_unit_snake}")
        print(f"Mapped unit (snake_case): {mapped_unit_snake}")
        
    except Exception as mapper_error:
        print(f"Name mapping test failed: {mapper_error}")
        import traceback
        traceback.print_exc()

    # ====================================================================
    # NEW TESTS FOR ITEM MAPPING FIX VERIFICATION
    # ====================================================================

    print("\n" + "=" * 60)
    print("TESTING ITEM MAPPING FIX")
    print("=" * 60)

    # Test 1: Unit data transformation
    print("\n1. Testing unit data transformation...")
    print("-" * 40)

    try:
        test_unit = {
            'character_id': 'TFT15_Jinx',
            'itemNames': ['TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_Bloodthirster'],
            'tier': 3,
            'rarity': 4
        }

        print(f"Input unit: {test_unit}")
        mapped_unit = mapper.map_unit_data(test_unit)
        print(f"Output unit: {mapped_unit}")

        # Check that itemNames was removed and item_names was added
        if 'itemNames' in mapped_unit:
            print("  ✗ FAIL: 'itemNames' field still exists (should be deleted)")
        else:
            print("  ✓ PASS: 'itemNames' field was removed")

        if 'item_names' not in mapped_unit:
            print("  ✗ FAIL: 'item_names' field not created")
        else:
            print(f"  ✓ PASS: 'item_names' field created with values: {mapped_unit['item_names']}")

            # Check that items were mapped
            expected_items = ['Infinity_Edge', 'Last_Whisper', 'Bloodthirster']
            if mapped_unit['item_names'] == expected_items:
                print(f"  ✓ PASS: Items correctly mapped!")
            else:
                print(f"  ✗ FAIL: Items not correctly mapped")
                print(f"    Got: {mapped_unit['item_names']}")
                print(f"    Expected: {expected_items}")

    except Exception as e:
        print(f"  ✗ Unit transformation test failed: {e}")
        traceback.print_exc()

    # Test 2: Complete match data mapping
    print("\n2. Testing complete match data mapping...")
    print("-" * 40)

    try:
        from name_mapper import map_match_data

        test_match = {
            'metadata': {'match_id': 'TEST_12345'},
            'info': {
                'game_datetime': 1234567890000,
                'participants': [{
                    'puuid': 'test-player-1',
                    'placement': 1,
                    'units': [
                        {
                            'character_id': 'TFT15_Jinx',
                            'itemNames': ['TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper'],
                            'tier': 3
                        }
                    ],
                    'traits': []
                }]
            }
        }

        mapped_match = map_match_data(test_match)
        participant = mapped_match['info']['participants'][0]
        unit = participant['units'][0]

        print(f"Unit character_id: {unit['character_id']}")
        print(f"Unit items: {unit.get('item_names', 'MISSING')}")

        if 'itemNames' in unit:
            print("  ✗ FAIL: 'itemNames' still exists in mapped match")
        else:
            print("  ✓ PASS: 'itemNames' removed from mapped match")

        if 'item_names' in unit and unit['item_names'] == ['Infinity_Edge', 'Last_Whisper']:
            print("  ✓ PASS: Match data correctly mapped!")
        else:
            print(f"  ✗ FAIL: Match data mapping incorrect")
            print(f"    Got: {unit.get('item_names', 'MISSING')}")

    except Exception as e:
        print(f"  ✗ Match data mapping test failed: {e}")
        traceback.print_exc()

    # Test 3: BigQuery insert simulation
    print("\n3. Testing BigQuery insert format...")
    print("-" * 40)

    try:
        # Simulate what happens in bigquery_operations.py insert_match_data
        test_match_bq = {
            'metadata': {'match_id': 'TEST_12345'},
            'info': {
                'game_datetime': 1234567890000,
                'participants': [{
                    'puuid': 'test',
                    'units': [{
                        'character_id': 'TFT15_Jinx',
                        'itemNames': ['TFT_Item_InfinityEdge'],
                        'tier': 3,
                        'rarity': 4
                    }]
                }]
            }
        }

        # Apply mapping (as done in insert_match_data)
        mapped_match_bq = map_match_data(test_match_bq)
        participant = mapped_match_bq['info']['participants'][0]
        unit = participant['units'][0]

        # Simulate BigQuery insert structure
        units_struct = {
            'character_id': unit.get('character_id', ''),
            'tier': unit.get('tier', None),
            'rarity': unit.get('rarity', None),
            'name': unit.get('name', ''),
            'item_names': unit.get('item_names', [])  # Should use item_names, not itemNames
        }

        print(f"BigQuery struct item_names: {units_struct['item_names']}")

        if units_struct['item_names'] == ['Infinity_Edge']:
            print("  ✓ PASS: BigQuery will store clean mapped item names!")
        else:
            print(f"  ✗ FAIL: BigQuery struct has wrong item_names: {units_struct['item_names']}")

    except Exception as e:
        print(f"  ✗ BigQuery insert format test failed: {e}")
        traceback.print_exc()

    # Test 4: Query simplification
    print("\n4. Testing query simplification...")
    print("-" * 40)

    try:
        import traceback
        from querying import TFTQuery

        query = TFTQuery()
        query.add_item_on_unit('Jinx', 'Infinity_Edge')

        # Check the filter directly
        if query._filters:
            filter_obj = query._filters[0]
            sql_condition = filter_obj.condition

            print(f"Query condition: {sql_condition[:100]}...")

            # Check if query uses simple matching
            if 'REGEXP_REPLACE' in sql_condition:
                print("  ✗ FAIL: Query still uses REGEXP_REPLACE (should be simple match)")
            else:
                print("  ✓ PASS: Query does not use REGEXP_REPLACE")

            if 'CONCAT' in sql_condition:
                print("  ✗ FAIL: Query still uses CONCAT (should be simple match)")
            else:
                print("  ✓ PASS: Query does not use CONCAT")

            if 'item = @item_id' in sql_condition:
                print("  ✓ PASS: Query uses simple string matching!")
            else:
                print("  ? WARNING: Could not verify exact query structure")
        else:
            print("  ✗ FAIL: No filters found in query")

    except Exception as e:
        print(f"  ✗ Query simplification test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("ITEM MAPPING FIX TESTS COMPLETE")
    print("=" * 60)

    # ====================================================================
    # BIGQUERY ITEM QUERYING TESTS (with new mapped data)
    # ====================================================================

    print("\n" + "=" * 60)
    print("TESTING BIGQUERY ITEM QUERYING")
    print("=" * 60)

    # Test 5: Check actual item names in BigQuery
    print("\n5. Checking actual item names in BigQuery database...")
    print("-" * 40)

    try:
        query_items = '''
        SELECT DISTINCT item, COUNT(*) as count
        FROM `tft-analytics-tool.tft_analytics.match_participants`,
        UNNEST(units) AS unit,
        UNNEST(unit.item_names) AS item
        WHERE item LIKE '%Infinity%' OR item LIKE '%Edge%'
        GROUP BY item
        ORDER BY count DESC
        LIMIT 10
        '''

        result_items = client.query(query_items)
        items_found = []

        print("Items containing 'Infinity' or 'Edge' in database:")
        for row in result_items:
            items_found.append(row.item)
            print(f"  {row.item}: {row.count} occurrences")

        if not items_found:
            print("  ⚠️  WARNING: No items found - database might be empty")
        else:
            # Check if items are mapped or unmapped
            has_mapped = any('Infinity_Edge' in item and not item.startswith('TFT') for item in items_found)
            has_unmapped = any('TFT_Item_InfinityEdge' in item for item in items_found)

            if has_mapped and not has_unmapped:
                print("  ✓ PASS: Database has MAPPED item names (e.g., 'Infinity_Edge')")
            elif has_unmapped and not has_mapped:
                print("  ✗ FAIL: Database has UNMAPPED item names (e.g., 'TFT_Item_InfinityEdge')")
                print("    You need to re-collect data with the new mapping!")
            elif has_mapped and has_unmapped:
                print("  ⚠️  WARNING: Database has MIXED mapped and unmapped items")
                print("    Recommend clearing and re-collecting all data")
            else:
                print("  ? Unknown item format")

    except Exception as e:
        print(f"  ✗ Database item check failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 6: Test actual query execution with item filter
    print("\n6. Testing query execution with item filter...")
    print("-" * 40)

    try:
        from querying import TFTQuery

        # Baseline query: Jinx without item filter
        print("  Testing baseline (Jinx without item filter)...")
        query_baseline = TFTQuery()
        query_baseline.add_unit('Jinx')

        result_baseline = query_baseline.get_stats()
        baseline_count = result_baseline['play_count']
        print(f"    Jinx (no item filter): {baseline_count} compositions")

        # Filtered query: Jinx WITH Infinity_Edge
        print("  Testing filtered (Jinx with Infinity_Edge)...")
        query_filtered = TFTQuery()
        query_filtered.add_unit('Jinx')
        query_filtered.add_item_on_unit('Jinx', 'Infinity_Edge')

        result_filtered = query_filtered.get_stats()
        filtered_count = result_filtered['play_count']
        print(f"    Jinx + Infinity_Edge: {filtered_count} compositions")

        # Analyze results
        print("\n  Analysis:")
        print(f"    Baseline count: {baseline_count}")
        print(f"    Filtered count: {filtered_count}")

        if baseline_count == 0:
            print("    ⚠️  WARNING: No Jinx compositions found in database")
            print("    Database might be empty or Jinx not played")
        elif filtered_count == baseline_count:
            print("    ✗ FAIL: Filter has NO EFFECT (counts are equal)")
            print("    This means item filtering is NOT working!")
            print("    Likely cause: Database has unmapped items but query expects mapped items")
        elif filtered_count > baseline_count:
            print("    ✗ FAIL: Filtered count > baseline (impossible!)")
            print("    Something is seriously wrong with the query")
        elif filtered_count == 0:
            print("    ⚠️  WARNING: Zero results with item filter")
            print("    Either:")
            print("      - No Jinx has Infinity Edge (unlikely)")
            print("      - Item names don't match (database unmapped, query expects mapped)")
        else:
            reduction = baseline_count - filtered_count
            percentage = (filtered_count / baseline_count * 100) if baseline_count > 0 else 0
            print(f"    ✓ PASS: Filter is WORKING!")
            print(f"    Filtered out: {reduction} compositions")
            print(f"    Percentage with IE: {percentage:.1f}%")

    except Exception as e:
        print(f"  ✗ Query execution test failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 7: Test with different items to verify consistency
    print("\n7. Testing multiple item queries for consistency...")
    print("-" * 40)

    try:
        from querying import TFTQuery

        test_items = [
            ('Jinx', 'Last_Whisper'),
            ('Jinx', 'Bloodthirster'),
            ('Jinx', 'Guinsoos_Rageblade')
        ]

        for unit, item in test_items:
            query = TFTQuery()
            query.add_unit(unit)
            query.add_item_on_unit(unit, item)

            try:
                result = query.get_stats()
                count = result['play_count']
                print(f"  {unit} + {item}: {count} compositions")

                if count > 0:
                    print(f"    ✓ Query returned results")
                else:
                    print(f"    ⚠️  No results (might be rare or item name mismatch)")
            except Exception as e:
                print(f"  ✗ Query failed for {unit} + {item}: {e}")

    except Exception as e:
        print(f"  ✗ Multiple item test failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 8: Verify query SQL is using simple matching
    print("\n8. Verifying query SQL uses simple item matching...")
    print("-" * 40)

    try:
        from querying import TFTQuery

        query = TFTQuery()
        query.add_unit('Jinx')
        query.add_item_on_unit('Jinx', 'Infinity_Edge')

        # Check the filter condition directly (simpler and more reliable)
        if query._filters and len(query._filters) >= 2:
            item_filter = query._filters[1]  # Second filter is the item filter
            sql_condition = item_filter.condition

            print("  Checking SQL condition in filter...")

            # Check for simple matching
            if 'item = @item_id' in sql_condition:
                print("    ✓ PASS: Uses simple string matching (item = @item_id)")
            else:
                print("    ⚠️  WARNING: Could not verify simple matching in SQL")

            # Check no complex operations
            if 'REGEXP_REPLACE' not in sql_condition:
                print("    ✓ PASS: No REGEXP_REPLACE in query")
            else:
                print("    ✗ FAIL: Still using REGEXP_REPLACE")

            if 'CONCAT' not in sql_condition:
                print("    ✓ PASS: No CONCAT in query")
            else:
                print("    ✗ FAIL: Still using CONCAT")

            # Show the condition for verification
            print(f"\n  Item filter SQL condition:")
            for line in sql_condition.strip().split('\n'):
                print(f"    {line}")
        else:
            print("    ⚠️  Could not access filter conditions")

    except Exception as e:
        print(f"  ✗ SQL verification test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("BIGQUERY ITEM QUERYING TESTS COMPLETE")
    print("=" * 60)

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()