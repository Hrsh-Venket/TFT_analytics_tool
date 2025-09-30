#!/usr/bin/env python3
"""
Temporary debugging script to identify item naming formats in BigQuery database.
This will help fix the item query functionality.
"""

from google.cloud import bigquery
import json

print("=" * 60)
print("ITEM NAMING DEBUG SCRIPT")
print("=" * 60)
print()

try:
    client = bigquery.Client()

    # Query 1: Find all item names on Jinx units
    print("1. Checking item names found on Jinx units...")
    print("-" * 40)

    query1 = '''
    SELECT DISTINCT item
    FROM `tft-analytics-tool.tft_analytics.match_participants`,
    UNNEST(units) AS unit,
    UNNEST(unit.item_names) AS item
    WHERE unit.character_id = 'Jinx'
    LIMIT 30
    '''

    result1 = client.query(query1)
    jinx_items = []
    for row in result1:
        jinx_items.append(row.item)
        print(f"  - {row.item}")

    if not jinx_items:
        print("  No items found on Jinx units!")

    print()

    # Query 2: Check for Infinity Edge variations
    print("2. Looking for Infinity Edge variations...")
    print("-" * 40)

    query2 = '''
    SELECT DISTINCT item, COUNT(*) as count
    FROM `tft-analytics-tool.tft_analytics.match_participants`,
    UNNEST(units) AS unit,
    UNNEST(unit.item_names) AS item
    WHERE LOWER(item) LIKE '%infinity%'
    GROUP BY item
    ORDER BY count DESC
    LIMIT 10
    '''

    result2 = client.query(query2)
    infinity_variations = []
    for row in result2:
        infinity_variations.append(row.item)
        print(f"  - {row.item} ({row.count} occurrences)")

    if not infinity_variations:
        print("  No Infinity Edge variations found!")

    print()

    # Query 3: Check general item naming patterns
    print("3. Checking general item naming patterns...")
    print("-" * 40)

    query3 = '''
    SELECT item, COUNT(*) as count
    FROM `tft-analytics-tool.tft_analytics.match_participants`,
    UNNEST(units) AS unit,
    UNNEST(unit.item_names) AS item
    GROUP BY item
    ORDER BY count DESC
    LIMIT 20
    '''

    result3 = client.query(query3)
    print("Top 20 most common items:")
    common_items = []
    for row in result3:
        common_items.append(row.item)
        print(f"  - {row.item} ({row.count} occurrences)")

    print()

    # Query 4: Check if items have prefixes
    print("4. Analyzing item prefixes...")
    print("-" * 40)

    query4 = '''
    SELECT
        CASE
            WHEN item LIKE 'TFT_%' THEN 'TFT_ prefix'
            WHEN item LIKE 'TFTItem_%' THEN 'TFTItem_ prefix'
            WHEN item LIKE 'Item_%' THEN 'Item_ prefix'
            ELSE 'No common prefix'
        END as prefix_type,
        COUNT(DISTINCT item) as unique_items,
        COUNT(*) as total_occurrences
    FROM `tft-analytics-tool.tft_analytics.match_participants`,
    UNNEST(units) AS unit,
    UNNEST(unit.item_names) AS item
    GROUP BY prefix_type
    ORDER BY total_occurrences DESC
    '''

    result4 = client.query(query4)
    for row in result4:
        print(f"  {row.prefix_type}: {row.unique_items} unique items, {row.total_occurrences} total occurrences")

    print()

    # Query 5: Test specific failing queries (the exact ones user is trying)
    print("5. Testing the exact failing queries...")
    print("-" * 40)

    # Test the two specific queries from the user
    user_test_queries = [
        {
            "name": "User Query 1 (with underscore)",
            "item_name": "Infinity_Edge"
        },
        {
            "name": "User Query 2 (without underscore)",
            "item_name": "InfinityEdge"
        }
    ]

    for test in user_test_queries:
        print(f"\n  {test['name']} - Testing '{test['item_name']}':")

        # Test 1: Direct match
        query5a = f'''
        SELECT COUNT(*) as count
        FROM `tft-analytics-tool.tft_analytics.match_participants`,
        UNNEST(units) AS unit,
        UNNEST(unit.item_names) AS item
        WHERE unit.character_id = 'Jinx'
        AND item = '{test["item_name"]}'
        '''

        result5a = client.query(query5a)
        for row in result5a:
            print(f"    Direct match: {row.count}")

        # Test 2: Our current cleaning logic
        clean_item_id = test["item_name"].replace('TFT_Item_', '').replace('TFTItem_', '').replace('TFT_', '')

        query5b = f'''
        SELECT COUNT(*) as count
        FROM `tft-analytics-tool.tft_analytics.match_participants`,
        UNNEST(units) AS unit,
        UNNEST(unit.item_names) AS item
        WHERE unit.character_id = 'Jinx'
        AND (item = '{clean_item_id}'
             OR item = CONCAT('TFT_Item_', '{clean_item_id}')
             OR item = CONCAT('TFT_', '{clean_item_id}')
             OR item = CONCAT('TFTItem_', '{clean_item_id}')
             OR REGEXP_REPLACE(item, r'^TFT[0-9]*_Item_', '') = '{clean_item_id}')
        '''

        result5b = client.query(query5b)
        for row in result5b:
            print(f"    Our logic (clean_item_id='{clean_item_id}'): {row.count}")

        # Test 3: What if we try matching the exact known format?
        query5c = f'''
        SELECT COUNT(*) as count
        FROM `tft-analytics-tool.tft_analytics.match_participants`,
        UNNEST(units) AS unit,
        UNNEST(unit.item_names) AS item
        WHERE unit.character_id = 'Jinx'
        AND item = 'TFT_Item_{clean_item_id}'
        '''

        result5c = client.query(query5c)
        for row in result5c:
            print(f"    Direct TFT_Item_ prefix: {row.count}")

        # Test 4: What if the issue is the underscore?
        if "_" in test["item_name"]:
            no_underscore = test["item_name"].replace("_", "")
            query5d = f'''
            SELECT COUNT(*) as count
            FROM `tft-analytics-tool.tft_analytics.match_participants`,
            UNNEST(units) AS unit,
            UNNEST(unit.item_names) AS item
            WHERE unit.character_id = 'Jinx'
            AND item = 'TFT_Item_{no_underscore}'
            '''

            result5d = client.query(query5d)
            for row in result5d:
                print(f"    Without underscore 'TFT_Item_{no_underscore}': {row.count}")

    print()

    # Additional test: What exact transformations are happening?
    print("DEBUGGING TRANSFORMATIONS:")
    test_inputs = ["Infinity_Edge", "InfinityEdge"]
    for inp in test_inputs:
        clean = inp.replace('TFT_Item_', '').replace('TFTItem_', '').replace('TFT_', '')
        target = f'TFT_Item_{clean}'
        print(f"  '{inp}' -> clean: '{clean}' -> target: '{target}'")

    print()

    # Query 6: Check the actual working query with regex
    print("6. Testing regex-based item matching...")
    print("-" * 40)

    query6 = '''
    SELECT COUNT(*) as count
    FROM `tft-analytics-tool.tft_analytics.match_participants`,
    UNNEST(units) AS unit,
    UNNEST(unit.item_names) AS item
    WHERE unit.character_id = 'Jinx'
    AND (
        item = 'InfinityEdge'
        OR item = CONCAT('TFTItem_', 'InfinityEdge')
        OR item = CONCAT('TFT_Item_', 'InfinityEdge')
        OR REGEXP_REPLACE(item, r'^TFT.*?Item_', '') = 'InfinityEdge'
    )
    '''

    result6 = client.query(query6)
    for row in result6:
        print(f"  Regex-based search found {row.count} Jinx units with Infinity Edge")

    # Query 7: Test if underscore vs no underscore is the issue
    print("\n7. Testing underscore variations...")
    print("-" * 40)
    
    underscore_tests = [
        "Infinity_Edge",
        "InfinityEdge", 
        "TFT_Item_Infinity_Edge",
        "TFT_Item_InfinityEdge",
        "TFT15_Item_InfinityEdge"
    ]
    
    for test_item in underscore_tests:
        query7 = f'''
        SELECT COUNT(*) as count
        FROM `tft-analytics-tool.tft_analytics.match_participants`,
        UNNEST(units) AS unit,
        UNNEST(unit.item_names) AS item
        WHERE unit.character_id = 'Jinx'
        AND item = '{test_item}'
        '''
        
        result7 = client.query(query7)
        for row in result7:
            print(f"  '{test_item}': {row.count} matches")

    # Query 8: Test TFTQuery logic step by step
    print("\n8. Testing current TFTQuery logic step by step...")
    print("-" * 40)
    
    # Simulate the exact logic from querying.py
    user_input = "Infinity_Edge"
    clean_item_id = user_input.replace('TFT_Item_', '').replace('TFTItem_', '').replace('TFT_', '')
    print(f"User input: '{user_input}' -> Clean ID: '{clean_item_id}'")
    
    # Test each condition individually
    test_conditions = [
        ("Direct match", clean_item_id),
        ("TFT_Item_ prefix", f"TFT_Item_{clean_item_id}"),
        ("TFT_ prefix", f"TFT_{clean_item_id}"),
        ("TFTItem_ prefix", f"TFTItem_{clean_item_id}")
    ]
    
    for desc, test_val in test_conditions:
        query8 = f'''
        SELECT COUNT(*) as count
        FROM `tft-analytics-tool.tft_analytics.match_participants`,
        UNNEST(units) AS unit,
        UNNEST(unit.item_names) AS item
        WHERE unit.character_id = 'Jinx'
        AND item = '{test_val}'
        '''
        
        result8 = client.query(query8)
        for row in result8:
            print(f"  {desc} ('{test_val}'): {row.count}")

    # Query 9: Test current regex logic
    print("\n9. Testing current regex logic...")
    print("-" * 40)
    
    query9 = f'''
    SELECT COUNT(*) as count
    FROM `tft-analytics-tool.tft_analytics.match_participants`,
    UNNEST(units) AS unit,
    UNNEST(unit.item_names) AS item
    WHERE unit.character_id = 'Jinx'
    AND REGEXP_REPLACE(item, r'^TFT[0-9]*_Item_', '') = '{clean_item_id}'
    '''
    
    result9 = client.query(query9)
    for row in result9:
        print(f"  Regex strip ('{clean_item_id}'): {row.count}")

    # Query 10: Show actual items that would match our regex
    print("\n10. Items that match our regex pattern...")
    print("-" * 40)
    
    query10 = f'''
    SELECT DISTINCT item, REGEXP_REPLACE(item, r'^TFT[0-9]*_Item_', '') as stripped
    FROM `tft-analytics-tool.tft_analytics.match_participants`,
    UNNEST(units) AS unit,
    UNNEST(unit.item_names) AS item
    WHERE unit.character_id = 'Jinx'
    AND REGEXP_REPLACE(item, r'^TFT[0-9]*_Item_', '') = '{clean_item_id}'
    LIMIT 10
    '''
    
    result10 = client.query(query10)
    for row in result10:
        print(f"  '{row.item}' -> stripped: '{row.stripped}'")

    # Query 11: Find what the actual Infinity Edge item is called
    print("\n11. Finding actual Infinity Edge variations...")
    print("-" * 40)
    
    query11 = '''
    SELECT DISTINCT item, COUNT(*) as count
    FROM `tft-analytics-tool.tft_analytics.match_participants`,
    UNNEST(units) AS unit,
    UNNEST(unit.item_names) AS item
    WHERE (
        LOWER(item) LIKE '%infinity%'
        OR LOWER(item) LIKE '%edge%'
    )
    GROUP BY item
    ORDER BY count DESC
    LIMIT 10
    '''
    
    result11 = client.query(query11)
    for row in result11:
        print(f"  '{row.item}': {row.count} occurrences")

    print()
    print("=" * 60)
    print("DEBUGGING COMPLETE")
    print("=" * 60)

    # Summary
    print("\nSUMMARY:")
    if common_items:
        print(f"Most common item format: {common_items[0]}")
    if infinity_variations:
        print(f"Infinity Edge is stored as: {infinity_variations[0]}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()