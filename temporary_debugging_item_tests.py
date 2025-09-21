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

    # Query 5: Test specific query that's failing
    print("5. Testing the specific failing query...")
    print("-" * 40)

    # Test different variations
    test_variations = [
        "InfinityEdge",
        "Infinity_Edge",
        "Infinity Edge",
        "infinityedge",
        "TFT_InfinityEdge",
        "TFTItem_InfinityEdge"
    ]

    for item_name in test_variations:
        query5 = f'''
        SELECT COUNT(*) as count
        FROM `tft-analytics-tool.tft_analytics.match_participants`,
        UNNEST(units) AS unit,
        UNNEST(unit.item_names) AS item
        WHERE unit.character_id = 'Jinx'
        AND item = '{item_name}'
        '''

        result5 = client.query(query5)
        for row in result5:
            if row.count > 0:
                print(f"  ✓ Found {row.count} Jinx units with '{item_name}'")
            else:
                print(f"  ✗ No Jinx units found with '{item_name}'")

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