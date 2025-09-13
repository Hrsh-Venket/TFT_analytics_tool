#!/usr/bin/env python3
"""
Clear BigQuery Data Script

This script safely empties your BigQuery TFT analytics table to allow for clean data re-import.
Use this when you want to start fresh with new data collection runs.
"""

import sys
import logging
import os
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_global_matches_file(global_matches_file='global_matches_downloaded.json'):
    """
    Clear the global matches tracking file to allow re-processing of all matches.
    
    Args:
        global_matches_file: Path to the global matches tracking file
        
    Returns:
        Tuple of (success, match_count_cleared)
    """
    try:
        if os.path.exists(global_matches_file):
            # Load and show stats before clearing
            with open(global_matches_file, 'r') as f:
                try:
                    global_data = json.load(f)
                    match_count = len(global_data.get('downloaded_matches', []))
                except json.JSONDecodeError:
                    match_count = 0
            
            # Remove the file
            os.remove(global_matches_file)
            print(f"🗑️  Cleared global matches file: {global_matches_file}")
            print(f"   Matches that were marked as downloaded: {match_count:,}")
            return True, match_count
        else:
            print(f"ℹ️  Global matches file not found: {global_matches_file}")
            return True, 0
            
    except Exception as e:
        print(f"❌ Error clearing global matches file: {e}")
        return False, 0

def clear_bigquery_table(project_id=None, dataset_id='tft_analytics', table_name='match_participants', confirm=True, clear_tracking=True):
    """
    Clear all data from the BigQuery table and optionally the global matches tracking file.
    
    Args:
        project_id: GCP project ID (auto-detected if None)
        dataset_id: BigQuery dataset ID
        table_name: Table name to clear
        confirm: Whether to ask for confirmation before deletion
        clear_tracking: Whether to also clear the global matches tracking file
    """
    try:
        from google.cloud import bigquery
        from google.cloud.exceptions import NotFound
        
        client = bigquery.Client(project=project_id)
        project_id = project_id or client.project
        table_id = f"{project_id}.{dataset_id}.{table_name}"
        
        print("BigQuery Table Cleanup")
        print("=" * 50)
        print(f"Project: {project_id}")
        print(f"Dataset: {dataset_id}")
        print(f"Table: {table_name}")
        print(f"Full table ID: {table_id}")
        print()
        
        # Check if table exists
        try:
            table = client.get_table(table_id)
            print(f"✅ Table found: {table.full_table_id}")
            
            # Get current row count
            query = f"SELECT COUNT(*) as row_count FROM `{table_id}`"
            result = client.query(query).result()
            row_count = next(result).row_count
            print(f"📊 Current row count: {row_count:,} rows")
            
            if row_count == 0:
                print("ℹ️  Table is already empty - nothing to clear!")
                return True
            
        except NotFound:
            print(f"❌ Table not found: {table_id}")
            print("ℹ️  Nothing to clear - table doesn't exist yet.")
            return True
        
        # Check global matches file status
        global_matches_file = 'global_matches_downloaded.json'
        global_matches_count = 0
        has_global_file = os.path.exists(global_matches_file)
        
        if has_global_file:
            try:
                with open(global_matches_file, 'r') as f:
                    global_data = json.load(f)
                    global_matches_count = len(global_data.get('downloaded_matches', []))
            except:
                global_matches_count = 0
        
        # Confirmation prompt
        if confirm:
            print(f"\n⚠️  WARNING: This will perform a COMPLETE RESET of your TFT analytics data!")
            print("This action cannot be undone.")
            print(f"\nWhat will be cleared:")
            print(f"  • BigQuery table: {row_count:,} rows")
            if clear_tracking and has_global_file:
                print(f"  • Global matches tracking: {global_matches_count:,} matches")
            elif clear_tracking:
                print(f"  • Global matches tracking: (file doesn't exist)")
            else:
                print(f"  • Global matches tracking: (will be preserved)")
            
            print(f"\nReasons you might want to do this:")
            print("  • Starting fresh with new name mappings")
            print("  • Fixing data collection issues")
            print("  • Cleaning up test data")
            print("  • Re-processing matches with updated logic")
            print()
            
            response = input("Are you sure you want to proceed? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("❌ Operation cancelled by user")
                return False
        
        print(f"\n🗑️  Clearing table {table_name}...")
        start_time = datetime.now()
        
        # Use DELETE FROM to clear all rows (keeps table structure)
        delete_query = f"DELETE FROM `{table_id}` WHERE TRUE"
        
        print("   Executing DELETE operation...")
        job = client.query(delete_query)
        result = job.result()  # Wait for job to complete
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Verify deletion
        verify_query = f"SELECT COUNT(*) as row_count FROM `{table_id}`"
        verify_result = client.query(verify_query).result()
        final_count = next(verify_result).row_count
        
        print(f"✅ Table cleared successfully!")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Rows deleted: {row_count:,}")
        print(f"   Final row count: {final_count:,}")
        
        # Clear global matches tracking file if requested
        tracking_success = True
        if clear_tracking:
            print(f"\n🗑️  Clearing global matches tracking...")
            tracking_success, matches_cleared = clear_global_matches_file()
        
        if final_count == 0 and tracking_success:
            print(f"\n🎉 Complete reset successful!")
            print(f"   BigQuery table: empty")
            if clear_tracking:
                print(f"   Match tracking: cleared")
            print(f"\n✨ Your system is now ready for fresh data collection with name mappings!")
            print(f"\nNext steps:")
            print("1. Run: python data_collection.py")
            print("2. Test: python test_bigquery_querying.py")
        else:
            if final_count > 0:
                print(f"⚠️  Warning: Table still has {final_count} rows - deletion may not have completed fully")
            if not tracking_success and clear_tracking:
                print(f"⚠️  Warning: Could not clear global matches tracking file")
        
        return True
        
    except ImportError:
        print("❌ BigQuery dependencies not available")
        print("Install with: pip3 install --user google-cloud-bigquery")
        return False
    except Exception as e:
        print(f"❌ Error clearing table: {e}")
        return False

def get_table_info(project_id=None, dataset_id='tft_analytics', table_name='match_participants'):
    """
    Get information about the current table without clearing it.
    
    Args:
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID  
        table_name: Table name to inspect
    """
    try:
        from google.cloud import bigquery
        from google.cloud.exceptions import NotFound
        
        client = bigquery.Client(project=project_id)
        project_id = project_id or client.project
        table_id = f"{project_id}.{dataset_id}.{table_name}"
        
        print("BigQuery Table Information")
        print("=" * 50)
        
        try:
            table = client.get_table(table_id)
            
            # Get row count
            query = f"SELECT COUNT(*) as row_count FROM `{table_id}`"
            result = client.query(query).result()
            row_count = next(result).row_count
            
            # Get table size info
            print(f"📋 Table: {table.full_table_id}")
            print(f"📊 Rows: {row_count:,}")
            print(f"🗓️  Created: {table.created}")
            print(f"🔄 Modified: {table.modified}")
            print(f"📏 Schema: {len(table.schema)} fields")
            
            if row_count > 0:
                # Get sample of unit and trait names to see if mappings are applied
                sample_query = f"""
                SELECT 
                    ARRAY(SELECT DISTINCT unit.character_id FROM UNNEST(units) AS unit LIMIT 5) as sample_units,
                    ARRAY(SELECT DISTINCT trait.name FROM UNNEST(traits) AS trait LIMIT 5) as sample_traits
                FROM `{table_id}` 
                LIMIT 1
                """
                
                sample_result = client.query(sample_query).result()
                for row in sample_result:
                    print(f"\n🎮 Sample unit names: {row.sample_units}")
                    print(f"⭐ Sample trait names: {row.sample_traits}")
                    
                    # Check if names look mapped (simple heuristic)
                    units_look_mapped = any(not name.startswith('TFT') and '_' not in name for name in row.sample_units if name)
                    traits_look_mapped = any(not name.startswith('TFT') and '_' not in name for name in row.sample_traits if name)
                    
                    if units_look_mapped or traits_look_mapped:
                        print("✅ Data appears to use mapped names (good for querying)")
                    else:
                        print("⚠️  Data appears to use raw API names (consider re-importing with mappings)")
            
        except NotFound:
            print(f"❌ Table not found: {table_id}")
            print("ℹ️  Run data collection to create the table")
        
    except Exception as e:
        print(f"❌ Error getting table info: {e}")

def main():
    """Main script entry point."""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'info':
            get_table_info()
        elif command == 'clear':
            confirm = '--force' not in sys.argv
            keep_tracking = '--keep-tracking' in sys.argv
            clear_bigquery_table(confirm=confirm, clear_tracking=not keep_tracking)
        elif command == 'help':
            print("BigQuery Table Management Script")
            print("=" * 50)
            print("Usage:")
            print("  python clear_bigquery_data.py info                    # Show table information")
            print("  python clear_bigquery_data.py clear                   # Clear table + tracking (with confirmation)")
            print("  python clear_bigquery_data.py clear --force           # Clear table + tracking (no confirmation)")
            print("  python clear_bigquery_data.py clear --keep-tracking   # Clear only table, keep match tracking")
            print("  python clear_bigquery_data.py help                    # Show this help")
            print()
            print("Options:")
            print("  --force          Skip confirmation prompt")
            print("  --keep-tracking  Preserve global_matches_downloaded.json file")
            print()
            print("Note: By default, both BigQuery table AND match tracking are cleared for a complete reset.")
        else:
            print(f"❌ Unknown command: {command}")
            print("Use 'python clear_bigquery_data.py help' for usage information")
    else:
        # Default: show info
        get_table_info()

if __name__ == "__main__":
    main()