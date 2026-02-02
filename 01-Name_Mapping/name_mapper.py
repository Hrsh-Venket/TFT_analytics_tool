"""
Name Mapping Module for TFT Analytics

This module handles loading and applying name mappings for units, traits, and items
to convert raw API identifiers into clean, readable names for BigQuery storage.

Supports both Latest_Mappings and Default_Mappings directories.

Can initialize Latest_Mappings from collected match data (subset.json).
"""

import csv
import json
import logging
import re
import shutil
from typing import Dict, Optional, Literal, Set, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURABLE PREFIXES TO REMOVE
# Add or remove prefixes here as needed for new TFT sets
# =============================================================================
PREFIXES_TO_REMOVE = [
    "TFT14_",
    "TFT15_",
    "TFT16_",
    "TFT17_",
    "TFT_",
]

# Pattern to match trailing _XX number suffixes (e.g., _34, _123)
SUFFIX_PATTERN = re.compile(r'_\d+$')

# Get the directory where this file is located
MODULE_DIR = Path(__file__).parent.resolve()
LATEST_MAPPINGS_DIR = MODULE_DIR / "Latest_Mappings"
DEFAULT_MAPPINGS_DIR = MODULE_DIR / "Default_Mappings"

MappingType = Literal["latest", "default"]


class TFTNameMapper:
    """
    Handles loading and applying name mappings for TFT data.

    Converts raw API identifiers (like TFT14_Jinx_34) into clean names (like Jinx).
    Supports both Latest_Mappings and Default_Mappings directories.
    """

    def __init__(self, mapping_type: MappingType = "latest"):
        """
        Initialize the name mapper with mapping files from the specified directory.

        Args:
            mapping_type: Either "latest" or "default" to specify which mapping set to use
        """
        self.mapping_type = mapping_type
        self.mappings_dir = LATEST_MAPPINGS_DIR if mapping_type == "latest" else DEFAULT_MAPPINGS_DIR
        self.units_mapping: Dict[str, str] = {}
        self.traits_mapping: Dict[str, str] = {}
        self.items_mapping: Dict[str, str] = {}

        # Load all mappings
        self._load_mappings()

    def _load_mappings(self):
        """Load all mapping files from the mappings directory."""
        try:
            # Determine file names based on mapping type
            if self.mapping_type == "latest":
                units_file = self.mappings_dir / "units.csv"
                traits_file = self.mappings_dir / "traits.csv"
                items_file = self.mappings_dir / "items.csv"
            else:  # default
                units_file = self.mappings_dir / "units_default.csv"
                traits_file = self.mappings_dir / "traits_default.csv"
                items_file = self.mappings_dir / "items_default.csv"

            # Load units mapping
            if units_file.exists():
                self.units_mapping = self._load_mapping_file(units_file)
                logger.info(f"Loaded {len(self.units_mapping)} unit mappings from {units_file}")
            else:
                logger.warning(f"Units mapping file not found: {units_file}")

            # Load traits mapping
            if traits_file.exists():
                self.traits_mapping = self._load_mapping_file(traits_file)
                logger.info(f"Loaded {len(self.traits_mapping)} trait mappings from {traits_file}")
            else:
                logger.warning(f"Traits mapping file not found: {traits_file}")

            # Load items mapping
            if items_file.exists():
                self.items_mapping = self._load_mapping_file(items_file)
                logger.info(f"Loaded {len(self.items_mapping)} item mappings from {items_file}")
            else:
                logger.warning(f"Items mapping file not found: {items_file}")

        except Exception as e:
            logger.error(f"Error loading mappings: {e}")

    def _load_mapping_file(self, file_path: Path) -> Dict[str, str]:
        """
        Load a single mapping CSV file.

        Args:
            file_path: Path to the CSV mapping file

        Returns:
            Dictionary mapping raw names to clean names
        """
        mapping = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Expected format: old_name,new_name (or raw_name,clean_name)
                    raw_col = 'old_name' if 'old_name' in row else 'raw_name'
                    clean_col = 'new_name' if 'new_name' in row else 'clean_name'

                    if raw_col in row and clean_col in row:
                        raw_name = row[raw_col].strip()
                        clean_name = row[clean_col].strip()
                        if raw_name and clean_name:
                            mapping[raw_name] = clean_name
                    else:
                        # Try alternative: use first two columns
                        keys = list(row.keys())
                        if len(keys) >= 2:
                            raw_name = row[keys[0]].strip()
                            clean_name = row[keys[1]].strip()
                            if raw_name and clean_name:
                                mapping[raw_name] = clean_name
        except Exception as e:
            logger.error(f"Error loading mapping file {file_path}: {e}")

        return mapping

    def map_unit_name(self, raw_name: str) -> str:
        """
        Map a raw unit name to a clean name.

        Args:
            raw_name: Raw unit identifier from API (e.g., 'TFT14_Jinx_34')

        Returns:
            Clean unit name (e.g., 'Jinx') or original name if no mapping found
        """
        if not raw_name:
            return raw_name

        # Try exact match first
        if raw_name in self.units_mapping:
            return self.units_mapping[raw_name]

        # If no exact match, return original name
        return raw_name

    def map_trait_name(self, raw_name: str) -> str:
        """
        Map a raw trait name to a clean name.

        Args:
            raw_name: Raw trait identifier from API (e.g., 'TFT14_Vanguard')

        Returns:
            Clean trait name (e.g., 'Vanguard') or original name if no mapping found
        """
        if not raw_name:
            return raw_name

        # Try exact match first
        if raw_name in self.traits_mapping:
            return self.traits_mapping[raw_name]

        # If no exact match, return original name
        return raw_name

    def map_item_name(self, raw_name: str) -> str:
        """
        Map a raw item name to a clean name.

        Args:
            raw_name: Raw item identifier from API

        Returns:
            Clean item name or original name if no mapping found
        """
        if not raw_name:
            return raw_name

        # Try exact match first
        if raw_name in self.items_mapping:
            return self.items_mapping[raw_name]

        # If no exact match, return original name
        return raw_name

    def map_unit_data(self, unit_data: Dict) -> Dict:
        """
        Apply mappings to a complete unit data structure.

        Args:
            unit_data: Unit dictionary from API response

        Returns:
            Unit dictionary with mapped names
        """
        if not isinstance(unit_data, dict):
            return unit_data

        # Create a copy to avoid modifying original
        mapped_unit = unit_data.copy()

        # Map character_id and create name field
        # The Riot API provides character_id (e.g., "TFT15_Jinx") but not a name field
        # We need to create the name field from the mapped character_id for BigQuery
        if 'character_id' in mapped_unit:
            mapped_name = self.map_unit_name(mapped_unit['character_id'])
            mapped_unit['name'] = mapped_name  # Create name field for BigQuery
            # Keep character_id as the raw value for reference

        # Map item names - handle both itemNames (API format) and item_names (snake_case)
        # Always store in item_names field with mapped values
        if 'itemNames' in mapped_unit and isinstance(mapped_unit['itemNames'], list):
            # Map items and store in item_names field
            mapped_unit['item_names'] = [
                self.map_item_name(item) for item in mapped_unit['itemNames']
            ]
            # Remove the camelCase field to avoid confusion
            del mapped_unit['itemNames']
        elif 'item_names' in mapped_unit and isinstance(mapped_unit['item_names'], list):
            # Already in snake_case format, just map the values
            mapped_unit['item_names'] = [
                self.map_item_name(item) for item in mapped_unit['item_names']
            ]

        # If name field already exists (shouldn't happen with Riot API), map it
        elif 'name' in mapped_unit:
            mapped_unit['name'] = self.map_unit_name(mapped_unit['name'])

        return mapped_unit

    def map_trait_data(self, trait_data: Dict) -> Dict:
        """
        Apply mappings to a complete trait data structure.

        Args:
            trait_data: Trait dictionary from API response

        Returns:
            Trait dictionary with mapped names
        """
        if not isinstance(trait_data, dict):
            return trait_data

        # Create a copy to avoid modifying original
        mapped_trait = trait_data.copy()

        # Map trait name
        if 'name' in mapped_trait:
            mapped_trait['name'] = self.map_trait_name(mapped_trait['name'])

        return mapped_trait

    def get_mapping_stats(self) -> Dict[str, int]:
        """
        Get statistics about loaded mappings.

        Returns:
            Dictionary with mapping counts
        """
        return {
            'units': len(self.units_mapping),
            'traits': len(self.traits_mapping),
            'items': len(self.items_mapping),
            'mapping_type': self.mapping_type
        }


# Utility functions for name formatting
def format_name_with_underscores(name: str) -> str:
    """
    Format a name with underscores separating words.

    Handles cases like:
    - "DrMundo" -> "Dr_Mundo"
    - "JarvanIV" -> "Jarvan_IV"
    - "Infinity Edge" -> "Infinity_Edge"

    Args:
        name: Name to format

    Returns:
        Formatted name with underscores
    """
    if not name:
        return name

    # Replace spaces with underscores
    formatted = name.replace(' ', '_')

    # Insert underscores before capital letters (but not at the start)
    result = []
    for i, char in enumerate(formatted):
        # Add underscore before capitals (except first char and after underscore)
        if i > 0 and char.isupper() and formatted[i-1] != '_' and not formatted[i-1].isupper():
            result.append('_')
        result.append(char)

    return ''.join(result)


def initialize_latest_from_default():
    """
    Initialize Latest_Mappings from Default_Mappings.

    Copies all default mapping files to the latest directory.
    Useful for starting a new set with fresh data or resetting to defaults.
    """
    try:
        # Ensure Latest_Mappings directory exists
        LATEST_MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)

        # Copy each default file to latest
        default_files = {
            'units_default.csv': 'units.csv',
            'traits_default.csv': 'traits.csv',
            'items_default.csv': 'items.csv'
        }

        for default_file, latest_file in default_files.items():
            default_path = DEFAULT_MAPPINGS_DIR / default_file
            latest_path = LATEST_MAPPINGS_DIR / latest_file

            if default_path.exists():
                shutil.copy2(default_path, latest_path)
                logger.info(f"Copied {default_path} to {latest_path}")
            else:
                logger.warning(f"Default file not found: {default_path}")

        logger.info("Successfully initialized Latest_Mappings from Default_Mappings")
        return True

    except Exception as e:
        logger.error(f"Error initializing Latest_Mappings from defaults: {e}")
        return False


# Global mapper instance (will be initialized on first import)
_global_mapper: Optional[TFTNameMapper] = None


def get_mapper(mapping_type: MappingType = "latest") -> TFTNameMapper:
    """
    Get the global name mapper instance.

    Args:
        mapping_type: Either "latest" or "default" to specify which mapping set to use

    Returns:
        Singleton TFTNameMapper instance
    """
    global _global_mapper
    if _global_mapper is None or _global_mapper.mapping_type != mapping_type:
        _global_mapper = TFTNameMapper(mapping_type=mapping_type)
    return _global_mapper


def map_match_data(match_data: Dict, mapping_type: MappingType = "latest") -> Dict:
    """
    Apply name mappings to a complete match data structure.

    Args:
        match_data: Complete match data from API
        mapping_type: Either "latest" or "default" to specify which mapping set to use

    Returns:
        Match data with all names mapped
    """
    mapper = get_mapper(mapping_type)

    if not isinstance(match_data, dict) or 'info' not in match_data:
        return match_data

    # Create a copy to avoid modifying original
    mapped_match = match_data.copy()
    mapped_match['info'] = match_data['info'].copy()

    # Map participant data
    if 'participants' in mapped_match['info']:
        mapped_participants = []
        for participant in mapped_match['info']['participants']:
            if not isinstance(participant, dict):
                mapped_participants.append(participant)
                continue

            mapped_participant = participant.copy()

            # Map units
            if 'units' in mapped_participant and isinstance(mapped_participant['units'], list):
                mapped_participant['units'] = [
                    mapper.map_unit_data(unit) for unit in mapped_participant['units']
                ]

            # Map traits
            if 'traits' in mapped_participant and isinstance(mapped_participant['traits'], list):
                mapped_participant['traits'] = [
                    mapper.map_trait_data(trait) for trait in mapped_participant['traits']
                ]

            mapped_participants.append(mapped_participant)

        mapped_match['info']['participants'] = mapped_participants

    return mapped_match


# =============================================================================
# SUBSET.JSON INITIALIZATION FUNCTIONS
# =============================================================================

SUBSET_FILE = MODULE_DIR / "subset.json"


def clean_raw_name(raw_name: str) -> str:
    """
    Clean a raw API name by removing prefixes and suffixes.

    Removes:
    - Prefixes like TFT16_, TFT15_, etc. (configurable in PREFIXES_TO_REMOVE)
    - Trailing number suffixes like _34, _123

    Args:
        raw_name: Raw name from API (e.g., 'TFT16_Jinx_34')

    Returns:
        Cleaned name (e.g., 'Jinx')
    """
    if not raw_name:
        return raw_name

    cleaned = raw_name

    # Remove prefixes (try longest first to avoid partial matches)
    for prefix in sorted(PREFIXES_TO_REMOVE, key=len, reverse=True):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break

    # Remove trailing number suffix (e.g., _34)
    cleaned = SUFFIX_PATTERN.sub('', cleaned)

    return cleaned


def load_subset_json() -> dict:
    """
    Load the subset.json file.

    Returns:
        Parsed JSON data or empty dict if file not found
    """
    if not SUBSET_FILE.exists():
        logger.warning(f"subset.json not found at {SUBSET_FILE}")
        return {}

    with open(SUBSET_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_unique_names_from_subset(subset_data: dict) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Extract all unique unit, trait, and item names from subset data.

    Args:
        subset_data: Parsed subset.json data

    Returns:
        Tuple of (units_set, traits_set, items_set)
    """
    units = set()
    traits = set()
    items = set()

    matches = subset_data.get('matches', [])

    for match in matches:
        for participant in match.get('participants', []):
            # Extract units
            for unit in participant.get('units', []):
                char_id = unit.get('character_id', '')
                if char_id:
                    units.add(char_id)

                # Extract items from units
                for item in unit.get('item_names', []):
                    if item:
                        items.add(item)

            # Extract traits
            for trait in participant.get('traits', []):
                trait_name = trait.get('name', '')
                if trait_name:
                    traits.add(trait_name)

    return units, traits, items


def load_default_mappings() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Load all default mappings.

    Returns:
        Tuple of (units_mapping, traits_mapping, items_mapping)
    """
    default_mapper = TFTNameMapper(mapping_type="default")
    return (
        default_mapper.units_mapping,
        default_mapper.traits_mapping,
        default_mapper.items_mapping
    )


def write_mapping_csv(file_path: Path, mapping: Dict[str, str]):
    """
    Write a mapping dictionary to CSV file.

    Args:
        file_path: Path to output CSV
        mapping: Dictionary of old_name -> new_name
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['old_name', 'new_name'])

        for old_name in sorted(mapping.keys()):
            writer.writerow([old_name, mapping[old_name]])


def initialize_latest_from_subset(dry_run: bool = False) -> bool:
    """
    Initialize Latest_Mappings from subset.json.

    Process:
    1. Load subset.json and extract unique units/traits/items
    2. Create initial mappings by cleaning names (remove prefixes/suffixes)
    3. Apply any overrides from Default_Mappings
    4. Write to Latest_Mappings CSVs

    Args:
        dry_run: If True, only print what would be done without writing files

    Returns:
        True if successful, False otherwise
    """
    print("Initializing Latest_Mappings from subset.json...")
    print(f"Prefixes to remove: {PREFIXES_TO_REMOVE}")
    print()

    # Step 1: Load subset.json
    subset_data = load_subset_json()
    if not subset_data:
        print("Error: Could not load subset.json")
        print(f"Expected location: {SUBSET_FILE}")
        print("Run collect_subset.py first to generate this file.")
        return False

    print(f"Loaded {subset_data.get('num_matches', 0)} matches from subset.json")
    print(f"Collected at: {subset_data.get('collected_at', 'unknown')}")
    print()

    # Step 2: Extract unique names
    units, traits, items = extract_unique_names_from_subset(subset_data)
    print(f"Found unique entries:")
    print(f"  Units:  {len(units)}")
    print(f"  Traits: {len(traits)}")
    print(f"  Items:  {len(items)}")
    print()

    # Step 3: Create initial mappings by cleaning names
    units_mapping = {name: clean_raw_name(name) for name in units}
    traits_mapping = {name: clean_raw_name(name) for name in traits}
    items_mapping = {name: clean_raw_name(name) for name in items}

    # Step 4: Load and apply Default_Mappings overrides
    default_units, default_traits, default_items = load_default_mappings()

    overrides_applied = {'units': 0, 'traits': 0, 'items': 0}

    for old_name in units_mapping:
        if old_name in default_units:
            units_mapping[old_name] = default_units[old_name]
            overrides_applied['units'] += 1

    for old_name in traits_mapping:
        if old_name in default_traits:
            traits_mapping[old_name] = default_traits[old_name]
            overrides_applied['traits'] += 1

    for old_name in items_mapping:
        if old_name in default_items:
            items_mapping[old_name] = default_items[old_name]
            overrides_applied['items'] += 1

    print(f"Applied Default_Mappings overrides:")
    print(f"  Units:  {overrides_applied['units']}")
    print(f"  Traits: {overrides_applied['traits']}")
    print(f"  Items:  {overrides_applied['items']}")
    print()

    # Step 5: Write to Latest_Mappings
    if dry_run:
        print("DRY RUN - Would write to:")
        print(f"  {LATEST_MAPPINGS_DIR / 'units.csv'}")
        print(f"  {LATEST_MAPPINGS_DIR / 'traits.csv'}")
        print(f"  {LATEST_MAPPINGS_DIR / 'items.csv'}")
        print()
        print("Sample mappings:")
        print("  Units:")
        for old, new in list(units_mapping.items())[:5]:
            print(f"    {old} -> {new}")
        print("  Traits:")
        for old, new in list(traits_mapping.items())[:5]:
            print(f"    {old} -> {new}")
        print("  Items:")
        for old, new in list(items_mapping.items())[:5]:
            print(f"    {old} -> {new}")
    else:
        LATEST_MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)

        write_mapping_csv(LATEST_MAPPINGS_DIR / 'units.csv', units_mapping)
        write_mapping_csv(LATEST_MAPPINGS_DIR / 'traits.csv', traits_mapping)
        write_mapping_csv(LATEST_MAPPINGS_DIR / 'items.csv', items_mapping)

        print(f"Written to {LATEST_MAPPINGS_DIR}:")
        print(f"  units.csv:  {len(units_mapping)} entries")
        print(f"  traits.csv: {len(traits_mapping)} entries")
        print(f"  items.csv:  {len(items_mapping)} entries")

    print()
    print("Initialization complete!")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='TFT Name Mapper')
    parser.add_argument('--init-from-subset', action='store_true',
                        help='Initialize Latest_Mappings from subset.json')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without writing files')
    parser.add_argument('--test', action='store_true',
                        help='Run basic mapping tests')

    args = parser.parse_args()

    if args.init_from_subset:
        initialize_latest_from_subset(dry_run=args.dry_run)
    elif args.test:
        # Test the mapper with both mapping types
        print("Testing Latest Mappings:")
        latest_mapper = TFTNameMapper(mapping_type="latest")
        stats = latest_mapper.get_mapping_stats()
        print(f"Latest Mapper Statistics: {stats}")

        print("\nTesting Default Mappings:")
        default_mapper = TFTNameMapper(mapping_type="default")
        stats = default_mapper.get_mapping_stats()
        print(f"Default Mapper Statistics: {stats}")

        # Test some mappings
        print(f"\nRaw 'TFT15_Jinx' -> '{latest_mapper.map_unit_name('TFT15_Jinx')}'")
        print(f"Raw 'TFT15_DrMundo' -> '{latest_mapper.map_unit_name('TFT15_DrMundo')}'")

        # Test name cleaning
        print("\nTesting clean_raw_name():")
        print(f"'TFT16_Jinx_34' -> '{clean_raw_name('TFT16_Jinx_34')}'")
        print(f"'TFT15_DrMundo' -> '{clean_raw_name('TFT15_DrMundo')}'")
        print(f"'TFT_Item_InfinityEdge' -> '{clean_raw_name('TFT_Item_InfinityEdge')}'")
    else:
        parser.print_help()
