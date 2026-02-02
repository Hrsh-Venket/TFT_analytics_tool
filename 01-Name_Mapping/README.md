# TFT Name Mapping System

This directory contains the name mapping system for converting raw TFT API identifiers to clean, readable names.

## Directory Structure

```
01-Name_Mapping/
├── name_mapper.py           # Main mapping module
├── README.md                # This file
├── Latest_Mappings/         # Current mappings (used by default)
│   ├── units.csv
│   ├── traits.csv
│   └── items.csv
└── Default_Mappings/        # Baseline mappings (fallback/reference)
    ├── units_default.csv
    ├── traits_default.csv
    └── items_default.csv
```

## Mapping Files

Each CSV file has the format:
```csv
old_name,new_name
TFT15_Jinx,Jinx
TFT15_DrMundo,Dr._Mundo
```

- **old_name**: Raw identifier from Riot API (e.g., `TFT15_Jinx`)
- **new_name**: Clean display name (e.g., `Jinx`)

## Usage

### Basic Usage

```python
from name_mapper import get_mapper

# Get mapper (uses Latest_Mappings by default)
mapper = get_mapper()

# Map individual names
unit_name = mapper.map_unit_name("TFT15_Jinx")  # Returns: "Jinx"
trait_name = mapper.map_trait_name("TFT15_Vanguard")  # Returns: "Vanguard"
item_name = mapper.map_item_name("TFT_Item_InfinityEdge")  # Returns: "Infinity_Edge"

# Get mapping statistics
stats = mapper.get_mapping_stats()
print(stats)  # {'units': 68, 'traits': 26, 'items': 150, 'mapping_type': 'latest'}
```

### Using Default Mappings

```python
from name_mapper import get_mapper

# Use Default_Mappings instead of Latest_Mappings
mapper = get_mapper(mapping_type="default")
```

### Mapping Complete Match Data

```python
from name_mapper import map_match_data

# Map all names in a match data structure
mapped_match = map_match_data(raw_match_data)

# Or use default mappings
mapped_match = map_match_data(raw_match_data, mapping_type="default")
```

### Initialize Latest from Default

```python
from name_mapper import initialize_latest_from_default

# Copy all default mappings to latest (resets Latest_Mappings)
success = initialize_latest_from_default()
```

### Name Formatting Utilities

```python
from name_mapper import format_name_with_underscores

# Format names with underscore separation
formatted = format_name_with_underscores("DrMundo")  # Returns: "Dr_Mundo"
formatted = format_name_with_underscores("JarvanIV")  # Returns: "Jarvan_IV"
formatted = format_name_with_underscores("Infinity Edge")  # Returns: "Infinity_Edge"
```

## Design Philosophy

### Two-Tier Mapping System

1. **Latest_Mappings**: Working set that can be modified and updated
   - Used by default in production
   - Can be customized for current meta
   - Easily updated without affecting defaults

2. **Default_Mappings**: Reference baseline
   - Stable reference point
   - Can be used for comparisons
   - Source for resetting Latest_Mappings

### Path-Based Import

The module uses `Path(__file__).parent.resolve()` to locate mapping files, ensuring it works correctly when imported from any directory:

```python
# Works from any location
from path.to.name_mapping import name_mapper
```

## Workflow Examples

### Updating Mappings for New TFT Set

```python
from name_mapper import initialize_latest_from_default

# 1. Start fresh from defaults
initialize_latest_from_default()

# 2. Edit Latest_Mappings/*.csv files with new set data
# 3. Test with get_mapper()
```

### Comparing Latest vs Default

```python
from name_mapper import TFTNameMapper

latest = TFTNameMapper(mapping_type="latest")
default = TFTNameMapper(mapping_type="default")

# Compare a specific unit
unit_latest = latest.map_unit_name("TFT15_Jinx")
unit_default = default.map_unit_name("TFT15_Jinx")

if unit_latest != unit_default:
    print(f"Mapping changed: {unit_default} -> {unit_latest}")
```

## Integration with Existing Code

### Backward Compatibility

The new `name_mapper.py` maintains the same API as the original:

```python
# Old code still works:
from name_mapper import get_mapper, map_match_data

mapper = get_mapper()  # Works with new Latest_Mappings location
mapped_data = map_match_data(match_data)  # Works as before
```

### Migration from Root Location

If you have code importing from the root directory:

```python
# Old (from root):
from name_mapper import get_mapper

# New (from 01-Name_Mapping/):
from name_mapping.name_mapper import get_mapper
# OR add 01-Name_Mapping to sys.path
```

## Testing

Run the built-in tests:

```bash
python 01-Name_Mapping/name_mapper.py
```

Expected output:
```
Testing Latest Mappings:
Latest Mapper Statistics: {'units': 68, 'traits': 26, 'items': 150, 'mapping_type': 'latest'}

Testing Default Mappings:
Default Mapper Statistics: {'units': 68, 'traits': 26, 'items': 150, 'mapping_type': 'default'}

Raw 'TFT15_Jinx' -> 'Jinx'
Raw 'TFT15_DrMundo' -> 'Dr._Mundo'

Testing name formatting:
'DrMundo' -> 'Dr_Mondo'
'JarvanIV' -> 'Jarvan_IV'
'Infinity Edge' -> 'Infinity_Edge'
```

## File Formats

### Units CSV
```csv
old_name,new_name
TFT15_Aatrox,Aatrox
TFT15_Ahri,Ahri
TFT15_DrMundo,Dr._Mundo
```

### Traits CSV
```csv
old_name,new_name
TFT15_Vanguard,Vanguard
TFT15_Bruiser,Bruiser
```

### Items CSV
```csv
old_name,new_name
TFT_Item_InfinityEdge,Infinity_Edge
TFT_Item_GuardianAngel,Guardian_Angel
```

## Notes

- All CSV files use UTF-8 encoding
- Names are case-sensitive
- Unmapped names return the original raw name
- The system gracefully handles missing mapping files (logs warnings)
- Both `old_name,new_name` and `raw_name,clean_name` column headers are supported
