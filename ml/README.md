# TFT ML Data Pipeline

Convert BigQuery match data to training-ready format for board strength prediction.

## Quick Start

### 1. Test the Pipeline (10 matches)
```bash
python test_ml_pipeline.py
```

### 2. Generate Training Data
```bash
# Generate data from all matches
python ml/pipeline.py --output tft_training_data.h5

# Or limit to specific number of matches (for testing)
python ml/pipeline.py --limit 1000 --output tft_train_1k.h5
```

### 3. Use with PyTorch
```python
import torch
from ml.pipeline import load_from_hdf5

# Load data
X, Y, metadata = load_from_hdf5('tft_training_data.h5')

# Convert to tensors
X_tensor = torch.from_numpy(X).float()
Y_tensor = torch.from_numpy(Y).long()

# X_tensor shape: (num_matches, match_feature_dim)
# Y_tensor shape: (num_matches, 8)
```

## Data Format

### Input (X)
Each match is a flattened 1D array containing 8 players:
- **Per player**: `[level, unit1, unit2, ..., unit15, traits]`
- **Per unit**: `[unit_onehot, items_binary, tier]`
  - `unit_onehot`: One-hot encoding (60+ units)
  - `items_binary`: Binary encoding (40+ items)
  - `tier`: Star level (1-4)
- **Traits**: Binary encoding (20+ traits with tier_current values)

### Output (Y)
Array of 8 placements (1-8) for each player in the match.

## Pipeline Components

### `vocabulary.py`
Loads vocabularies from CSV mappings:
- `name_mappings/units_mapping.csv`
- `name_mappings/items_mapping.csv`
- `name_mappings/traits_mapping.csv`

### `encoder.py`
Encodes match data:
- `encode_player()`: Encodes one player's board
- `encode_match()`: Encodes complete match (8 players)
- Handles padding for boards with < 15 units

### `data_loader.py`
Fetches matches from BigQuery:
- Filters to complete matches (8 players)
- Converts BigQuery STRUCT format to Python dicts

### `pipeline.py`
Main orchestration:
- Loads vocabulary
- Fetches matches
- Encodes all matches
- Saves to HDF5 with compression

## Feature Dimensions

For typical TFT set:
- **Units**: ~60 (one-hot)
- **Items**: ~40 (binary)
- **Traits**: ~25 (tier values)
- **Player feature dim**: ~1,525 per player
- **Match feature dim**: ~12,200 (8 players)

## Requirements

```bash
pip install h5py numpy google-cloud-bigquery
```

For training (optional):
```bash
pip install torch
```
