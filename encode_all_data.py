#!/usr/bin/env python3
"""
Encode ALL BigQuery TFT data into train/validation/test splits.

This script loads ALL matches from BigQuery, encodes them, and saves
them as HDF5 files with 80/10/10 train/val/test splits.

Usage:
    python encode_all_data.py --output tft_data_splits.h5
    python encode_all_data.py --output tft_data_splits.h5 --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
"""

import argparse
import sys
sys.path.insert(0, '.')

from ml.pipeline import (
    process_matches_to_arrays,
    create_train_val_test_splits,
    save_splits_to_hdf5
)


def main():
    parser = argparse.ArgumentParser(
        description='Encode ALL BigQuery TFT data with train/val/test splits',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Encode all data with default 80/10/10 split
  python encode_all_data.py --output tft_data_splits.h5

  # Custom split ratios
  python encode_all_data.py --output tft_data_splits.h5 --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15

  # Encode specific number of matches (for testing)
  python encode_all_data.py --output test_splits.h5 --limit 1000

  # No shuffle (keep chronological order)
  python encode_all_data.py --output tft_data_splits.h5 --no-shuffle
        '''
    )

    # Input/Output
    parser.add_argument('--output', type=str, required=True,
                       help='Output HDF5 file for splits (e.g., tft_data_splits.h5)')
    parser.add_argument('--project-id', type=str, default=None,
                       help='GCP project ID (auto-detected if not provided)')
    parser.add_argument('--dataset-id', type=str, default='tft_analytics',
                       help='BigQuery dataset ID')

    # Data selection
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of matches to encode (None = ALL data)')
    parser.add_argument('--max-units', type=int, default=15,
                       help='Maximum units per board (for padding)')
    parser.add_argument('--mappings-dir', type=str, default='name_mappings',
                       help='Directory containing CSV mappings')

    # Split configuration
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training set ratio (default: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Test set ratio (default: 0.1)')
    parser.add_argument('--no-shuffle', action='store_true',
                       help='Do not shuffle data before splitting (keep chronological)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for shuffling (default: 42)')

    args = parser.parse_args()

    # Validate split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not (0.99 <= total_ratio <= 1.01):  # Allow small floating point errors
        print(f"❌ Error: Split ratios must sum to 1.0, got {total_ratio}")
        return 1

    print("=" * 80)
    print("ENCODE ALL BIGQUERY DATA WITH TRAIN/VAL/TEST SPLITS")
    print("=" * 80)
    print()
    print(f"Configuration:")
    print(f"  Output file: {args.output}")
    print(f"  BigQuery dataset: {args.dataset_id}")
    print(f"  Match limit: {'ALL DATA' if args.limit is None else f'{args.limit} matches'}")
    print(f"  Split ratios: Train={args.train_ratio}, Val={args.val_ratio}, Test={args.test_ratio}")
    print(f"  Shuffle: {'No (chronological)' if args.no_shuffle else f'Yes (seed={args.random_seed})'}")
    print()

    if args.limit is None:
        print("⚠️  WARNING: Encoding ALL data from BigQuery")
        print("   This may take a while depending on dataset size...")
        print()

    try:
        # Step 1: Load and encode all matches from BigQuery
        print("=" * 80)
        print("STEP 1: LOAD AND ENCODE MATCHES")
        print("=" * 80)

        X, Y, encoder = process_matches_to_arrays(
            project_id=args.project_id,
            dataset_id=args.dataset_id,
            limit=args.limit,
            max_units=args.max_units,
            mappings_dir=args.mappings_dir
        )

        if len(X) == 0:
            print("❌ Error: No matches were encoded!")
            return 1

        # Step 2: Create train/val/test splits
        print()
        print("=" * 80)
        print("STEP 2: CREATE TRAIN/VAL/TEST SPLITS")
        print("=" * 80)

        X_train, Y_train, X_val, Y_val, X_test, Y_test = create_train_val_test_splits(
            X, Y,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            shuffle=not args.no_shuffle,
            random_seed=args.random_seed
        )

        # Step 3: Save to HDF5
        print()
        print("=" * 80)
        print("STEP 3: SAVE TO HDF5")
        print("=" * 80)

        metadata = {
            'num_matches_total': X.shape[0],
            'num_matches_train': X_train.shape[0],
            'num_matches_val': X_val.shape[0],
            'num_matches_test': X_test.shape[0],
            'match_feature_dim': X.shape[1],
            'max_units': args.max_units,
            'train_ratio': args.train_ratio,
            'val_ratio': args.val_ratio,
            'test_ratio': args.test_ratio,
            'shuffled': not args.no_shuffle,
            'random_seed': args.random_seed if not args.no_shuffle else 'N/A',
            **encoder.get_feature_info()
        }

        save_splits_to_hdf5(
            X_train, Y_train,
            X_val, Y_val,
            X_test, Y_test,
            args.output,
            metadata
        )

        # Summary
        print()
        print("=" * 80)
        print("✓ SUCCESS - DATA ENCODING COMPLETE")
        print("=" * 80)
        print()
        print(f"Total matches encoded: {len(X):,}")
        print(f"  Train: {len(X_train):,} matches ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Val:   {len(X_val):,} matches ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test:  {len(X_test):,} matches ({len(X_test)/len(X)*100:.1f}%)")
        print()
        print(f"Saved to: {args.output}")
        print()

        # PyTorch usage example
        print("=" * 80)
        print("USAGE WITH PYTORCH")
        print("=" * 80)
        print(f"""
import torch
from ml.pipeline import load_splits_from_hdf5

# Load splits
X_train, Y_train, X_val, Y_val, X_test, Y_test, metadata = load_splits_from_hdf5('{args.output}')

# Convert to PyTorch tensors
train_X = torch.from_numpy(X_train).float()
train_Y = torch.from_numpy(Y_train).long()

val_X = torch.from_numpy(X_val).float()
val_Y = torch.from_numpy(Y_val).long()

test_X = torch.from_numpy(X_test).float()
test_Y = torch.from_numpy(Y_test).long()

# Create DataLoaders
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(train_X, train_Y)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TensorDataset(val_X, val_Y)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_dataset = TensorDataset(test_X, test_Y)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Train your model
for epoch in range(num_epochs):
    for batch_X, batch_Y in train_loader:
        # Your training code here
        pass
""")
        print("=" * 80)

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Encoding interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
