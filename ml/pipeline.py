"""
Main ML data pipeline for TFT.
Converts BigQuery match data to HDF5 training format.
"""

import numpy as np
import h5py
import argparse
from typing import Optional
from ml.vocabulary import TFTVocabulary
from ml.encoder import TFTMatchEncoder
from ml.data_loader import load_matches_from_bigquery


def process_matches_to_arrays(
    project_id: Optional[str] = None,
    dataset_id: str = 'tft_analytics',
    limit: Optional[int] = None,
    max_units: int = 15,
    mappings_dir: str = 'name_mappings'
):
    """
    Load matches from BigQuery and encode to X, Y arrays.

    Args:
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        limit: Optional limit on number of matches
        max_units: Maximum units per board
        mappings_dir: Directory with CSV mappings

    Returns:
        Tuple of (X, Y, encoder) where:
            X: numpy array of shape (num_matches, match_feature_dim)
            Y: numpy array of shape (num_matches, 8)
            encoder: TFTMatchEncoder instance (for feature info)
    """
    print("="*80)
    print("TFT ML DATA PIPELINE")
    print("="*80)

    # Step 1: Load vocabulary
    print("\n[1/4] Loading vocabulary...")
    vocab = TFTVocabulary(mappings_dir=mappings_dir)
    stats = vocab.get_stats()
    print(f"  Units: {stats['units']}")
    print(f"  Items: {stats['items']}")
    print(f"  Traits: {stats['traits']}")

    # Step 2: Initialize encoder
    print("\n[2/4] Initializing encoder...")
    encoder = TFTMatchEncoder(vocabulary=vocab, max_units=max_units)
    feature_info = encoder.get_feature_info()
    print(f"  Player feature dim: {feature_info['player_feature_dim']}")
    print(f"  Match feature dim: {feature_info['match_feature_dim']}")

    # Step 3: Load matches from BigQuery
    print("\n[3/4] Loading matches from BigQuery...")
    matches = load_matches_from_bigquery(
        project_id=project_id,
        dataset_id=dataset_id,
        limit=limit
    )

    if not matches:
        raise ValueError("No matches loaded!")

    # Step 4: Encode all matches
    print(f"\n[4/4] Encoding {len(matches)} matches...")
    X_list = []
    Y_list = []

    for i, match in enumerate(matches):
        try:
            X, Y = encoder.encode_match(match)
            X_list.append(X)
            Y_list.append(Y)

            if (i + 1) % 100 == 0:
                print(f"  Encoded {i + 1}/{len(matches)} matches...")
        except Exception as e:
            print(f"  Warning: Failed to encode match {match['match_id']}: {e}")
            continue

    # Convert to numpy arrays
    X = np.array(X_list, dtype=np.int32)
    Y = np.array(Y_list, dtype=np.int32)

    print(f"\n✓ Pipeline complete!")
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")

    return X, Y, encoder


def create_train_val_test_splits(X: np.ndarray, Y: np.ndarray,
                                  train_ratio: float = 0.8,
                                  val_ratio: float = 0.1,
                                  test_ratio: float = 0.1,
                                  shuffle: bool = True,
                                  random_seed: int = 42):
    """
    Split data into train, validation, and test sets.

    Args:
        X: Features array
        Y: Labels array
        train_ratio: Ratio for training set (default: 0.8)
        val_ratio: Ratio for validation set (default: 0.1)
        test_ratio: Ratio for test set (default: 0.1)
        shuffle: Whether to shuffle before splitting
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (X_train, Y_train, X_val, Y_val, X_test, Y_test)
    """
    # Validate ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    n_samples = len(X)
    indices = np.arange(n_samples)

    # Shuffle if requested
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    # Calculate split points
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    # Split indices
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    # Split data
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    print(f"\n✓ Split data into train/val/test:")
    print(f"  Train: {len(X_train)} samples ({len(X_train)/n_samples*100:.1f}%)")
    print(f"  Val:   {len(X_val)} samples ({len(X_val)/n_samples*100:.1f}%)")
    print(f"  Test:  {len(X_test)} samples ({len(X_test)/n_samples*100:.1f}%)")

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def save_to_hdf5(X: np.ndarray, Y: np.ndarray, output_file: str, metadata: dict = None):
    """
    Save X and Y arrays to HDF5 file.

    Args:
        X: Features array
        Y: Labels array
        output_file: Output HDF5 file path
        metadata: Optional metadata dict to save
    """
    print(f"\nSaving to HDF5: {output_file}")

    with h5py.File(output_file, 'w') as f:
        # Save arrays
        f.create_dataset('X', data=X, compression='gzip', compression_opts=4)
        f.create_dataset('Y', data=Y, compression='gzip', compression_opts=4)

        # Save metadata as attributes
        if metadata:
            for key, value in metadata.items():
                f.attrs[key] = value

    print(f"✓ Saved to {output_file}")
    print(f"  X: {X.shape}, dtype: {X.dtype}")
    print(f"  Y: {Y.shape}, dtype: {Y.dtype}")


def save_splits_to_hdf5(X_train, Y_train, X_val, Y_val, X_test, Y_test,
                        output_file: str, metadata: dict = None):
    """
    Save train/val/test splits to a single HDF5 file.

    Args:
        X_train, Y_train: Training data
        X_val, Y_val: Validation data
        X_test, Y_test: Test data
        output_file: Output HDF5 file path
        metadata: Optional metadata dict to save
    """
    print(f"\nSaving splits to HDF5: {output_file}")

    with h5py.File(output_file, 'w') as f:
        # Create groups for each split
        train_group = f.create_group('train')
        val_group = f.create_group('val')
        test_group = f.create_group('test')

        # Save training data
        train_group.create_dataset('X', data=X_train, compression='gzip', compression_opts=4)
        train_group.create_dataset('Y', data=Y_train, compression='gzip', compression_opts=4)

        # Save validation data
        val_group.create_dataset('X', data=X_val, compression='gzip', compression_opts=4)
        val_group.create_dataset('Y', data=Y_val, compression='gzip', compression_opts=4)

        # Save test data
        test_group.create_dataset('X', data=X_test, compression='gzip', compression_opts=4)
        test_group.create_dataset('Y', data=Y_test, compression='gzip', compression_opts=4)

        # Save metadata as attributes
        if metadata:
            for key, value in metadata.items():
                f.attrs[key] = value

    print(f"✓ Saved splits to {output_file}")
    print(f"  Train: X{X_train.shape}, Y{Y_train.shape}")
    print(f"  Val:   X{X_val.shape}, Y{Y_val.shape}")
    print(f"  Test:  X{X_test.shape}, Y{Y_test.shape}")


def load_from_hdf5(input_file: str):
    """
    Load X and Y arrays from HDF5 file.
    Can be easily converted to PyTorch tensors.

    Args:
        input_file: Input HDF5 file path

    Returns:
        Tuple of (X, Y, metadata)

    Usage with PyTorch:
        X, Y, metadata = load_from_hdf5('tft_data.h5')
        X_tensor = torch.from_numpy(X).float()
        Y_tensor = torch.from_numpy(Y).long()
    """
    print(f"Loading from HDF5: {input_file}")

    with h5py.File(input_file, 'r') as f:
        X = f['X'][:]
        Y = f['Y'][:]

        # Load metadata
        metadata = dict(f.attrs)

    print(f"✓ Loaded from {input_file}")
    print(f"  X: {X.shape}, dtype: {X.dtype}")
    print(f"  Y: {Y.shape}, dtype: {Y.dtype}")

    return X, Y, metadata


def load_splits_from_hdf5(input_file: str):
    """
    Load train/val/test splits from HDF5 file.

    Args:
        input_file: Input HDF5 file path

    Returns:
        Tuple of (X_train, Y_train, X_val, Y_val, X_test, Y_test, metadata)

    Usage with PyTorch:
        X_train, Y_train, X_val, Y_val, X_test, Y_test, metadata = load_splits_from_hdf5('tft_splits.h5')

        train_X = torch.from_numpy(X_train).float()
        train_Y = torch.from_numpy(Y_train).long()

        val_X = torch.from_numpy(X_val).float()
        val_Y = torch.from_numpy(Y_val).long()

        test_X = torch.from_numpy(X_test).float()
        test_Y = torch.from_numpy(Y_test).long()
    """
    print(f"Loading splits from HDF5: {input_file}")

    with h5py.File(input_file, 'r') as f:
        # Load training data
        X_train = f['train/X'][:]
        Y_train = f['train/Y'][:]

        # Load validation data
        X_val = f['val/X'][:]
        Y_val = f['val/Y'][:]

        # Load test data
        X_test = f['test/X'][:]
        Y_test = f['test/Y'][:]

        # Load metadata
        metadata = dict(f.attrs)

    print(f"✓ Loaded splits from {input_file}")
    print(f"  Train: X{X_train.shape}, Y{Y_train.shape}")
    print(f"  Val:   X{X_val.shape}, Y{Y_val.shape}")
    print(f"  Test:  X{X_test.shape}, Y{Y_test.shape}")

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, metadata


def main():
    parser = argparse.ArgumentParser(
        description='TFT ML Data Pipeline - Convert BigQuery data to HDF5 training format'
    )
    parser.add_argument('--project-id', type=str, default=None,
                       help='GCP project ID (auto-detected if not provided)')
    parser.add_argument('--dataset-id', type=str, default='tft_analytics',
                       help='BigQuery dataset ID')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of matches (for testing)')
    parser.add_argument('--max-units', type=int, default=15,
                       help='Maximum units per board (for padding)')
    parser.add_argument('--mappings-dir', type=str, default='name_mappings',
                       help='Directory containing CSV mappings')
    parser.add_argument('--output', type=str, default='tft_training_data.h5',
                       help='Output HDF5 file')

    args = parser.parse_args()

    try:
        # Process matches
        X, Y, encoder = process_matches_to_arrays(
            project_id=args.project_id,
            dataset_id=args.dataset_id,
            limit=args.limit,
            max_units=args.max_units,
            mappings_dir=args.mappings_dir
        )

        # Save to HDF5
        metadata = {
            'num_matches': X.shape[0],
            'match_feature_dim': X.shape[1],
            'max_units': args.max_units,
            **encoder.get_feature_info()
        }

        save_to_hdf5(X, Y, args.output, metadata)

        # Test loading
        print("\nTesting HDF5 load...")
        X_loaded, Y_loaded, metadata_loaded = load_from_hdf5(args.output)
        print("\n✓ Pipeline complete! Data saved and verified.")

        print("\n" + "="*80)
        print("USAGE WITH PYTORCH")
        print("="*80)
        print("import torch")
        print(f"from ml.pipeline import load_from_hdf5")
        print()
        print(f"X, Y, metadata = load_from_hdf5('{args.output}')")
        print("X_tensor = torch.from_numpy(X).float()")
        print("Y_tensor = torch.from_numpy(Y).long()")
        print()
        print("# X_tensor shape:", X.shape)
        print("# Y_tensor shape:", Y.shape)
        print("="*80)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
