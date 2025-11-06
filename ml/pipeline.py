"""
Main ML data pipeline for TFT.
Converts BigQuery match data to HDF5 training format.

Uses incremental HDF5 writing to avoid memory issues with large datasets.
"""

import numpy as np
import h5py
import argparse
import psutil
import os
from typing import Optional, Tuple
from ml.vocabulary import TFTVocabulary
from ml.encoder import TFTMatchEncoder
from ml.data_loader import load_matches_from_bigquery


def get_memory_usage_mb():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def log_memory(prefix=""):
    """Log current memory usage."""
    mem_mb = get_memory_usage_mb()
    print(f"  [{prefix}] Memory usage: {mem_mb:.1f} MB")
    return mem_mb


def process_matches_to_hdf5_incremental(
    output_file: str,
    project_id: Optional[str] = None,
    dataset_id: str = 'tft_analytics',
    limit: Optional[int] = None,
    max_units: int = 15,
    mappings_dir: str = 'name_mappings',
    batch_size: int = 100
) -> Tuple[TFTMatchEncoder, int]:
    """
    Load matches from BigQuery and write directly to HDF5 incrementally.
    This avoids memory issues by never loading all data into memory at once.

    Args:
        output_file: Output HDF5 file path (temporary file, will be overwritten)
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        limit: Optional limit on number of matches
        max_units: Maximum units per board
        mappings_dir: Directory with CSV mappings
        batch_size: Progress reporting interval

    Returns:
        Tuple of (encoder, num_matches_encoded)
    """
    print("="*80)
    print("TFT ML DATA PIPELINE (INCREMENTAL HDF5)")
    print("="*80)

    initial_mem = log_memory("Initial")

    # Step 1: Load vocabulary
    print("\n[1/5] Loading vocabulary...")
    vocab = TFTVocabulary(mappings_dir=mappings_dir)
    stats = vocab.get_stats()
    print(f"  Units: {stats['units']}")
    print(f"  Items: {stats['items']}")
    print(f"  Traits: {stats['traits']}")
    log_memory("After vocab load")

    # Step 2: Initialize encoder
    print("\n[2/5] Initializing encoder...")
    encoder = TFTMatchEncoder(vocabulary=vocab, max_units=max_units)
    feature_info = encoder.get_feature_info()
    print(f"  Player feature dim: {feature_info['player_feature_dim']}")
    print(f"  Match feature dim: {feature_info['match_feature_dim']}")
    match_feature_dim = feature_info['match_feature_dim']
    log_memory("After encoder init")

    # Step 3: Load matches from BigQuery
    print("\n[3/5] Loading matches from BigQuery...")
    matches = load_matches_from_bigquery(
        project_id=project_id,
        dataset_id=dataset_id,
        limit=limit
    )

    if not matches:
        raise ValueError("No matches loaded!")

    num_matches = len(matches)
    print(f"✓ Loaded {num_matches} matches")
    log_memory("After loading matches")

    # Step 4: Create HDF5 file with pre-allocated datasets
    print(f"\n[4/5] Creating HDF5 file: {output_file}")
    print(f"  Pre-allocating space for {num_matches} matches...")

    with h5py.File(output_file, 'w') as f:
        # Pre-allocate datasets
        X_dataset = f.create_dataset(
            'X',
            shape=(num_matches, match_feature_dim),
            dtype=np.int32,
            chunks=(min(batch_size, num_matches), match_feature_dim),
            compression='gzip',
            compression_opts=4
        )

        Y_dataset = f.create_dataset(
            'Y',
            shape=(num_matches, 8),
            dtype=np.int32,
            chunks=(min(batch_size, num_matches), 8),
            compression='gzip',
            compression_opts=4
        )

        print(f"✓ HDF5 file created")
        log_memory("After HDF5 creation")

        # Step 5: Encode matches and write incrementally
        print(f"\n[5/5] Encoding and writing {num_matches} matches incrementally...")

        encoded_count = 0
        failed_count = 0

        for i, match in enumerate(matches):
            try:
                X, Y = encoder.encode_match(match)

                # Write directly to HDF5
                X_dataset[encoded_count] = X
                Y_dataset[encoded_count] = Y

                encoded_count += 1

                # Progress reporting
                if (encoded_count) % batch_size == 0:
                    print(f"  Encoded {encoded_count}/{num_matches} matches...")
                    log_memory(f"Batch {encoded_count}")

            except Exception as e:
                failed_count += 1
                print(f"  Warning: Failed to encode match {match.get('match_id', 'unknown')}: {e}")
                continue

        # If we had failures, resize datasets
        if encoded_count < num_matches:
            print(f"\n  Resizing datasets: {num_matches} → {encoded_count} (removed {failed_count} failed matches)")
            X_dataset.resize((encoded_count, match_feature_dim))
            Y_dataset.resize((encoded_count, 8))

    print(f"\n✓ Incremental encoding complete!")
    print(f"  Successfully encoded: {encoded_count}/{num_matches} matches")
    print(f"  Failed: {failed_count} matches")

    final_mem = log_memory("Final")
    print(f"  Memory delta: {final_mem - initial_mem:.1f} MB")

    return encoder, encoded_count


# Legacy functions removed - use process_matches_to_hdf5_incremental() instead
# Old process_matches_to_arrays() accumulated all data in memory (caused OOM)
# Old create_train_val_test_splits() worked with in-memory arrays
# New approach: write directly to HDF5 incrementally, then split from HDF5


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


def create_splits_from_hdf5(
    input_file: str,
    output_file: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    random_seed: int = 42,
    metadata: dict = None,
    chunk_size: int = 500
):
    """
    Create train/val/test splits from an HDF5 file without loading everything into memory.
    Reads and writes in chunks to maintain low memory usage.

    Args:
        input_file: Input HDF5 file with 'X' and 'Y' datasets
        output_file: Output HDF5 file for splits
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        shuffle: Whether to shuffle before splitting
        random_seed: Random seed for reproducibility
        metadata: Optional metadata to save
        chunk_size: Size of chunks for reading/writing
    """
    print("\n" + "="*80)
    print("CREATING TRAIN/VAL/TEST SPLITS FROM HDF5")
    print("="*80)

    initial_mem = log_memory("Initial")

    # Validate ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    # Open input file and get dataset info
    print(f"\nReading from: {input_file}")
    with h5py.File(input_file, 'r') as f_in:
        n_samples = f_in['X'].shape[0]
        X_shape = f_in['X'].shape
        Y_shape = f_in['Y'].shape

        print(f"  Total samples: {n_samples}")
        print(f"  X shape: {X_shape}")
        print(f"  Y shape: {Y_shape}")

        # Create shuffled indices
        indices = np.arange(n_samples)
        if shuffle:
            print(f"\nShuffling with random seed {random_seed}...")
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        else:
            print("\nNo shuffling (keeping chronological order)...")

        # Calculate split points
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        print(f"\nSplit sizes:")
        print(f"  Train: {len(train_idx)} samples ({len(train_idx)/n_samples*100:.1f}%)")
        print(f"  Val:   {len(val_idx)} samples ({len(val_idx)/n_samples*100:.1f}%)")
        print(f"  Test:  {len(test_idx)} samples ({len(test_idx)/n_samples*100:.1f}%)")

        # Create output file with splits
        print(f"\nWriting splits to: {output_file}")
        with h5py.File(output_file, 'w') as f_out:
            # Create groups
            train_group = f_out.create_group('train')
            val_group = f_out.create_group('val')
            test_group = f_out.create_group('test')

            # Create datasets
            train_X = train_group.create_dataset('X', shape=(len(train_idx), X_shape[1]),
                                                  dtype=f_in['X'].dtype, compression='gzip', compression_opts=4)
            train_Y = train_group.create_dataset('Y', shape=(len(train_idx), Y_shape[1]),
                                                  dtype=f_in['Y'].dtype, compression='gzip', compression_opts=4)

            val_X = val_group.create_dataset('X', shape=(len(val_idx), X_shape[1]),
                                             dtype=f_in['X'].dtype, compression='gzip', compression_opts=4)
            val_Y = val_group.create_dataset('Y', shape=(len(val_idx), Y_shape[1]),
                                             dtype=f_in['Y'].dtype, compression='gzip', compression_opts=4)

            test_X = test_group.create_dataset('X', shape=(len(test_idx), X_shape[1]),
                                               dtype=f_in['X'].dtype, compression='gzip', compression_opts=4)
            test_Y = test_group.create_dataset('Y', shape=(len(test_idx), Y_shape[1]),
                                               dtype=f_in['Y'].dtype, compression='gzip', compression_opts=4)

            # Write data in chunks
            def write_split(split_name, split_idx, out_X, out_Y):
                print(f"\n  Writing {split_name} split ({len(split_idx)} samples)...")
                for start in range(0, len(split_idx), chunk_size):
                    end = min(start + chunk_size, len(split_idx))
                    batch_idx = split_idx[start:end]

                    # HDF5 requires sorted indices for fancy indexing
                    # Sort for reading, then restore original order for writing
                    sort_order = np.argsort(batch_idx)
                    sorted_batch_idx = batch_idx[sort_order]
                    unsort_order = np.argsort(sort_order)

                    # Read from input file with sorted indices
                    X_batch = f_in['X'][sorted_batch_idx]
                    Y_batch = f_in['Y'][sorted_batch_idx]

                    # Restore original (shuffled) order before writing
                    X_batch = X_batch[unsort_order]
                    Y_batch = Y_batch[unsort_order]

                    # Write to output file
                    out_X[start:end] = X_batch
                    out_Y[start:end] = Y_batch

                    if (end) % (chunk_size * 5) == 0 or end == len(split_idx):
                        print(f"    Progress: {end}/{len(split_idx)} samples...")
                        log_memory(f"{split_name} write")

            write_split('train', train_idx, train_X, train_Y)
            write_split('val', val_idx, val_X, val_Y)
            write_split('test', test_idx, test_X, test_Y)

            # Save metadata
            if metadata:
                for key, value in metadata.items():
                    f_out.attrs[key] = value

    print(f"\n✓ Splits saved to {output_file}")
    final_mem = log_memory("Final")
    print(f"  Memory delta: {final_mem - initial_mem:.1f} MB")
    print("="*80)


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
    """
    Simple test script for incremental HDF5 encoding.
    For full train/val/test splits, use encode_all_data.py instead.
    """
    parser = argparse.ArgumentParser(
        description='TFT ML Data Pipeline - Convert BigQuery data to HDF5 training format (INCREMENTAL)'
    )
    parser.add_argument('--project-id', type=str, default=None,
                       help='GCP project ID (auto-detected if not provided)')
    parser.add_argument('--dataset-id', type=str, default='tft_analytics',
                       help='BigQuery dataset ID')
    parser.add_argument('--limit', type=int, default=100,
                       help='Limit number of matches (default: 100 for testing)')
    parser.add_argument('--max-units', type=int, default=15,
                       help='Maximum units per board (for padding)')
    parser.add_argument('--mappings-dir', type=str, default='name_mappings',
                       help='Directory containing CSV mappings')
    parser.add_argument('--output', type=str, default='tft_training_data.h5',
                       help='Output HDF5 file')

    args = parser.parse_args()

    try:
        # Process matches incrementally
        encoder, num_encoded = process_matches_to_hdf5_incremental(
            output_file=args.output,
            project_id=args.project_id,
            dataset_id=args.dataset_id,
            limit=args.limit,
            max_units=args.max_units,
            mappings_dir=args.mappings_dir,
            batch_size=50
        )

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
        print("# X_tensor shape:", X_loaded.shape)
        print("# Y_tensor shape:", Y_loaded.shape)
        print("="*80)
        print()
        print("NOTE: For train/val/test splits, use encode_all_data.py instead")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
