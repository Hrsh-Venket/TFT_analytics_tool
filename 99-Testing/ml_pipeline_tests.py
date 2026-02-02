#!/usr/bin/env python3
"""
Test script for ML data pipeline.
Tests with a small number of matches.
"""

from ml.pipeline import process_matches_to_arrays, save_to_hdf5, load_from_hdf5


def test_pipeline():
    """Test the ML pipeline with 10 matches."""
    print("Testing ML pipeline with 10 matches...")

    try:
        # Process a small number of matches
        X, Y, encoder = process_matches_to_arrays(
            limit=10,
            max_units=15,
            mappings_dir='name_mappings'
        )

        # Save to test file
        output_file = 'test_data.h5'
        metadata = {
            'num_matches': X.shape[0],
            'match_feature_dim': X.shape[1],
            **encoder.get_feature_info()
        }
        save_to_hdf5(X, Y, output_file, metadata)

        # Load and verify
        X_loaded, Y_loaded, metadata_loaded = load_from_hdf5(output_file)

        # Verify shapes
        assert X_loaded.shape == X.shape, "X shape mismatch!"
        assert Y_loaded.shape == Y.shape, "Y shape mismatch!"

        print("\n✓ All tests passed!")
        print(f"\nFeature info:")
        for key, value in encoder.get_feature_info().items():
            print(f"  {key}: {value}")

        # Show PyTorch usage
        print("\n" + "="*60)
        print("To use with PyTorch:")
        print("="*60)
        print("import torch")
        print("from ml.pipeline import load_from_hdf5")
        print()
        print(f"X, Y, metadata = load_from_hdf5('{output_file}')")
        print("X_tensor = torch.from_numpy(X).float()")
        print("Y_tensor = torch.from_numpy(Y).long()")
        print()
        print(f"# X_tensor shape: {X.shape}")
        print(f"# Y_tensor shape: {Y.shape}")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(test_pipeline())
