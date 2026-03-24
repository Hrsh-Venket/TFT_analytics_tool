#!/usr/bin/env python3
"""
Debug script to check if data is encoded correctly.
Inspects raw encoded values and decoder output.
"""

import numpy as np
import h5py
from ml.vocabulary import TFTVocabulary
from ml.encoder import TFTMatchEncoder
from ml.model.decoder import TFTMatchDecoder
from ml.pipeline import load_splits_from_hdf5


def inspect_raw_encoding():
    """Inspect raw encoded data from HDF5 file."""
    print("="*80)
    print("DEBUGGING DATA ENCODING")
    print("="*80)

    # Load vocabulary
    print("\n[1] Loading vocabulary...")
    vocab = TFTVocabulary()
    print(f"  Units: {vocab.num_units}")
    print(f"  Items: {vocab.num_items}")
    print(f"  Traits: {vocab.num_traits}")

    # Create encoder and decoder
    encoder = TFTMatchEncoder(vocabulary=vocab, max_units=15)
    decoder = TFTMatchDecoder(vocabulary=vocab, max_units=15)

    feature_info = encoder.get_feature_info()
    print(f"\n[2] Feature dimensions:")
    print(f"  Player feature dim: {feature_info['player_feature_dim']}")
    print(f"  Match feature dim: {feature_info['match_feature_dim']}")
    print(f"  Unit feature size: {vocab.num_units + vocab.num_items + 1}")

    # Load data from HDF5
    print(f"\n[3] Loading data from HDF5...")
    X_train, Y_train, X_val, Y_val, X_test, Y_test, metadata = load_splits_from_hdf5('tft_data_splits.h5')

    print(f"\n[4] Data shapes:")
    print(f"  Test X: {X_test.shape}")
    print(f"  Test Y: {Y_test.shape}")

    # Inspect first sample (flat encoding)
    print(f"\n[5] Inspecting first test sample (FLAT encoding)...")
    sample_flat = X_test[0]  # Shape: (match_feature_dim,)
    print(f"  Shape: {sample_flat.shape}")
    print(f"  Min value: {sample_flat.min()}")
    print(f"  Max value: {sample_flat.max()}")
    print(f"  Non-zero values: {np.count_nonzero(sample_flat)}/{len(sample_flat)}")

    # Reshape to (8, player_feature_dim)
    player_feature_dim = feature_info['player_feature_dim']
    sample_reshaped = sample_flat.reshape(8, player_feature_dim)

    print(f"\n[6] After reshaping to (8, player_feature_dim):")
    print(f"  Shape: {sample_reshaped.shape}")

    # Inspect first player
    print(f"\n[7] Inspecting Player 1:")
    player1 = sample_reshaped[0]

    # Extract level (first value)
    level = int(player1[0])
    print(f"  Level (raw): {player1[0]}")
    print(f"  Level (int): {level}")

    # Extract first few values
    print(f"\n  First 20 values:")
    print(f"  {player1[:20]}")

    # Check if there are any non-zero values
    print(f"\n  Non-zero values in player1: {np.count_nonzero(player1)}/{len(player1)}")

    # Check unit section
    unit_feature_size = vocab.num_units + vocab.num_items + 1
    offset = 1  # After level

    print(f"\n[8] Checking units (max 15):")
    print(f"  Unit feature size: {unit_feature_size}")

    for i in range(5):  # Check first 5 units
        unit_start = offset + (i * unit_feature_size)
        unit_end = unit_start + unit_feature_size
        unit_features = player1[unit_start:unit_end]

        # Check one-hot encoding
        unit_onehot = unit_features[:vocab.num_units]
        items_binary = unit_features[vocab.num_units:vocab.num_units + vocab.num_items]
        tier = int(unit_features[-1])

        non_zero_units = np.count_nonzero(unit_onehot)
        non_zero_items = np.count_nonzero(items_binary)

        print(f"\n  Unit {i+1}:")
        print(f"    One-hot non-zeros: {non_zero_units}")
        print(f"    Items non-zeros: {non_zero_items}")
        print(f"    Tier: {tier}")

        if non_zero_units > 0:
            unit_idx = np.argmax(unit_onehot)
            unit_name = decoder.idx_to_unit.get(unit_idx, "Unknown")
            print(f"    Unit name: {unit_name}")

            if non_zero_items > 0:
                item_indices = np.where(items_binary > 0)[0]
                item_names = [decoder.idx_to_item.get(idx, "Unknown") for idx in item_indices]
                print(f"    Items: {item_names}")

    # Check traits section
    traits_start = 1 + (15 * unit_feature_size)
    traits_features = player1[traits_start:]

    print(f"\n[9] Checking traits:")
    print(f"  Traits section size: {len(traits_features)}")
    print(f"  Non-zero traits: {np.count_nonzero(traits_features)}")

    active_traits = []
    for idx, tier_current in enumerate(traits_features):
        if tier_current > 0:
            trait_name = decoder.idx_to_trait.get(idx, "Unknown")
            active_traits.append(f"{trait_name} ({int(tier_current)})")

    print(f"  Active traits: {', '.join(active_traits)}")

    # Now test decoder
    print(f"\n[10] Testing decoder on this player:")
    decoded_player = decoder.decode_player(player1)
    print(f"  Decoded level: {decoded_player['level']}")
    print(f"  Decoded units: {len(decoded_player['units'])}")
    print(f"  Decoded traits: {len(decoded_player['traits'])}")

    if decoded_player['units']:
        print(f"\n  Decoded unit details:")
        for unit in decoded_player['units']:
            print(f"    - {unit}")
    else:
        print(f"\n  WARNING: No units decoded!")

    print(f"\n" + "="*80)
    print("DIAGNOSIS:")
    print("="*80)

    if np.count_nonzero(player1) == 0:
        print("❌ PROBLEM: Player data is all zeros!")
        print("   The encoding step may have failed or data is corrupted.")
    elif level == 0 and len(decoded_player['units']) == 0:
        print("❌ PROBLEM: Level is 0 and no units found!")
        print("   Either:")
        print("   1. Data was not encoded with unit information")
        print("   2. Decoder is reading from wrong positions")
        print("   3. This is an early-game snapshot with no units")
    elif len(decoded_player['units']) > 0:
        print("✓ Data appears to be encoded correctly!")
    else:
        print("⚠️  UNCLEAR: Data has non-zero values but decoder found no units")
        print("   Decoder may have bugs or encoding format mismatch")


if __name__ == "__main__":
    inspect_raw_encoding()
