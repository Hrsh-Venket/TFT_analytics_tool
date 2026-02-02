"""
Data utilities for TFT Ranking Model.
Handles data loading, normalization, and DataLoader creation.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
from ml.pipeline import load_splits_from_hdf5
from ml.model.config import Config


class FeatureNormalizer:
    """
    Normalizes features using mean and standard deviation.
    Fit on training data, then apply to validation and test data.
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False

    def fit(self, X: np.ndarray):
        """
        Fit normalizer on training data.

        Args:
            X: Training features (any shape)
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        # Avoid division by zero - set std to 1 for constant features
        self.std[self.std == 0] = 1.0

        self.fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted mean and std.

        Args:
            X: Features to normalize

        Returns:
            Normalized features
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform!")

        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


class ShuffledPlayerDataset(Dataset):
    """
    Dataset that shuffles player order within each match.

    This prevents the model from learning positional biases
    (e.g., player at position 0 is always 1st place).

    Each time a match is accessed, the 8 players are randomly
    permuted, with the same permutation applied to both features and labels.
    """

    def __init__(self, X: torch.Tensor, Y: torch.Tensor, shuffle: bool = True):
        """
        Initialize dataset.

        Args:
            X: Features tensor of shape (num_matches, 8, player_feature_dim)
            Y: Labels tensor of shape (num_matches, 8)
            shuffle: Whether to shuffle players (default: True)
        """
        self.X = X
        self.Y = Y
        self.shuffle = shuffle

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Get a match with shuffled player order.

        Returns:
            Tuple of (features, labels) with same random permutation applied to both
        """
        x = self.X[idx]  # Shape: (8, player_feature_dim)
        y = self.Y[idx]  # Shape: (8,)

        if self.shuffle:
            # Generate random permutation of players (0-7)
            perm = torch.randperm(8)

            # Apply same permutation to both features and labels
            x = x[perm]  # Shuffle players
            y = y[perm]  # Shuffle labels to match

        return x, y


def placement_to_relevance(Y: np.ndarray) -> np.ndarray:
    """
    Convert placements (1-8) to relevance scores (8-1).

    For NDCG loss, higher scores should indicate better performance.
    - 1st place (placement=1) → relevance=8
    - 8th place (placement=8) → relevance=1

    Args:
        Y: Placements array of shape (num_matches, 8)

    Returns:
        Relevance scores array of shape (num_matches, 8)
    """
    return 9 - Y


def relevance_to_placement(relevance: np.ndarray) -> np.ndarray:
    """
    Convert relevance scores (8-1) back to placements (1-8).

    Args:
        relevance: Relevance scores array

    Returns:
        Placements array
    """
    return 9 - relevance


def load_and_prepare_data(
    hdf5_path: str,
    normalize: bool = Config.NORMALIZE_FEATURES,
    verbose: bool = True
):
    """
    Load data from HDF5 and prepare for training.

    Steps:
        1. Load train/val/test splits from HDF5
        2. Determine player_feature_dim from data shape
        3. Reshape X from (N, 8*player_feature_dim) to (N, 8, player_feature_dim)
        4. Convert placements to relevance scores
        5. Optionally normalize features
        6. Convert to PyTorch tensors

    Args:
        hdf5_path: Path to HDF5 file with splits
        normalize: Whether to normalize features
        verbose: Print information

    Returns:
        Dictionary with:
            - train_X, train_Y: Training tensors
            - val_X, val_Y: Validation tensors
            - test_X, test_Y: Test tensors
            - player_feature_dim: Feature dimension per player
            - metadata: Metadata from HDF5
            - normalizer: Fitted normalizer (if normalize=True)
    """
    if verbose:
        print("=" * 80)
        print("LOADING AND PREPARING DATA")
        print("=" * 80)

    # Load splits from HDF5
    if verbose:
        print(f"\nLoading from: {hdf5_path}")

    X_train, Y_train, X_val, Y_val, X_test, Y_test, metadata = load_splits_from_hdf5(hdf5_path)

    # Determine player_feature_dim
    match_feature_dim = X_train.shape[1]
    player_feature_dim = match_feature_dim // Config.NUM_PLAYERS

    if verbose:
        print(f"\nData dimensions:")
        print(f"  Match feature dim: {match_feature_dim}")
        print(f"  Player feature dim: {player_feature_dim}")
        print(f"  Players per match: {Config.NUM_PLAYERS}")

    # Verify dimensions are correct
    assert match_feature_dim == player_feature_dim * Config.NUM_PLAYERS, \
        f"Dimension mismatch: {match_feature_dim} != {player_feature_dim} * {Config.NUM_PLAYERS}"

    # Reshape X from (N, 8*player_feature_dim) to (N, 8, player_feature_dim)
    if verbose:
        print(f"\nReshaping data...")

    X_train_reshaped = X_train.reshape(-1, Config.NUM_PLAYERS, player_feature_dim)
    X_val_reshaped = X_val.reshape(-1, Config.NUM_PLAYERS, player_feature_dim)
    X_test_reshaped = X_test.reshape(-1, Config.NUM_PLAYERS, player_feature_dim)

    if verbose:
        print(f"  Train: {X_train.shape} → {X_train_reshaped.shape}")
        print(f"  Val:   {X_val.shape} → {X_val_reshaped.shape}")
        print(f"  Test:  {X_test.shape} → {X_test_reshaped.shape}")

    # Convert placements to relevance scores
    if verbose:
        print(f"\nConverting placements to relevance scores...")

    Y_train_relevance = placement_to_relevance(Y_train)
    Y_val_relevance = placement_to_relevance(Y_val)
    Y_test_relevance = placement_to_relevance(Y_test)

    # Normalize features (optional)
    normalizer = None
    if normalize:
        if verbose:
            print(f"\nNormalizing features...")

        # Flatten for normalization, then reshape back
        normalizer = FeatureNormalizer()

        # Fit on training data (reshape back to 2D for normalization)
        X_train_flat = X_train_reshaped.reshape(-1, player_feature_dim)
        normalizer.fit(X_train_flat)

        # Transform all splits
        X_train_reshaped = normalizer.transform(X_train_flat).reshape(-1, Config.NUM_PLAYERS, player_feature_dim)

        X_val_flat = X_val_reshaped.reshape(-1, player_feature_dim)
        X_val_reshaped = normalizer.transform(X_val_flat).reshape(-1, Config.NUM_PLAYERS, player_feature_dim)

        X_test_flat = X_test_reshaped.reshape(-1, player_feature_dim)
        X_test_reshaped = normalizer.transform(X_test_flat).reshape(-1, Config.NUM_PLAYERS, player_feature_dim)

        if verbose:
            print(f"  ✓ Features normalized")

    # Convert to PyTorch tensors
    if verbose:
        print(f"\nConverting to PyTorch tensors...")

    train_X = torch.from_numpy(X_train_reshaped).float()
    train_Y = torch.from_numpy(Y_train_relevance).float()

    val_X = torch.from_numpy(X_val_reshaped).float()
    val_Y = torch.from_numpy(Y_val_relevance).float()

    test_X = torch.from_numpy(X_test_reshaped).float()
    test_Y = torch.from_numpy(Y_test_relevance).float()

    if verbose:
        print(f"  Train X: {train_X.shape}, dtype: {train_X.dtype}")
        print(f"  Train Y: {train_Y.shape}, dtype: {train_Y.dtype}")
        print(f"  Val X:   {val_X.shape}, dtype: {val_X.dtype}")
        print(f"  Val Y:   {val_Y.shape}, dtype: {val_Y.dtype}")
        print(f"  Test X:  {test_X.shape}, dtype: {test_X.dtype}")
        print(f"  Test Y:  {test_Y.shape}, dtype: {test_Y.dtype}")

    if verbose:
        print("\n" + "=" * 80)
        print("✓ DATA PREPARATION COMPLETE")
        print("=" * 80 + "\n")

    return {
        'train_X': train_X,
        'train_Y': train_Y,
        'val_X': val_X,
        'val_Y': val_Y,
        'test_X': test_X,
        'test_Y': test_Y,
        'player_feature_dim': player_feature_dim,
        'metadata': metadata,
        'normalizer': normalizer
    }


def create_data_loaders(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    val_X: torch.Tensor,
    val_Y: torch.Tensor,
    test_X: torch.Tensor,
    test_Y: torch.Tensor,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
    shuffle_players: bool = Config.SHUFFLE_PLAYERS,
    verbose: bool = True
):
    """
    Create PyTorch DataLoaders for train/val/test.

    Args:
        train_X, train_Y: Training tensors
        val_X, val_Y: Validation tensors
        test_X, test_Y: Test tensors
        batch_size: Batch size
        num_workers: Number of DataLoader workers
        shuffle_players: Whether to shuffle player order within matches (default: True)
        verbose: Print information

    Returns:
        Dictionary with train_loader, val_loader, test_loader
    """
    if verbose:
        print("Creating DataLoaders...")
        print(f"  Batch size: {batch_size}")
        print(f"  Num workers: {num_workers}")
        print(f"  Shuffle players: {shuffle_players}")

    # Create datasets with player shuffling to prevent positional bias
    # Training: shuffle players for data augmentation
    train_dataset = ShuffledPlayerDataset(train_X, train_Y, shuffle=shuffle_players)

    # Validation: also shuffle to match training conditions
    val_dataset = ShuffledPlayerDataset(val_X, val_Y, shuffle=shuffle_players)

    # Test: NO shuffling for consistent evaluation
    test_dataset = ShuffledPlayerDataset(test_X, test_Y, shuffle=False)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if Config.DEVICE == "cuda" else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if Config.DEVICE == "cuda" else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if Config.DEVICE == "cuda" else False
    )

    if verbose:
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        print("  ✓ DataLoaders created\n")

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader
    }


if __name__ == "__main__":
    # Test data loading (requires HDF5 file)
    print("Testing data utilities...")
    print("Note: This requires tft_data_splits.h5 to be available")
