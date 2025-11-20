"""
Evaluation script for TFT Ranking Model.
Load a saved model and evaluate on test set.
"""

import torch
import argparse
from ml.model.config import Config
from ml.model.architecture import TFTRankingModel
from ml.model.data_utils import load_and_prepare_data, create_data_loaders
from ml.model.metrics import evaluate_model, print_metrics


def load_model(checkpoint_path: str, device: str = "cuda") -> TFTRankingModel:
    """
    Load a saved model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file (.pt)
        device: Device to load model on

    Returns:
        Loaded TFT Ranking Model
    """
    print(f"Loading model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model config
    model_config = checkpoint['config']
    player_feature_dim = model_config['player_feature_dim']
    hidden_dim = model_config['hidden_dim']
    n_attention_layers = model_config['n_attention_layers']
    n_heads = model_config['n_heads']
    dropout = model_config['dropout']

    # Create model
    model = TFTRankingModel(
        player_feature_dim=player_feature_dim,
        hidden_dim=hidden_dim,
        n_attention_layers=n_attention_layers,
        n_heads=n_heads,
        dropout=dropout
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    print(f"  Trained for {checkpoint['epoch']} epochs")
    if 'metrics' in checkpoint and 'ndcg@8' in checkpoint['metrics']:
        print(f"  Best validation NDCG@8: {checkpoint['metrics']['ndcg@8']:.4f}")

    return model


def evaluate_saved_model(
    model_path: str,
    hdf5_path: str,
    split: str = "test",
    device: str = None,
    verbose: bool = True
):
    """
    Evaluate a saved model on a dataset split.

    Args:
        model_path: Path to saved model checkpoint
        hdf5_path: Path to HDF5 file with data splits
        split: Which split to evaluate on ("train", "val", or "test")
        device: Device to run on (default: auto-detect)
        verbose: Print detailed information

    Returns:
        Dictionary with evaluation metrics
    """
    if device is None:
        device = Config.DEVICE

    if verbose:
        print("=" * 80)
        print("TFT RANKING MODEL - EVALUATION")
        print("=" * 80)
        print(f"\nModel: {model_path}")
        print(f"Data: {hdf5_path}")
        print(f"Split: {split}")
        print(f"Device: {device}\n")

    # Load model
    model = load_model(model_path, device)

    if verbose:
        model.print_model_info()

    # Load data
    if verbose:
        print("\n" + "=" * 80)
        print("LOADING DATA")
        print("=" * 80)

    data = load_and_prepare_data(
        hdf5_path=hdf5_path,
        normalize=Config.NORMALIZE_FEATURES,
        verbose=verbose
    )

    # Select split
    if split == "train":
        X, Y = data['train_X'], data['train_Y']
    elif split == "val":
        X, Y = data['val_X'], data['val_Y']
    elif split == "test":
        X, Y = data['test_X'], data['test_Y']
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

    # Create DataLoader for selected split
    from torch.utils.data import TensorDataset, DataLoader

    dataset = TensorDataset(X, Y)
    data_loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device == "cuda" else False
    )

    if verbose:
        print(f"\nEvaluating on {split} set ({len(dataset)} samples)...\n")

    # Evaluate
    metrics = evaluate_model(model, data_loader, device)

    # Print results
    print_metrics(metrics, f"{split.upper()} SET RESULTS")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate TFT Ranking Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Evaluate best model on test set
  python -m ml.model.evaluate --model saved_models/best_model.pt --data tft_data_splits.h5

  # Evaluate on validation set
  python -m ml.model.evaluate --model saved_models/best_model.pt --data tft_data_splits.h5 --split val

  # Evaluate on training set
  python -m ml.model.evaluate --model saved_models/best_model.pt --data tft_data_splits.h5 --split train
        '''
    )

    parser.add_argument('--model', type=str, required=True,
                       help='Path to saved model checkpoint (.pt)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to HDF5 file with data splits')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Which split to evaluate on (default: test)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to run on (default: auto-detect)')

    args = parser.parse_args()

    # Evaluate
    evaluate_saved_model(
        model_path=args.model,
        hdf5_path=args.data,
        split=args.split,
        device=args.device,
        verbose=True
    )
