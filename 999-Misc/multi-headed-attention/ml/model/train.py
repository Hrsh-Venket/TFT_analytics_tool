"""
Training script for TFT Ranking Model.

Handles:
    - Training loop with LambdaNDCG loss
    - Validation after each epoch
    - TensorBoard logging
    - Early stopping
    - Model checkpointing (best model)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from typing import Dict
from tqdm import tqdm

from ml.model.loss import LambdaNDCGLoss2

from ml.model.config import Config
from ml.model.architecture import create_model
from ml.model.data_utils import load_and_prepare_data, create_data_loaders
from ml.model.metrics import evaluate_model, print_metrics


def set_random_seed(seed: int = Config.RANDOM_SEED):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True  # Uncomment for full reproducibility (slower)
    # torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int
) -> float:
    """
    Train for one epoch.

    Args:
        model: TFT Ranking Model
        train_loader: Training DataLoader
        criterion: Loss function (LambdaNDCGLoss2)
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

    for batch_X, batch_Y_relevance in pbar:
        # Move to device
        batch_X = batch_X.to(device)
        batch_Y_relevance = batch_Y_relevance.to(device)

        # Forward pass
        pred_scores = model(batch_X)

        # Compute loss
        # LambdaNDCGLoss2 expects (scores, relevance)
        # scores: (batch, 8), relevance: (batch, 8)
        loss = criterion(pred_scores, batch_Y_relevance)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss
        total_loss += loss.item()

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str
) -> Dict[str, float]:
    """
    Validate the model.

    Args:
        model: TFT Ranking Model
        val_loader: Validation DataLoader
        criterion: Loss function
        device: Device to run on

    Returns:
        Dictionary with loss and metrics
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(val_loader)

    with torch.no_grad():
        for batch_X, batch_Y_relevance in val_loader:
            batch_X = batch_X.to(device)
            batch_Y_relevance = batch_Y_relevance.to(device)

            # Forward pass
            pred_scores = model(batch_X)

            # Compute loss
            loss = criterion(pred_scores, batch_Y_relevance)
            total_loss += loss.item()

    avg_loss = total_loss / num_batches

    # Compute evaluation metrics
    metrics = evaluate_model(model, val_loader, device)
    metrics['loss'] = avg_loss

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    filepath: str,
    config: Config
):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': {
            'hidden_dim': config.HIDDEN_DIM,
            'n_attention_layers': config.N_ATTENTION_LAYERS,
            'n_heads': config.N_HEADS,
            'dropout': config.DROPOUT,
            'player_feature_dim': model.player_feature_dim
        }
    }

    torch.save(checkpoint, filepath)


def train_model(
    hdf5_path: str,
    config: Config = None,
    verbose: bool = True
):
    """
    Main training function.

    Args:
        hdf5_path: Path to HDF5 file with train/val/test splits
        config: Configuration object (defaults to Config class)
        verbose: Print detailed information
    """
    if config is None:
        config = Config

    # Print configuration
    if verbose:
        config.summary()

    # Set random seed
    set_random_seed(config.RANDOM_SEED)

    # Create save directory
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

    # Load and prepare data
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 1: LOADING DATA")
        print("=" * 80)

    data = load_and_prepare_data(
        hdf5_path=hdf5_path,
        normalize=config.NORMALIZE_FEATURES,
        verbose=verbose
    )

    # Create data loaders
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 2: CREATING DATA LOADERS")
        print("=" * 80 + "\n")

    loaders = create_data_loaders(
        train_X=data['train_X'],
        train_Y=data['train_Y'],
        val_X=data['val_X'],
        val_Y=data['val_Y'],
        test_X=data['test_X'],
        test_Y=data['test_Y'],
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle_players=config.SHUFFLE_PLAYERS,
        verbose=verbose
    )

    # Create model
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 3: CREATING MODEL")
        print("=" * 80)

    model = create_model(
        player_feature_dim=data['player_feature_dim'],
        config=config
    )
    model = model.to(config.DEVICE)

    if verbose:
        model.print_model_info()
        print(f"Device: {config.DEVICE}\n")

    # Loss function
    criterion = LambdaNDCGLoss2(sigma=config.LAMBDA_NDCG_SIGMA)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # TensorBoard
    writer = SummaryWriter(log_dir=config.TENSORBOARD_LOG_DIR)

    if verbose:
        print("\n" + "=" * 80)
        print("STEP 4: TRAINING")
        print("=" * 80)
        print(f"TensorBoard logs: {config.TENSORBOARD_LOG_DIR}")
        print(f"Run: tensorboard --logdir={config.TENSORBOARD_LOG_DIR}\n")

    # Training state
    best_ndcg = 0.0
    patience_counter = 0
    best_epoch = 0

    # Training loop
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{config.EPOCHS}")
        print(f"{'='*80}")

        # Train
        train_loss = train_one_epoch(
            model=model,
            train_loader=loaders['train_loader'],
            criterion=criterion,
            optimizer=optimizer,
            device=config.DEVICE,
            epoch=epoch
        )

        # Validate
        val_metrics = validate(
            model=model,
            val_loader=loaders['val_loader'],
            criterion=criterion,
            device=config.DEVICE
        )

        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('NDCG@8/val', val_metrics['ndcg@8'], epoch)
        writer.add_scalar('MAE/val', val_metrics['mae'], epoch)
        writer.add_scalar('Top4_Accuracy/val', val_metrics['top4_accuracy'], epoch)

        for rank in range(1, 9):
            writer.add_scalar(f'Rank_Accuracy/val_rank_{rank}',
                            val_metrics[f'accuracy_rank_{rank}'], epoch)

        # Print epoch summary
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}")
        print(f"  Val NDCG@8: {val_metrics['ndcg@8']:.4f}")
        print(f"  Val MAE:    {val_metrics['mae']:.4f}")
        print(f"  Val Top-4:  {val_metrics['top4_accuracy']:.4f}")

        # Save best model
        current_ndcg = val_metrics['ndcg@8']
        if current_ndcg > best_ndcg:
            best_ndcg = current_ndcg
            best_epoch = epoch
            patience_counter = 0

            if config.SAVE_BEST_MODEL:
                best_model_path = os.path.join(config.MODEL_SAVE_DIR, 'best_model.pt')
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=val_metrics,
                    filepath=best_model_path,
                    config=config
                )
                print(f"  ✓ Saved best model (NDCG@8: {best_ndcg:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement (patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE})")

        # Save every epoch (optional)
        if config.SAVE_EVERY_EPOCH:
            epoch_model_path = os.path.join(config.MODEL_SAVE_DIR, f'model_epoch_{epoch}.pt')
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                filepath=epoch_model_path,
                config=config
            )

        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n{'='*80}")
            print(f"EARLY STOPPING (no improvement for {config.EARLY_STOPPING_PATIENCE} epochs)")
            print(f"{'='*80}")
            break

    # Training complete
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best NDCG@8: {best_ndcg:.4f} (Epoch {best_epoch})")
    print(f"Best model saved to: {os.path.join(config.MODEL_SAVE_DIR, 'best_model.pt')}")

    # Final evaluation on test set
    if verbose:
        print(f"\n{'='*80}")
        print("FINAL EVALUATION ON TEST SET")
        print(f"{'='*80}")

        # Load best model
        best_model_path = os.path.join(config.MODEL_SAVE_DIR, 'best_model.pt')
        checkpoint = torch.load(best_model_path, map_location=config.DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluate on test set
        test_metrics = evaluate_model(model, loaders['test_loader'], config.DEVICE)
        print_metrics(test_metrics, "TEST SET RESULTS")

    writer.close()

    return model, test_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train TFT Ranking Model')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to HDF5 file with splits')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')

    args = parser.parse_args()

    # Override config if specified
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.lr:
        Config.LEARNING_RATE = args.lr
    if args.epochs:
        Config.EPOCHS = args.epochs

    # Train model
    train_model(hdf5_path=args.data, config=Config, verbose=True)
