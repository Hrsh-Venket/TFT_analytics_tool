"""
Evaluation metrics for TFT Ranking Model.

Metrics:
    - NDCG@8: Normalized Discounted Cumulative Gain at 8
    - Per-rank Accuracy: Accuracy for each placement (1st, 2nd, ..., 8th)
    - Mean Absolute Error (MAE): Average error in predicted placements
    - Top-4 Accuracy: Percentage of correct top-4 predictions
"""

import torch
import numpy as np
from typing import Dict, Tuple


def scores_to_placements(scores: torch.Tensor) -> torch.Tensor:
    """
    Convert scores to placements (rankings).

    Higher scores get better (lower) placements.
    E.g., [5.2, 3.1, 7.8, 2.0] → [2, 3, 1, 4]

    Args:
        scores: Tensor of shape (batch, 8) or (8,)

    Returns:
        placements: Tensor of same shape with values 1-8
    """
    # argsort twice gives ranks
    # First argsort gives indices that would sort the array
    # Second argsort gives the rank of each element

    # For descending order (higher scores → lower placements):
    # Negate scores before sorting
    if scores.dim() == 1:
        # Single sample
        return torch.argsort(torch.argsort(-scores)) + 1
    else:
        # Batch of samples
        return torch.argsort(torch.argsort(-scores, dim=1), dim=1) + 1


def relevance_to_placements(relevance: torch.Tensor) -> torch.Tensor:
    """
    Convert relevance scores back to placements.
    Relevance = 9 - placement

    Args:
        relevance: Tensor of shape (batch, 8) with values 1-8

    Returns:
        placements: Tensor of same shape with values 1-8
    """
    return 9 - relevance


def compute_ndcg(
    pred_scores: torch.Tensor,
    true_relevance: torch.Tensor,
    k: int = 8
) -> float:
    """
    Compute NDCG@k (Normalized Discounted Cumulative Gain).

    Args:
        pred_scores: Predicted scores, shape (batch, 8)
                    Higher scores = better placement
        true_relevance: True relevance scores, shape (batch, 8)
                       Higher values = better placement (8=1st, 1=8th)
        k: Compute NDCG@k (default: 8 for all players)

    Returns:
        NDCG@k score (0 to 1, higher is better)
    """
    batch_size = pred_scores.shape[0]
    ndcg_scores = []

    for i in range(batch_size):
        # Get predicted ranking (indices of sorted scores, descending)
        pred_ranking = torch.argsort(-pred_scores[i])[:k]

        # Get relevance at predicted positions
        pred_relevance = true_relevance[i][pred_ranking]

        # Compute DCG (Discounted Cumulative Gain)
        # DCG = sum(rel_i / log2(i+2)) for i in 0..k-1
        positions = torch.arange(1, k + 1, dtype=torch.float32, device=pred_scores.device)
        discounts = torch.log2(positions + 1)
        dcg = torch.sum(pred_relevance / discounts)

        # Compute IDCG (Ideal DCG) - sort true relevance descending
        ideal_relevance = torch.sort(true_relevance[i], descending=True)[0][:k]
        idcg = torch.sum(ideal_relevance / discounts)

        # NDCG = DCG / IDCG
        if idcg > 0:
            ndcg = dcg / idcg
        else:
            ndcg = torch.tensor(0.0)

        ndcg_scores.append(ndcg.item())

    return np.mean(ndcg_scores)


def compute_per_rank_accuracy(
    pred_placements: torch.Tensor,
    true_placements: torch.Tensor
) -> Dict[int, float]:
    """
    Compute accuracy for each rank (1st, 2nd, ..., 8th).

    Args:
        pred_placements: Predicted placements, shape (batch, 8)
        true_placements: True placements, shape (batch, 8)

    Returns:
        Dictionary mapping rank → accuracy
        E.g., {1: 0.25, 2: 0.30, ..., 8: 0.15}
    """
    accuracies = {}

    for rank in range(1, 9):
        # Find positions where true placement is this rank
        true_mask = (true_placements == rank)

        # Check if predictions match
        correct = (pred_placements[true_mask] == rank)

        # Compute accuracy for this rank
        if correct.numel() > 0:
            accuracies[rank] = correct.float().mean().item()
        else:
            accuracies[rank] = 0.0

    return accuracies


def compute_mae(
    pred_placements: torch.Tensor,
    true_placements: torch.Tensor
) -> float:
    """
    Compute Mean Absolute Error between predicted and true placements.

    Args:
        pred_placements: Predicted placements, shape (batch, 8)
        true_placements: True placements, shape (batch, 8)

    Returns:
        Mean Absolute Error
    """
    return torch.abs(pred_placements - true_placements).float().mean().item()


def compute_top_k_accuracy(
    pred_placements: torch.Tensor,
    true_placements: torch.Tensor,
    k: int = 4
) -> float:
    """
    Compute Top-K accuracy.

    A prediction is correct if a player who placed in top-K is predicted to be in top-K.

    Args:
        pred_placements: Predicted placements, shape (batch, 8)
        true_placements: True placements, shape (batch, 8)
        k: Top-K threshold (default: 4)

    Returns:
        Top-K accuracy (0 to 1)
    """
    # Mask for players who actually placed in top-K
    true_top_k = (true_placements <= k)

    # Mask for players predicted to place in top-K
    pred_top_k = (pred_placements <= k)

    # Correct if both agree (both top-K or both not top-K)
    correct = (true_top_k == pred_top_k)

    return correct.float().mean().item()


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.

    Args:
        model: TFT Ranking Model
        data_loader: DataLoader with (X, Y_relevance) batches
        device: Device to run on

    Returns:
        Dictionary with all metrics:
            - ndcg@8: NDCG at 8
            - accuracy_rank_1, ..., accuracy_rank_8: Per-rank accuracies
            - mae: Mean Absolute Error
            - top4_accuracy: Top-4 accuracy
            - avg_rank_accuracy: Average per-rank accuracy
    """
    model.eval()
    all_pred_scores = []
    all_true_relevance = []

    with torch.no_grad():
        for batch_X, batch_Y_relevance in data_loader:
            batch_X = batch_X.to(device)
            batch_Y_relevance = batch_Y_relevance.to(device)

            # Forward pass
            pred_scores = model(batch_X)

            all_pred_scores.append(pred_scores.cpu())
            all_true_relevance.append(batch_Y_relevance.cpu())

    # Concatenate all batches
    all_pred_scores = torch.cat(all_pred_scores, dim=0)
    all_true_relevance = torch.cat(all_true_relevance, dim=0)

    # Convert to placements
    pred_placements = scores_to_placements(all_pred_scores)
    true_placements = relevance_to_placements(all_true_relevance)

    # Compute metrics
    metrics = {}

    # NDCG@8
    metrics['ndcg@8'] = compute_ndcg(all_pred_scores, all_true_relevance, k=8)

    # Per-rank accuracy
    per_rank_acc = compute_per_rank_accuracy(pred_placements, true_placements)
    for rank, acc in per_rank_acc.items():
        metrics[f'accuracy_rank_{rank}'] = acc

    # Average per-rank accuracy
    metrics['avg_rank_accuracy'] = np.mean(list(per_rank_acc.values()))

    # MAE
    metrics['mae'] = compute_mae(pred_placements, true_placements)

    # Top-4 accuracy
    metrics['top4_accuracy'] = compute_top_k_accuracy(pred_placements, true_placements, k=4)

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "METRICS"):
    """Print metrics in a formatted way."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    # Primary metrics
    print(f"\n[PRIMARY METRICS]")
    print(f"  NDCG@8:          {metrics['ndcg@8']:.4f}")
    print(f"  Top-4 Accuracy:  {metrics['top4_accuracy']:.4f}")
    print(f"  MAE:             {metrics['mae']:.4f}")
    print(f"  Avg Rank Acc:    {metrics['avg_rank_accuracy']:.4f}")

    # Per-rank accuracy
    print(f"\n[PER-RANK ACCURACY]")
    for rank in range(1, 9):
        key = f'accuracy_rank_{rank}'
        if key in metrics:
            print(f"  Rank {rank}: {metrics[key]:.4f}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Test metrics with dummy data
    print("Testing metrics...")

    # Create dummy predictions and targets
    batch_size = 100
    pred_scores = torch.randn(batch_size, 8)  # Random scores
    true_relevance = torch.randint(1, 9, (batch_size, 8)).float()  # Random relevance 1-8

    # Test NDCG
    ndcg = compute_ndcg(pred_scores, true_relevance, k=8)
    print(f"NDCG@8 (random data): {ndcg:.4f}")

    # Test placements conversion
    pred_placements = scores_to_placements(pred_scores)
    true_placements = relevance_to_placements(true_relevance)

    print(f"\nExample scores: {pred_scores[0]}")
    print(f"Predicted placements: {pred_placements[0]}")
    print(f"True relevance: {true_relevance[0]}")
    print(f"True placements: {true_placements[0]}")

    # Test per-rank accuracy
    per_rank_acc = compute_per_rank_accuracy(pred_placements, true_placements)
    print(f"\nPer-rank accuracy: {per_rank_acc}")

    # Test MAE
    mae = compute_mae(pred_placements, true_placements)
    print(f"MAE: {mae:.4f}")

    # Test Top-4 accuracy
    top4_acc = compute_top_k_accuracy(pred_placements, true_placements, k=4)
    print(f"Top-4 Accuracy: {top4_acc:.4f}")

    print("\n✓ Metrics test passed!")
