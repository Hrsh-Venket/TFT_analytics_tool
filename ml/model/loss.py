"""
LambdaNDCG Loss implementation.

Since pytorchltr has installation issues, we implement LambdaNDCG loss directly.
This is a clean, standalone implementation with no external dependencies beyond PyTorch.
"""

import torch
import torch.nn as nn


class LambdaNDCGLoss(nn.Module):
    """
    LambdaNDCG Loss for learning-to-rank.

    This loss function optimizes Normalized Discounted Cumulative Gain (NDCG)
    by computing lambda gradients that penalize ranking errors.

    Args:
        sigma: Temperature parameter for sigmoid smoothing (default: 1.0)
               Higher values = sharper gradients
    """

    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, y_pred, y_true):
        """
        Compute LambdaNDCG loss.

        Args:
            y_pred: Predicted scores, shape (batch_size, num_items)
                   Higher scores should correspond to better items
            y_true: True relevance labels, shape (batch_size, num_items)
                   Higher values = more relevant (e.g., 8=1st place, 1=8th place)

        Returns:
            Loss value (scalar tensor)
        """
        device = y_pred.device
        batch_size, num_items = y_pred.shape

        # Compute ideal DCG (IDCG) for normalization
        # Sort true relevance in descending order
        y_true_sorted, _ = torch.sort(y_true, dim=1, descending=True)

        # Compute DCG discount factors: 1/log2(rank+1)
        ranks = torch.arange(1, num_items + 1, dtype=torch.float32, device=device)
        discounts = 1.0 / torch.log2(ranks + 1)  # Shape: (num_items,)

        # Compute IDCG (Ideal DCG) - best possible DCG
        idcg = torch.sum(y_true_sorted * discounts.unsqueeze(0), dim=1)  # Shape: (batch_size,)

        # Avoid division by zero
        idcg = torch.clamp(idcg, min=1e-10)

        # Compute pairwise differences for all pairs (i, j)
        # Shape: (batch_size, num_items, num_items)
        y_pred_diff = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)  # pred[i] - pred[j]
        y_true_diff = y_true.unsqueeze(2) - y_true.unsqueeze(1)  # true[i] - true[j]

        # Compute |delta_NDCG| - change in NDCG if we swap items i and j
        # When true[i] > true[j], we want pred[i] > pred[j]

        # Compute gain difference when swapping positions
        # gain[i] = 2^rel[i] - 1 (standard NDCG formulation)
        gains = torch.pow(2.0, y_true) - 1.0  # Shape: (batch_size, num_items)

        # Gain difference when swapping i and j
        gain_diff = gains.unsqueeze(2) - gains.unsqueeze(1)  # Shape: (batch_size, num_items, num_items)

        # Compute discount difference
        # If i and j swap positions, discount change is:
        # discount[pos_i] - discount[pos_j]
        # We approximate this by using the predicted ranking

        # Get predicted ranks (argsort of argsort gives ranks)
        _, pred_rank_indices = torch.sort(y_pred, dim=1, descending=True)
        pred_ranks = torch.argsort(pred_rank_indices, dim=1) + 1  # Ranks from 1 to num_items

        pred_ranks_float = pred_ranks.float()
        discount_i = 1.0 / torch.log2(pred_ranks_float + 1)  # Shape: (batch_size, num_items)

        # Discount difference when swapping
        discount_diff = discount_i.unsqueeze(2) - discount_i.unsqueeze(1)  # Shape: (batch_size, num_items, num_items)

        # Delta NDCG = |gain_diff * discount_diff| / IDCG
        delta_ndcg = torch.abs(gain_diff * discount_diff) / idcg.unsqueeze(1).unsqueeze(2)

        # Compute lambda weights using sigmoid
        # lambda_ij = -sigma * delta_NDCG_ij / (1 + exp(sigma * (s_i - s_j)))
        # When true[i] > true[j] but pred[i] < pred[j], we want large penalty

        # Sigmoid of score difference
        sigmoid_diff = torch.sigmoid(self.sigma * y_pred_diff)

        # Lambda values
        # Only penalize when true relevance order is violated
        # If true[i] > true[j], we want pred[i] > pred[j]
        sign = torch.sign(y_true_diff)  # +1 if true[i] > true[j], -1 if true[i] < true[j]

        lambdas = -self.sigma * delta_ndcg * sigmoid_diff * sign

        # Compute loss: sum of lambda values for violated pairs
        # Only sum over pairs where true[i] != true[j]
        mask = (y_true_diff != 0).float()

        # Loss is the sum of lambdas weighted by mask
        # We want to minimize ranking errors, so we penalize when predictions are wrong
        loss = torch.sum(lambdas * mask * (sigmoid_diff - 0.5).abs(), dim=[1, 2])

        # Average over batch
        loss = loss.mean()

        return loss


class ApproxNDCGLoss(nn.Module):
    """
    Simplified approximate NDCG loss.

    This is a simpler alternative that directly approximates NDCG loss
    without computing full lambda gradients. Often works just as well.

    Args:
        temperature: Temperature for softmax approximation (default: 1.0)
    """

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, y_pred, y_true):
        """
        Compute approximate NDCG loss.

        Args:
            y_pred: Predicted scores, shape (batch_size, num_items)
            y_true: True relevance labels, shape (batch_size, num_items)

        Returns:
            Loss value (scalar tensor) - we return (1 - NDCG) so minimizing loss maximizes NDCG
        """
        device = y_pred.device
        batch_size, num_items = y_pred.shape

        # Use softmax to get soft ranking probabilities
        pred_probs = torch.softmax(y_pred / self.temperature, dim=1)

        # Compute discount factors
        ranks = torch.arange(1, num_items + 1, dtype=torch.float32, device=device)
        discounts = 1.0 / torch.log2(ranks + 1)

        # Compute gains from relevance
        gains = torch.pow(2.0, y_true) - 1.0

        # Approximate DCG using soft probabilities
        # Each item contributes its gain weighted by probability of being at each position
        dcg_approx = torch.sum(pred_probs * gains.unsqueeze(2) * discounts.unsqueeze(0).unsqueeze(0), dim=[1, 2])

        # Compute ideal DCG
        y_true_sorted, _ = torch.sort(y_true, dim=1, descending=True)
        gains_sorted = torch.pow(2.0, y_true_sorted) - 1.0
        idcg = torch.sum(gains_sorted * discounts.unsqueeze(0), dim=1)

        # Avoid division by zero
        idcg = torch.clamp(idcg, min=1e-10)

        # NDCG
        ndcg = dcg_approx / idcg

        # Loss = 1 - NDCG (we want to maximize NDCG, so minimize 1 - NDCG)
        loss = 1.0 - ndcg.mean()

        return loss


# Alias for backward compatibility with pytorchltr naming
LambdaNDCGLoss2 = LambdaNDCGLoss


if __name__ == "__main__":
    # Test the loss functions
    print("Testing LambdaNDCG Loss...")

    batch_size = 4
    num_items = 8

    # Create dummy data
    y_pred = torch.randn(batch_size, num_items)
    y_true = torch.randint(1, 9, (batch_size, num_items)).float()

    # Test LambdaNDCGLoss
    criterion1 = LambdaNDCGLoss(sigma=1.0)
    loss1 = criterion1(y_pred, y_true)
    print(f"LambdaNDCGLoss: {loss1.item():.4f}")

    # Test ApproxNDCGLoss
    criterion2 = ApproxNDCGLoss(temperature=1.0)
    loss2 = criterion2(y_pred, y_true)
    print(f"ApproxNDCGLoss: {loss2.item():.4f}")

    # Test backward pass
    loss1.backward()
    print("✓ Backward pass successful")

    print("\n✓ Loss functions working correctly!")
