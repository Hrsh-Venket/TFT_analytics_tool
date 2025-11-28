"""
TFT Ranking Model Architecture.
Input -> FC -> Multi-Head Self-Attention -> FC -> Scores
"""

import torch
import torch.nn as nn
from ml.model.config import Config


class TFTRankingModel(nn.Module):
    """
    TFT Ranking Model using Multi-Head Self-Attention.

    Architecture:
        Input: (batch, 8, player_feature_dim)
        ↓
        FC Layer (embedding): (batch, 8, hidden_dim)
        ↓
        N × Multi-Head Self-Attention (with dropout)
        ↓
        FC Layer (output): (batch, 8, 1)
        ↓
        Squeeze: (batch, 8) — scores for ranking

    The model outputs 8 scores (one per player).
    Higher scores indicate better expected placement.
    """

    def __init__(
        self,
        player_feature_dim: int,
        hidden_dim: int = Config.HIDDEN_DIM,
        n_attention_layers: int = Config.N_ATTENTION_LAYERS,
        n_heads: int = Config.N_HEADS,
        dropout: float = Config.DROPOUT
    ):
        """
        Initialize TFT Ranking Model.

        Args:
            player_feature_dim: Dimension of features for each player
            hidden_dim: Hidden dimension for attention layers
            n_attention_layers: Number of multi-head attention layers
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.player_feature_dim = player_feature_dim
        self.hidden_dim = hidden_dim
        self.n_attention_layers = n_attention_layers
        self.n_heads = n_heads
        self.dropout = dropout

        # Input normalization (normalizes raw features before embedding)
        self.input_norm = nn.BatchNorm1d(player_feature_dim)

        # Input embedding layer
        self.input_fc = nn.Linear(player_feature_dim, hidden_dim)

        # Multi-head self-attention layers
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,  # Standard transformer FF size
                dropout=dropout,
                batch_first=True  # Input shape: (batch, seq, feature)
            )
            for _ in range(n_attention_layers)
        ])

        # Output layer
        self.output_fc = nn.Linear(hidden_dim, 1)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 8, player_feature_dim)

        Returns:
            scores: Output tensor of shape (batch, 8)
                   Higher scores indicate better expected placement
        """
        # Input normalization
        # BatchNorm1d expects (batch, features) or (batch, features, seq)
        # Our input is (batch, 8, features), so we need to permute
        batch_size, num_players, feature_dim = x.shape
        x = x.permute(0, 2, 1)  # (batch, features, 8)
        x = self.input_norm(x)   # Normalize across batch
        x = x.permute(0, 2, 1)  # (batch, 8, features)

        # Input embedding
        x = self.input_fc(x)  # (batch, 8, hidden_dim)
        x = self.dropout_layer(x)

        # Multi-head self-attention layers
        for attention_layer in self.attention_layers:
            x = attention_layer(x)  # (batch, 8, hidden_dim)

        # Output projection
        x = self.output_fc(x)  # (batch, 8, 1)
        scores = x.squeeze(-1)  # (batch, 8)

        return scores

    def get_model_info(self):
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'player_feature_dim': self.player_feature_dim,
            'hidden_dim': self.hidden_dim,
            'n_attention_layers': self.n_attention_layers,
            'n_heads': self.n_heads,
            'dropout': self.dropout,
            'total_params': total_params,
            'trainable_params': trainable_params
        }

    def print_model_info(self):
        """Print model architecture information."""
        info = self.get_model_info()

        print("\n" + "=" * 80)
        print("TFT RANKING MODEL - ARCHITECTURE INFO")
        print("=" * 80)
        print(f"Player Feature Dim: {info['player_feature_dim']}")
        print(f"Hidden Dim: {info['hidden_dim']}")
        print(f"Attention Layers: {info['n_attention_layers']}")
        print(f"Attention Heads: {info['n_heads']}")
        print(f"Dropout: {info['dropout']}")
        print(f"Total Parameters: {info['total_params']:,}")
        print(f"Trainable Parameters: {info['trainable_params']:,}")
        print("=" * 80 + "\n")


def create_model(player_feature_dim: int, config: Config = None) -> TFTRankingModel:
    """
    Create TFT Ranking Model with configuration.

    Args:
        player_feature_dim: Dimension of features for each player
        config: Config object (defaults to Config class)

    Returns:
        TFTRankingModel instance
    """
    if config is None:
        config = Config

    model = TFTRankingModel(
        player_feature_dim=player_feature_dim,
        hidden_dim=config.HIDDEN_DIM,
        n_attention_layers=config.N_ATTENTION_LAYERS,
        n_heads=config.N_HEADS,
        dropout=config.DROPOUT
    )

    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing TFT Ranking Model...")

    # Example: player_feature_dim = 1541 (typical value)
    test_player_dim = 1541
    model = create_model(test_player_dim)
    model.print_model_info()

    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 8, test_player_dim)

    print(f"Input shape: {test_input.shape}")
    output = model(test_input)
    print(f"Output shape: {output.shape}")
    print(f"Output (scores): {output[0]}")

    print("\n✓ Model test passed!")
