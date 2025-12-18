"""
Configuration file for TFT Ranking Model.
All hyperparameters are easily adjustable here.
"""

import torch


class Config:
    """Configuration for TFT Ranking Model."""

    # ============================================================================
    # MODEL ARCHITECTURE
    # ============================================================================
    N_ATTENTION_LAYERS = 3          # Number of multi-head attention layers
    N_HEADS = 2                      # Number of attention heads per layer
    HIDDEN_DIM = 256                 # Hidden dimension for embeddings
    DROPOUT = 0.2                    # Dropout probability

    # ============================================================================
    # TRAINING HYPERPARAMETERS
    # ============================================================================
    BATCH_SIZE = 64                  # Batch size for training
    LEARNING_RATE = 1e-4             # Learning rate for optimizer
    EPOCHS = 50                      # Maximum number of training epochs
    EARLY_STOPPING_PATIENCE = 10     # Stop if no improvement for N epochs

    # ============================================================================
    # LOSS FUNCTION
    # ============================================================================
    LAMBDA_NDCG_SIGMA = 1.0          # Sigma parameter for LambdaNDCG loss

    # ============================================================================
    # DATA PREPROCESSING
    # ============================================================================
    NORMALIZE_FEATURES = True        # Standardize input features
    SHUFFLE_PLAYERS = True           # Shuffle player order within matches (prevents positional bias)
    NUM_WORKERS = 2                  # DataLoader workers (set to 0 for debugging)

    # ============================================================================
    # MODEL SAVING
    # ============================================================================
    SAVE_BEST_MODEL = True           # Save model with best validation NDCG
    SAVE_EVERY_EPOCH = False         # Save model after every epoch
    MODEL_SAVE_DIR = "saved_models"  # Directory to save models

    # ============================================================================
    # TENSORBOARD
    # ============================================================================
    TENSORBOARD_LOG_DIR = "runs"     # TensorBoard log directory

    # ============================================================================
    # DEVICE
    # ============================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ============================================================================
    # RANDOM SEED
    # ============================================================================
    RANDOM_SEED = 42                 # For reproducibility

    # ============================================================================
    # RANKING PARAMETERS
    # ============================================================================
    NUM_PLAYERS = 8                  # Number of players per match (always 8 in TFT)

    @classmethod
    def summary(cls):
        """Print configuration summary."""
        print("=" * 80)
        print("TFT RANKING MODEL - CONFIGURATION")
        print("=" * 80)
        print("\n[MODEL ARCHITECTURE]")
        print(f"  Attention Layers: {cls.N_ATTENTION_LAYERS}")
        print(f"  Attention Heads: {cls.N_HEADS}")
        print(f"  Hidden Dimension: {cls.HIDDEN_DIM}")
        print(f"  Dropout: {cls.DROPOUT}")

        print("\n[TRAINING]")
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        print(f"  Learning Rate: {cls.LEARNING_RATE}")
        print(f"  Epochs: {cls.EPOCHS}")
        print(f"  Early Stopping Patience: {cls.EARLY_STOPPING_PATIENCE}")

        print("\n[LOSS FUNCTION]")
        print(f"  Lambda NDCG Sigma: {cls.LAMBDA_NDCG_SIGMA}")

        print("\n[DATA]")
        print(f"  Normalize Features: {cls.NORMALIZE_FEATURES}")
        print(f"  Shuffle Players: {cls.SHUFFLE_PLAYERS}")
        print(f"  DataLoader Workers: {cls.NUM_WORKERS}")

        print("\n[SAVING]")
        print(f"  Save Best Model: {cls.SAVE_BEST_MODEL}")
        print(f"  Save Every Epoch: {cls.SAVE_EVERY_EPOCH}")
        print(f"  Model Save Directory: {cls.MODEL_SAVE_DIR}")

        print("\n[DEVICE]")
        print(f"  Device: {cls.DEVICE}")

        print("\n[MISC]")
        print(f"  Random Seed: {cls.RANDOM_SEED}")
        print(f"  Players per Match: {cls.NUM_PLAYERS}")
        print("=" * 80)
