#!/usr/bin/env python3
"""
View model predictions with detailed board breakdowns.

Usage:
    python view_predictions.py --model saved_models/best_model.pt --data tft_data_splits.h5 --num 3
"""

import argparse
import torch
from ml.model.evaluate import load_model
from ml.model.data_utils import load_and_prepare_data
from ml.model.decoder import TFTMatchDecoder, print_match_predictions
from ml.model.config import Config
from ml.vocabulary import TFTVocabulary


def main():
    parser = argparse.ArgumentParser(
        description='View TFT model predictions with detailed board breakdowns'
    )
    parser.add_argument('--model', type=str, default='saved_models/best_model.pt',
                       help='Path to saved model')
    parser.add_argument('--data', type=str, default='tft_data_splits.h5',
                       help='Path to HDF5 data file')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Which split to view (default: test)')
    parser.add_argument('--num', type=int, default=3,
                       help='Number of matches to display (default: 3)')
    parser.add_argument('--start', type=int, default=0,
                       help='Starting index (default: 0)')

    args = parser.parse_args()

    print("="*80)
    print("TFT MODEL PREDICTIONS - DETAILED VIEW")
    print("="*80)
    print(f"\nModel: {args.model}")
    print(f"Data: {args.data}")
    print(f"Split: {args.split}")
    print(f"Showing matches {args.start} to {args.start + args.num - 1}")

    # Load model
    print("\nLoading model...")
    model = load_model(args.model, device=Config.DEVICE)

    # Load data
    print("Loading data...")
    data = load_and_prepare_data(args.data, normalize=True, verbose=False)

    # Select split
    if args.split == 'train':
        X, Y = data['train_X'], data['train_Y']
    elif args.split == 'val':
        X, Y = data['val_X'], data['val_Y']
    else:
        X, Y = data['test_X'], data['test_Y']

    # Get sample matches
    end_idx = min(args.start + args.num, len(X))
    sample_X = X[args.start:end_idx]
    sample_Y = Y[args.start:end_idx]

    print(f"Loaded {len(sample_X)} matches\n")

    # Make predictions
    model.eval()
    with torch.no_grad():
        sample_X_device = sample_X.to(Config.DEVICE)
        pred_scores = model(sample_X_device)

    # Initialize decoder
    vocab = TFTVocabulary()
    decoder = TFTMatchDecoder(vocab)

    # Pretty print predictions
    print_match_predictions(
        X=sample_X,
        Y_relevance=sample_Y,
        pred_scores=pred_scores.cpu(),
        decoder=decoder,
        num_matches=len(sample_X)
    )


if __name__ == "__main__":
    main()
