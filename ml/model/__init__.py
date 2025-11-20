"""
TFT Ranking Model package.

A deep learning model for predicting TFT match placements using multi-head self-attention.
"""

from ml.model.config import Config
from ml.model.architecture import TFTRankingModel, create_model
from ml.model.loss import LambdaNDCGLoss, LambdaNDCGLoss2, ApproxNDCGLoss
from ml.model.data_utils import (
    load_and_prepare_data,
    create_data_loaders,
    FeatureNormalizer
)
from ml.model.metrics import (
    evaluate_model,
    print_metrics,
    compute_ndcg,
    compute_mae,
    compute_top_k_accuracy
)
from ml.model.train import train_model
from ml.model.evaluate import load_model, evaluate_saved_model

__all__ = [
    'Config',
    'TFTRankingModel',
    'create_model',
    'LambdaNDCGLoss',
    'LambdaNDCGLoss2',
    'ApproxNDCGLoss',
    'load_and_prepare_data',
    'create_data_loaders',
    'FeatureNormalizer',
    'evaluate_model',
    'print_metrics',
    'compute_ndcg',
    'compute_mae',
    'compute_top_k_accuracy',
    'train_model',
    'load_model',
    'evaluate_saved_model'
]
