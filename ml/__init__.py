"""
TFT ML Data Pipeline

Converts BigQuery match data to training-ready format.
Uses incremental HDF5 writing to avoid memory issues.
"""

from ml.vocabulary import TFTVocabulary
from ml.encoder import TFTMatchEncoder
from ml.data_loader import load_matches_from_bigquery
from ml.pipeline import (
    process_matches_to_hdf5_incremental,
    create_splits_from_hdf5,
    save_to_hdf5,
    load_from_hdf5,
    save_splits_to_hdf5,
    load_splits_from_hdf5
)

__all__ = [
    'TFTVocabulary',
    'TFTMatchEncoder',
    'load_matches_from_bigquery',
    'process_matches_to_hdf5_incremental',
    'create_splits_from_hdf5',
    'save_to_hdf5',
    'load_from_hdf5',
    'save_splits_to_hdf5',
    'load_splits_from_hdf5'
]
