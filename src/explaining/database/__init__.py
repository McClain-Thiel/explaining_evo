"""
Database layer for Evo2 feature analysis.

This module provides database models and utilities for storing and 
retrieving sequences, features, and their relationships.
"""

from src.explaining.database.models import Sequence, FeatureSet, Feature, SequenceFeature
from src.explaining.database.connection import (
    get_database_url,
    create_db_engine,
    get_session,
    init_db
)
from src.explaining.database.operations import (
    get_or_create_sequence,
    get_or_create_feature_set,
    add_features_from_autoencoder,
    add_sequence_features,
    process_sequences_with_autoencoder,
    get_sequences_by_feature_activation,
    search_sequences
)

__all__ = [
    "Sequence",
    "FeatureSet",
    "Feature",
    "SequenceFeature",
    "get_database_url",
    "create_db_engine",
    "get_session",
    "init_db",
    "get_or_create_sequence",
    "get_or_create_feature_set",
    "add_features_from_autoencoder",
    "add_sequence_features",
    "process_sequences_with_autoencoder",
    "get_sequences_by_feature_activation",
    "search_sequences"
]
