import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple, Iterator
import numpy as np
import torch
from tqdm import tqdm
from sqlmodel import Session, select, or_

from src.explaining.database.models import Sequence, FeatureSet, Feature, SequenceFeature
from src.explaining.sparse_autoencoder import SparseAutoencoder


def hash_sequence(sequence: str) -> str:
    """
    Generate a hash for a sequence.
    
    Args:
        sequence: DNA/genomic sequence
        
    Returns:
        Hash string
    """
    return hashlib.sha256(sequence.encode()).hexdigest()


def get_or_create_sequence(
    session: Session, 
    sequence: str, 
    metadata: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None
) -> Tuple[Sequence, bool]:
    """
    Get an existing sequence or create a new one.
    
    Args:
        session: Database session
        sequence: DNA/genomic sequence
        metadata: Optional metadata dictionary
        source: Optional source identifier
        
    Returns:
        Tuple of (sequence_model, created) where created is True if new
    """
    sequence_hash = hash_sequence(sequence)
    
    # Try to find existing sequence
    stmt = select(Sequence).where(Sequence.sequence_hash == sequence_hash)
    result = session.exec(stmt).first()
    
    if result is not None:
        return result, False
    
    # Create new sequence
    new_sequence = Sequence(
        sequence=sequence,
        sequence_hash=sequence_hash,
        metadata=metadata or {},
        source=source
    )
    
    session.add(new_sequence)
    session.commit()
    session.refresh(new_sequence)
    
    return new_sequence, True


def get_or_create_feature_set(
    session: Session,
    name: str,
    model_name: str,
    layer_name: str,
    autoencoder: SparseAutoencoder,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[FeatureSet, bool]:
    """
    Get an existing feature set or create a new one.
    
    Args:
        session: Database session
        name: Name for the feature set
        model_name: Model name (e.g., "evo2_7b")
        layer_name: Layer name (e.g., "blocks.6.mlp")
        autoencoder: Trained sparse autoencoder
        config: Optional configuration dictionary
        
    Returns:
        Tuple of (feature_set, created) where created is True if new
    """
    # Try to find existing feature set
    stmt = select(FeatureSet).where(
        FeatureSet.name == name,
        FeatureSet.model_name == model_name,
        FeatureSet.layer_name == layer_name
    )
    result = session.exec(stmt).first()
    
    if result is not None:
        return result, False
    
    # Get expansion factor
    expansion_factor = getattr(autoencoder, 'expansion_factor', 
                              autoencoder.latent_dim / autoencoder.input_dim)
    
    # Create new feature set
    new_feature_set = FeatureSet(
        name=name,
        model_name=model_name,
        layer_name=layer_name,
        expansion_factor=expansion_factor,
        latent_dim=autoencoder.latent_dim,
        input_dim=autoencoder.input_dim,
        l1_coefficient=autoencoder.l1_coefficient,
        config=config or {}
    )
    
    session.add(new_feature_set)
    session.commit()
    session.refresh(new_feature_set)
    
    return new_feature_set, True


def add_features_from_autoencoder(
    session: Session,
    feature_set: FeatureSet,
    autoencoder: SparseAutoencoder,
    activations: torch.Tensor,
    batch_size: int = 1024,
    show_progress: bool = True
) -> List[Feature]:
    """
    Add features from a trained autoencoder to the database.
    
    Args:
        session: Database session
        feature_set: FeatureSet model
        autoencoder: Trained sparse autoencoder
        activations: Training activations used to compute feature statistics
        batch_size: Batch size for processing
        show_progress: Whether to show progress bar
        
    Returns:
        List of created Feature models
    """
    # Get feature vectors
    feature_vectors = autoencoder.get_feature_vectors().numpy()
    
    # Compute feature statistics
    stats = autoencoder.get_feature_stats(activations, batch_size)
    
    features = []
    iterator = range(autoencoder.latent_dim)
    
    if show_progress:
        iterator = tqdm(iterator, desc="Adding features to database")
    
    for feature_idx in iterator:
        # Create feature
        feature = Feature(
            feature_idx=feature_idx,
            feature_set_id=feature_set.id,
            mean_activation=float(stats['mean_activation'][feature_idx]),
            max_activation=float(stats['max_activation'][feature_idx]),
            activation_frequency=float(stats['activation_freq'][feature_idx]),
            sparsity=float(stats['activation_freq'][feature_idx]),
            feature_vector=feature_vectors[feature_idx].tolist()
        )
        
        session.add(feature)
        features.append(feature)
    
    session.commit()
    
    # Refresh all features to get their IDs
    for feature in features:
        session.refresh(feature)
    
    return features


def add_sequence_features(
    session: Session,
    sequence: Sequence,
    feature_set: FeatureSet,
    latent_activations: np.ndarray,
    activation_threshold: float = 0.1,
    max_features_per_sequence: Optional[int] = None
) -> List[SequenceFeature]:
    """
    Add feature activations for a sequence.
    
    Args:
        session: Database session
        sequence: Sequence model
        feature_set: FeatureSet model
        latent_activations: Latent activations for the sequence
        activation_threshold: Minimum activation value to store
        max_features_per_sequence: Maximum number of features to store per sequence
        
    Returns:
        List of created SequenceFeature models
    """
    # Get all features for the feature set
    stmt = select(Feature).where(Feature.feature_set_id == feature_set.id)
    features = {f.feature_idx: f for f in session.exec(stmt).all()}
    
    # Get indices of activations above threshold
    activations = latent_activations.flatten()
    active_indices = np.where(activations > activation_threshold)[0]
    
    # Sort by activation value if we need to limit
    if max_features_per_sequence is not None and len(active_indices) > max_features_per_sequence:
        sorted_indices = sorted(active_indices, key=lambda i: activations[i], reverse=True)
        active_indices = sorted_indices[:max_features_per_sequence]
    
    sequence_features = []
    
    for idx in active_indices:
        if idx in features:
            # Create sequence-feature link
            sequence_feature = SequenceFeature(
                sequence_id=sequence.id,
                feature_id=features[idx].id,
                activation=float(activations[idx])
            )
            
            session.add(sequence_feature)
            sequence_features.append(sequence_feature)
    
    session.commit()
    
    return sequence_features


def process_sequences_with_autoencoder(
    session: Session,
    autoencoder: SparseAutoencoder,
    sequences: List[str],
    raw_activations: torch.Tensor,
    feature_set_name: str,
    model_name: str,
    layer_name: str,
    metadata_list: Optional[List[Dict[str, Any]]] = None,
    source: Optional[str] = None,
    activation_threshold: float = 0.1,
    max_features_per_sequence: Optional[int] = None,
    batch_size: int = 32,
    show_progress: bool = True
) -> Tuple[FeatureSet, List[Sequence]]:
    """
    Process multiple sequences with an autoencoder and store in database.
    
    Args:
        session: Database session
        autoencoder: Trained sparse autoencoder
        sequences: List of DNA/genomic sequences
        raw_activations: Raw neuron activations
        feature_set_name: Name for the feature set
        model_name: Model name (e.g., "evo2_7b")
        layer_name: Layer name (e.g., "blocks.6.mlp")
        metadata_list: Optional list of metadata dictionaries (one per sequence)
        source: Optional source identifier
        activation_threshold: Minimum activation value to store
        max_features_per_sequence: Maximum number of features to store per sequence
        batch_size: Batch size for processing
        show_progress: Whether to show progress bar
        
    Returns:
        Tuple of (feature_set, sequences)
    """
    # Create or get feature set
    feature_set, _ = get_or_create_feature_set(
        session=session,
        name=feature_set_name,
        model_name=model_name,
        layer_name=layer_name,
        autoencoder=autoencoder
    )
    
    # Check if we already have features for this feature set
    stmt = select(Feature).where(Feature.feature_set_id == feature_set.id)
    existing_features = session.exec(stmt).all()
    
    if not existing_features:
        # Add features from autoencoder
        add_features_from_autoencoder(
            session=session,
            feature_set=feature_set,
            autoencoder=autoencoder,
            activations=raw_activations,
            batch_size=batch_size,
            show_progress=show_progress
        )
    
    # Get or create sequences and add feature activations
    db_sequences = []
    
    iterator = range(len(sequences))
    if show_progress:
        iterator = tqdm(iterator, desc="Processing sequences")
    
    for i in iterator:
        sequence = sequences[i]
        metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else None
        
        # Get or create sequence
        db_sequence, created = get_or_create_sequence(
            session=session,
            sequence=sequence,
            metadata=metadata,
            source=source
        )
        
        db_sequences.append(db_sequence)
        
        # Skip if sequence already processed for this feature set
        if not created:
            stmt = select(SequenceFeature).join(Feature).where(
                SequenceFeature.sequence_id == db_sequence.id,
                Feature.feature_set_id == feature_set.id
            )
            existing = session.exec(stmt).first()
            if existing:
                continue
        
        # Compute latent activations for this sequence
        with torch.no_grad():
            latent = autoencoder.encode(raw_activations[i:i+1].to(autoencoder.device))
            
        # Add sequence features
        add_sequence_features(
            session=session,
            sequence=db_sequence,
            feature_set=feature_set,
            latent_activations=latent.cpu().numpy(),
            activation_threshold=activation_threshold,
            max_features_per_sequence=max_features_per_sequence
        )
    
    return feature_set, db_sequences


def get_sequences_by_feature_activation(
    session: Session,
    feature: Feature,
    min_activation: Optional[float] = None,
    max_sequences: Optional[int] = None
) -> List[Tuple[Sequence, float]]:
    """
    Get sequences with highest activation for a specific feature.
    
    Args:
        session: Database session
        feature: Feature model
        min_activation: Optional minimum activation threshold
        max_sequences: Optional maximum number of sequences to return
        
    Returns:
        List of (sequence, activation) tuples sorted by activation
    """
    stmt = select(Sequence, SequenceFeature.activation).join(
        SequenceFeature, Sequence.id == SequenceFeature.sequence_id
    ).where(
        SequenceFeature.feature_id == feature.id
    )
    
    if min_activation is not None:
        stmt = stmt.where(SequenceFeature.activation >= min_activation)
    
    stmt = stmt.order_by(SequenceFeature.activation.desc())
    
    if max_sequences is not None:
        stmt = stmt.limit(max_sequences)
        
    results = session.exec(stmt).all()
    return results


def search_sequences(
    session: Session,
    query: Optional[str] = None,
    source: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None
) -> List[Sequence]:
    """
    Search sequences in the database.
    
    Args:
        session: Database session
        query: Optional string to search in sequence
        source: Optional source filter
        metadata_filter: Optional metadata filter dictionary
        limit: Optional maximum number of results
        
    Returns:
        List of matching Sequence models
    """
    stmt = select(Sequence)
    
    filters = []
    
    if query:
        filters.append(Sequence.sequence.contains(query))
    
    if source:
        filters.append(Sequence.source == source)
    
    # Apply metadata filters if provided
    # Note: JSON filtering is implemented differently across databases
    # This is a simple implementation that may need to be adapted
    
    # Apply all filters
    if filters:
        stmt = stmt.where(or_(*filters))
    
    if limit:
        stmt = stmt.limit(limit)
        
    return session.exec(stmt).all() 