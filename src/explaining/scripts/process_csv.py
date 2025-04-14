#!/usr/bin/env python3
"""
Script to process a CSV file containing sequences, extract features using the Evo2 model
and a sparse autoencoder, and store everything in the database.
"""

import argparse
import os
import sys
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch
import logging
from tqdm import tqdm
from contextlib import contextmanager

from sqlmodel import Session

from src.explaining.models.explainable_evo import ExplainableEvo
from src.explaining.models.sparse_autoencoder import SparseAutoencoder
from src.explaining.database.connection import create_db_engine, init_db
from src.explaining.database.operations import process_sequences_with_autoencoder


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process a CSV file of sequences, extract features, and store in database"
    )
    
    # Input file
    parser.add_argument(
        "--csv_file", 
        type=str, 
        required=True,
        help="Path to CSV file containing sequences"
    )
    parser.add_argument(
        "--sequence_column", 
        type=str, 
        default="sequence",
        help="Name of column containing sequences"
    )
    parser.add_argument(
        "--metadata_columns", 
        type=str, 
        nargs="*",
        default=[],
        help="Names of columns to include as metadata"
    )
    parser.add_argument(
        "--source", 
        type=str, 
        default=None,
        help="Source identifier for the sequences"
    )
    
    # Evo2 model config
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="evo2_7b", 
        help="Name of Evo2 model to use ('evo2_7b' or 'evo2_40b')"
    )
    parser.add_argument(
        "--local_path", 
        type=str, 
        default=None,
        help="Optional path to local model checkpoint"
    )
    parser.add_argument(
        "--layer_name", 
        type=str, 
        required=True,
        help="Name of the layer to extract activations from"
    )
    
    # Autoencoder options
    parser.add_argument(
        "--feature_set_name", 
        type=str, 
        default=None,
        help="Name for the feature set (defaults to '{model_name}_{layer_name}')"
    )
    parser.add_argument(
        "--autoencoder_path", 
        type=str, 
        default=None,
        help="Path to pre-trained autoencoder (if None, will train a new one)"
    )
    parser.add_argument(
        "--expansion_factor", 
        type=float, 
        default=8.0,
        help="Expansion factor for autoencoder (latent_dim = input_dim * expansion_factor)"
    )
    parser.add_argument(
        "--l1_coefficient", 
        type=float, 
        default=1e-3,
        help="L1 sparsity coefficient for the autoencoder"
    )
    parser.add_argument(
        "--activation_threshold", 
        type=float, 
        default=0.1,
        help="Minimum activation value to store"
    )
    parser.add_argument(
        "--max_features_per_sequence", 
        type=int, 
        default=100,
        help="Maximum number of features to store per sequence"
    )
    
    # Processing options
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--max_sequences", 
        type=int, 
        default=None,
        help="Maximum number of sequences to process"
    )
    parser.add_argument(
        "--cuda", 
        action="store_true",
        help="Use CUDA if available"
    )
    
    # Database options
    parser.add_argument(
        "--database_url", 
        type=str, 
        default=None,
        help="Database connection URL (overrides env variable)"
    )
    parser.add_argument(
        "--init_db", 
        action="store_true",
        help="Initialize database tables before processing"
    )
    
    return parser.parse_args()


def read_csv_file(
    csv_file: str,
    sequence_column: str,
    metadata_columns: List[str],
    max_sequences: Optional[int] = None
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Read sequences and metadata from a CSV file.
    
    Args:
        csv_file: Path to CSV file
        sequence_column: Name of column containing sequences
        metadata_columns: Names of columns to include as metadata
        max_sequences: Maximum number of sequences to read
        
    Returns:
        Tuple of (sequences, metadata_list)
    """
    sequences = []
    metadata_list = []
    
    with open(csv_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        
        for i, row in enumerate(reader):
            if max_sequences is not None and i >= max_sequences:
                break
                
            if sequence_column not in row:
                raise ValueError(f"Sequence column '{sequence_column}' not found in CSV")
                
            sequence = row[sequence_column]
            sequences.append(sequence)
            
            # Extract metadata
            metadata = {}
            for column in metadata_columns:
                if column in row:
                    metadata[column] = row[column]
            
            metadata_list.append(metadata)
    
    return sequences, metadata_list


@contextmanager
def get_db_session(database_url: Optional[str] = None):
    """Context manager for database session."""
    engine = create_db_engine(database_url)
    with Session(engine) as session:
        yield session


def main() -> None:
    """Main function to process CSV file and store results in database."""
    args = parse_args()
    
    # Initialize database if requested
    if args.init_db:
        engine = create_db_engine(args.database_url)
        init_db(engine)
    
    # Read sequences from CSV file
    logger.info(f"Reading sequences from {args.csv_file}")
    sequences, metadata_list = read_csv_file(
        csv_file=args.csv_file,
        sequence_column=args.sequence_column,
        metadata_columns=args.metadata_columns,
        max_sequences=args.max_sequences
    )
    logger.info(f"Read {len(sequences)} sequences")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    logger.info(f"Using device: {device}")
    
    # Set up extractor
    logger.info(f"Setting up Evo2 activation extractor for model {args.model_name}")
    extractor = ExplainableEvo(
        model_name=args.model_name,
        local_path=args.local_path,
        device=device
    )
    
    # Check if the requested layer is available
    available_layers = extractor.list_available_layers()
    if args.layer_name not in available_layers:
        logger.warning(f"Layer {args.layer_name} not found in available layers.")
        logger.info(f"Available layers: {available_layers}")
        closest_match = min(available_layers, key=lambda x: len(set(x) - set(args.layer_name)))
        use_layer = input(f"Would you like to use {closest_match} instead? (y/n): ")
        if use_layer.lower() == "y":
            args.layer_name = closest_match
        else:
            logger.error("Exiting due to invalid layer name.")
            sys.exit(1)
    
    # Extract activations
    logger.info(f"Extracting activations from layer {args.layer_name}")
    activations = extractor.collect_activations(
        sequences=sequences,
        layer_names=[args.layer_name],
        batch_size=args.batch_size,
        show_progress=True
    )
    raw_activations = activations[args.layer_name]
    logger.info(f"Extracted activations shape: {raw_activations.shape}")
    
    # Load or train autoencoder
    if args.autoencoder_path:
        logger.info(f"Loading pre-trained autoencoder from {args.autoencoder_path}")
        autoencoder = SparseAutoencoder.load_from_checkpoint(args.autoencoder_path, device=device)
    else:
        logger.info("Training new autoencoder")
        input_dim = raw_activations.shape[1]
        latent_dim = int(input_dim * args.expansion_factor)
        
        logger.info(f"Autoencoder dimensions: input_dim={input_dim}, latent_dim={latent_dim}")
        
        autoencoder = SparseAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            l1_coefficient=args.l1_coefficient,
            device=device
        )
        
        # Split data for training
        val_size = min(int(len(raw_activations) * 0.1), 1000)
        indices = torch.randperm(len(raw_activations))
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        train_activations = raw_activations[train_indices]
        val_activations = raw_activations[val_indices]
        
        # Train the autoencoder
        metrics = autoencoder.train_model(
            train_activations=train_activations,
            val_activations=val_activations,
            batch_size=args.batch_size,
            num_epochs=50,  # Default value, could be parameterized
            learning_rate=3e-4,  # Default value, could be parameterized
            verbose=True
        )
        
        # Save trained autoencoder
        output_dir = Path("models")
        output_dir.mkdir(exist_ok=True)
        model_path = output_dir / f"autoencoder_{args.model_name}_{args.layer_name}.pt"
        autoencoder.save(model_path)
        logger.info(f"Saved trained autoencoder to {model_path}")
    
    # Set feature set name if not provided
    if args.feature_set_name is None:
        args.feature_set_name = f"{args.model_name}_{args.layer_name}"
    
    # Process sequences and store in database
    logger.info("Processing sequences and storing in database")
    with get_db_session(args.database_url) as session:
        feature_set, db_sequences = process_sequences_with_autoencoder(
            session=session,
            autoencoder=autoencoder,
            sequences=sequences,
            raw_activations=raw_activations,
            feature_set_name=args.feature_set_name,
            model_name=args.model_name,
            layer_name=args.layer_name,
            metadata_list=metadata_list,
            source=args.source,
            activation_threshold=args.activation_threshold,
            max_features_per_sequence=args.max_features_per_sequence,
            batch_size=args.batch_size,
            show_progress=True
        )
        
        logger.info(f"Processed and stored {len(db_sequences)} sequences")
        logger.info(f"Feature set: {feature_set.name} (id: {feature_set.id})")
        logger.info(f"Number of features: {feature_set.latent_dim}")
    
    logger.info("Processing completed successfully!")


if __name__ == "__main__":
    main() 