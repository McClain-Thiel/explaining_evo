#!/usr/bin/env python3
"""
Script to query the database and retrieve information about sequences, features, etc.
"""

import argparse
import os
import sys
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from tabulate import tabulate

from sqlmodel import Session, select, func, and_, or_

from src.explaining.database.connection import get_database_url, create_db_engine
from src.explaining.database.models import Sequence, FeatureSet, Feature, SequenceFeature
from src.explaining.config import QuerySettings


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_args() -> Tuple[argparse.Namespace, QuerySettings]:
    """Parse command line arguments and create settings."""
    parser = argparse.ArgumentParser(
        description="Query the database for Evo2 feature analysis"
    )
    
    parser.add_argument(
        "--database_url", 
        type=str, 
        default=None,
        help="Database connection URL (overrides env variable)"
    )
    parser.add_argument(
        "--min_activation", 
        type=float, 
        default=None,
        help="Minimum activation threshold for features"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Maximum number of results to return"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to config file (YAML or JSON)"
    )
    
    # Command subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List feature sets
    list_feature_sets_parser = subparsers.add_parser(
        "list-feature-sets", 
        help="List available feature sets"
    )
    list_feature_sets_parser.add_argument(
        "--model", 
        type=str, 
        default=None,
        help="Filter by model name"
    )
    list_feature_sets_parser.add_argument(
        "--layer", 
        type=str, 
        default=None,
        help="Filter by layer name"
    )
    
    # List sequences
    list_sequences_parser = subparsers.add_parser(
        "list-sequences", 
        help="List sequences in the database"
    )
    list_sequences_parser.add_argument(
        "--source", 
        type=str, 
        default=None,
        help="Filter by source"
    )
    list_sequences_parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit the number of results"
    )
    list_sequences_parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output file (CSV format)"
    )
    
    # Get sequence features
    sequence_features_parser = subparsers.add_parser(
        "sequence-features", 
        help="Get features for a specific sequence"
    )
    sequence_features_parser.add_argument(
        "--sequence_id", 
        type=int, 
        required=True,
        help="Sequence ID"
    )
    sequence_features_parser.add_argument(
        "--feature_set_id", 
        type=int, 
        default=None,
        help="Feature set ID (if None, will use all feature sets)"
    )
    sequence_features_parser.add_argument(
        "--min_activation", 
        type=float, 
        default=None,
        help="Minimum activation threshold"
    )
    sequence_features_parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit the number of results"
    )
    sequence_features_parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output file (CSV format)"
    )
    
    # Get feature activations
    feature_activations_parser = subparsers.add_parser(
        "feature-activations", 
        help="Get sequences that activate a specific feature"
    )
    feature_activations_parser.add_argument(
        "--feature_id", 
        type=int, 
        required=True,
        help="Feature ID"
    )
    feature_activations_parser.add_argument(
        "--min_activation", 
        type=float, 
        default=None,
        help="Minimum activation threshold"
    )
    feature_activations_parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit the number of results"
    )
    feature_activations_parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output file (CSV format)"
    )
    
    # Export feature set
    export_feature_set_parser = subparsers.add_parser(
        "export-feature-set", 
        help="Export a feature set to a file"
    )
    export_feature_set_parser.add_argument(
        "--feature_set_id", 
        type=int, 
        required=True,
        help="Feature set ID"
    )
    export_feature_set_parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Output file (JSON format)"
    )
    
    args = parser.parse_args()
    
    # Load settings
    settings = QuerySettings()
    
    # Override from config file if provided
    if args.config:
        import yaml
        import json
        
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            sys.exit(1)
            
        # Load config based on file extension
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path) as f:
                config_data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path) as f:
                config_data = json.load(f)
        else:
            logger.error(f"Unsupported config file format: {config_path.suffix}")
            sys.exit(1)
            
        # Update settings from config
        for key, value in config_data.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
    
    # Override from command line arguments
    if args.database_url:
        settings.database_url = args.database_url
    if args.min_activation is not None:
        settings.min_activation = args.min_activation
    if args.limit is not None:
        settings.limit = args.limit
        
    # Also override from command-specific arguments
    if args.command == "sequence-features" or args.command == "feature-activations":
        if getattr(args, "min_activation", None) is not None:
            settings.min_activation = args.min_activation
        if getattr(args, "limit", None) is not None:
            settings.limit = args.limit
    elif args.command == "list-sequences":
        if getattr(args, "limit", None) is not None:
            settings.limit = args.limit
    
    return args, settings


def list_feature_sets(
    session: Session,
    model_name: Optional[str] = None,
    layer_name: Optional[str] = None
) -> None:
    """
    List available feature sets in the database.
    
    Args:
        session: Database session
        model_name: Optional filter by model name
        layer_name: Optional filter by layer name
    """
    stmt = select(FeatureSet)
    
    if model_name:
        stmt = stmt.where(FeatureSet.model_name == model_name)
    
    if layer_name:
        stmt = stmt.where(FeatureSet.layer_name == layer_name)
    
    feature_sets = session.exec(stmt).all()
    
    if not feature_sets:
        logger.info("No feature sets found in database")
        return
    
    # Count features in each feature set
    feature_counts = {}
    for fs in feature_sets:
        stmt = select(func.count(Feature.id)).where(Feature.feature_set_id == fs.id)
        count = session.exec(stmt).one()
        feature_counts[fs.id] = count
    
    # Count sequences associated with each feature set
    sequence_counts = {}
    for fs in feature_sets:
        stmt = select(func.count(func.distinct(SequenceFeature.sequence_id))).join(
            Feature, SequenceFeature.feature_id == Feature.id
        ).where(
            Feature.feature_set_id == fs.id
        )
        count = session.exec(stmt).one()
        sequence_counts[fs.id] = count
    
    # Prepare table data
    data = []
    for fs in feature_sets:
        data.append([
            fs.id,
            fs.name,
            fs.model_name,
            fs.layer_name,
            fs.expansion_factor,
            feature_counts[fs.id],
            sequence_counts[fs.id],
            fs.created_at.strftime("%Y-%m-%d %H:%M:%S")
        ])
    
    # Print table
    headers = ["ID", "Name", "Model", "Layer", "Expansion", "Features", "Sequences", "Created"]
    print(tabulate(data, headers=headers, tablefmt="grid"))


def list_sequences(
    session: Session,
    source: Optional[str] = None,
    limit: int = 10,
    output_file: Optional[str] = None
) -> None:
    """
    List sequences in the database.
    
    Args:
        session: Database session
        source: Optional filter by source
        limit: Maximum number of results
        output_file: Optional output file (CSV format)
    """
    stmt = select(Sequence)
    
    if source:
        stmt = stmt.where(Sequence.source == source)
    
    stmt = stmt.limit(limit)
    sequences = session.exec(stmt).all()
    
    if not sequences:
        logger.info("No sequences found in database")
        return
    
    # Prepare table data
    data = []
    csv_data = []
    for seq in sequences:
        # Count features associated with this sequence
        stmt = select(func.count(SequenceFeature.id)).where(
            SequenceFeature.sequence_id == seq.id
        )
        feature_count = session.exec(stmt).one()
        
        # Truncate sequence for display
        display_seq = seq.sequence[:50] + "..." if len(seq.sequence) > 50 else seq.sequence
        
        # Add to table data
        data.append([
            seq.id,
            display_seq,
            seq.source or "N/A",
            feature_count,
            seq.created_at.strftime("%Y-%m-%d %H:%M:%S")
        ])
        
        # Add to CSV data
        csv_data.append({
            "id": seq.id,
            "sequence": seq.sequence,
            "source": seq.source,
            "feature_count": feature_count,
            "created_at": seq.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": json.dumps(seq.metadata)
        })
    
    # Print table
    headers = ["ID", "Sequence", "Source", "Features", "Created"]
    print(tabulate(data, headers=headers, tablefmt="grid"))
    
    # Write to CSV if requested
    if output_file:
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "sequence", "source", "feature_count", "created_at", "metadata"])
            writer.writeheader()
            writer.writerows(csv_data)
            
        logger.info(f"Wrote {len(csv_data)} sequences to {output_file}")


def get_sequence_features(
    session: Session,
    sequence_id: int,
    feature_set_id: Optional[int] = None,
    min_activation: float = 0.1,
    limit: int = 10,
    output_file: Optional[str] = None
) -> None:
    """
    Get features for a specific sequence.
    
    Args:
        session: Database session
        sequence_id: Sequence ID
        feature_set_id: Optional feature set ID
        min_activation: Minimum activation threshold
        limit: Maximum number of results
        output_file: Optional output file (CSV format)
    """
    # Check if sequence exists
    sequence = session.get(Sequence, sequence_id)
    if not sequence:
        logger.error(f"Sequence with ID {sequence_id} not found")
        return
    
    # Query for features
    stmt = select(
        Feature, SequenceFeature.activation
    ).join(
        SequenceFeature, Feature.id == SequenceFeature.feature_id
    ).where(
        SequenceFeature.sequence_id == sequence_id,
        SequenceFeature.activation >= min_activation
    )
    
    if feature_set_id:
        stmt = stmt.where(Feature.feature_set_id == feature_set_id)
    
    stmt = stmt.order_by(SequenceFeature.activation.desc()).limit(limit)
    results = session.exec(stmt).all()
    
    if not results:
        logger.info(f"No features found for sequence {sequence_id} with activation >= {min_activation}")
        return
    
    # Get feature sets
    feature_set_ids = set(feature.feature_set_id for feature, _ in results)
    feature_sets = {}
    for fs_id in feature_set_ids:
        feature_sets[fs_id] = session.get(FeatureSet, fs_id)
    
    # Prepare table data
    data = []
    csv_data = []
    for feature, activation in results:
        feature_set = feature_sets[feature.feature_set_id]
        
        # Add to table data
        data.append([
            feature.id,
            feature.feature_idx,
            f"{feature_set.name} ({feature_set.id})",
            f"{activation:.4f}",
            feature.description or "N/A"
        ])
        
        # Add to CSV data
        csv_data.append({
            "feature_id": feature.id,
            "feature_idx": feature.feature_idx,
            "feature_set_id": feature_set.id,
            "feature_set_name": feature_set.name,
            "activation": activation,
            "description": feature.description or "",
            "mean_activation": feature.mean_activation,
            "max_activation": feature.max_activation,
            "activation_frequency": feature.activation_frequency
        })
    
    # Print table
    print(f"Sequence: {sequence.sequence[:50]}...")
    headers = ["ID", "Index", "Feature Set", "Activation", "Description"]
    print(tabulate(data, headers=headers, tablefmt="grid"))
    
    # Write to CSV if requested
    if output_file:
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f, 
                fieldnames=[
                    "feature_id", "feature_idx", "feature_set_id", "feature_set_name",
                    "activation", "description", "mean_activation", "max_activation",
                    "activation_frequency"
                ]
            )
            writer.writeheader()
            writer.writerows(csv_data)
            
        logger.info(f"Wrote {len(csv_data)} feature activations to {output_file}")


def get_feature_activations(
    session: Session,
    feature_id: int,
    min_activation: float = 0.1,
    limit: int = 10,
    output_file: Optional[str] = None
) -> None:
    """
    Get sequences that activate a specific feature.
    
    Args:
        session: Database session
        feature_id: Feature ID
        min_activation: Minimum activation threshold
        limit: Maximum number of results
        output_file: Optional output file (CSV format)
    """
    # Check if feature exists
    feature = session.get(Feature, feature_id)
    if not feature:
        logger.error(f"Feature with ID {feature_id} not found")
        return
    
    # Get feature set
    feature_set = session.get(FeatureSet, feature.feature_set_id)
    
    # Query for sequences
    stmt = select(
        Sequence, SequenceFeature.activation
    ).join(
        SequenceFeature, Sequence.id == SequenceFeature.sequence_id
    ).where(
        SequenceFeature.feature_id == feature_id,
        SequenceFeature.activation >= min_activation
    ).order_by(
        SequenceFeature.activation.desc()
    ).limit(limit)
    
    results = session.exec(stmt).all()
    
    if not results:
        logger.info(f"No sequences found for feature {feature_id} with activation >= {min_activation}")
        return
    
    # Prepare table data
    data = []
    csv_data = []
    for sequence, activation in results:
        # Truncate sequence for display
        display_seq = sequence.sequence[:50] + "..." if len(sequence.sequence) > 50 else sequence.sequence
        
        # Add to table data
        data.append([
            sequence.id,
            display_seq,
            f"{activation:.4f}",
            sequence.source or "N/A"
        ])
        
        # Add to CSV data
        csv_data.append({
            "sequence_id": sequence.id,
            "sequence": sequence.sequence,
            "activation": activation,
            "source": sequence.source or "",
            "metadata": json.dumps(sequence.metadata)
        })
    
    # Print table
    print(f"Feature: {feature_id} (index {feature.feature_idx}) from set {feature_set.name}")
    if feature.description:
        print(f"Description: {feature.description}")
    
    headers = ["ID", "Sequence", "Activation", "Source"]
    print(tabulate(data, headers=headers, tablefmt="grid"))
    
    # Write to CSV if requested
    if output_file:
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f, 
                fieldnames=["sequence_id", "sequence", "activation", "source", "metadata"]
            )
            writer.writeheader()
            writer.writerows(csv_data)
            
        logger.info(f"Wrote {len(csv_data)} sequence activations to {output_file}")


def export_feature_set(
    session: Session,
    feature_set_id: int,
    output_file: str
) -> None:
    """
    Export a feature set to a file.
    
    Args:
        session: Database session
        feature_set_id: Feature set ID
        output_file: Output file (JSON format)
    """
    # Check if feature set exists
    feature_set = session.get(FeatureSet, feature_set_id)
    if not feature_set:
        logger.error(f"Feature set with ID {feature_set_id} not found")
        return
    
    # Get features
    stmt = select(Feature).where(Feature.feature_set_id == feature_set_id)
    features = session.exec(stmt).all()
    
    if not features:
        logger.error(f"No features found for feature set {feature_set_id}")
        return
    
    # Prepare export data
    export_data = {
        "feature_set": {
            "id": feature_set.id,
            "name": feature_set.name,
            "model_name": feature_set.model_name,
            "layer_name": feature_set.layer_name,
            "expansion_factor": feature_set.expansion_factor,
            "latent_dim": feature_set.latent_dim,
            "input_dim": feature_set.input_dim,
            "l1_coefficient": feature_set.l1_coefficient,
            "config": feature_set.config,
            "created_at": feature_set.created_at.isoformat()
        },
        "features": []
    }
    
    for feature in features:
        feature_data = {
            "id": feature.id,
            "feature_idx": feature.feature_idx,
            "mean_activation": feature.mean_activation,
            "max_activation": feature.max_activation,
            "activation_frequency": feature.activation_frequency,
            "sparsity": feature.sparsity,
            "description": feature.description,
            # Skip feature vector to keep the file size reasonable
            # "feature_vector": feature.feature_vector
        }
        export_data["features"].append(feature_data)
    
    # Write to JSON file
    with open(output_file, "w") as f:
        json.dump(export_data, f, indent=2)
    
    logger.info(f"Exported feature set {feature_set.name} ({feature_set_id}) with {len(features)} features to {output_file}")


def main() -> None:
    """Main function to run database queries."""
    args, settings = parse_args()
    
    if not args.command:
        logger.error("No command specified. Use --help to see available commands.")
        return
    
    # Set database URL from settings if provided
    if settings.database_url:
        os.environ["DATABASE_URL"] = settings.database_url
    
    engine = create_db_engine()
    
    with Session(engine) as session:
        if args.command == "list-feature-sets":
            list_feature_sets(session, args.model, args.layer)
        
        elif args.command == "list-sequences":
            list_sequences(session, args.source, settings.limit, args.output)
        
        elif args.command == "sequence-features":
            get_sequence_features(
                session, args.sequence_id, args.feature_set_id,
                settings.min_activation, settings.limit, args.output
            )
        
        elif args.command == "feature-activations":
            get_feature_activations(
                session, args.feature_id, settings.min_activation, settings.limit, args.output
            )
        
        elif args.command == "export-feature-set":
            export_feature_set(session, args.feature_set_id, args.output)


if __name__ == "__main__":
    main() 