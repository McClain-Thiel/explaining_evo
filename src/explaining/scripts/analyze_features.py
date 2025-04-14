#!/usr/bin/env python3
"""
Script to analyze feature vectors learned by the autoencoder, including
clustering, similarity analysis, and visualization.
"""

import argparse
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from tabulate import tabulate
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import pandas as pd
from sqlmodel import Session, select

from src.explaining.database.connection import get_database_url, create_db_engine
from src.explaining.database.models import FeatureSet, Feature
from src.explaining.config import FeatureAnalysisSettings


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_args() -> Tuple[argparse.Namespace, FeatureAnalysisSettings]:
    """Parse command line arguments and create settings."""
    parser = argparse.ArgumentParser(
        description="Analyze feature vectors for Evo2 feature analysis"
    )
    
    parser.add_argument(
        "--database_url", 
        type=str, 
        default=None,
        help="Database connection URL (overrides env variable)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to config file (YAML or JSON)"
    )
    parser.add_argument(
        "--top_n", 
        type=int, 
        default=None,
        help="Number of top features to return"
    )
    parser.add_argument(
        "--n_clusters", 
        type=int, 
        default=None,
        help="Number of clusters for K-means"
    )
    parser.add_argument(
        "--method", 
        type=str, 
        default=None,
        help="Dimensionality reduction method (pca, tsne, umap)"
    )
    parser.add_argument(
        "--n_components", 
        type=int, 
        default=None,
        help="Number of components for dimensionality reduction"
    )
    
    # Command subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Find similar features
    similar_features_parser = subparsers.add_parser(
        "similar-features", 
        help="Find features similar to a given feature"
    )
    similar_features_parser.add_argument(
        "--feature_id", 
        type=int, 
        required=True,
        help="Feature ID to find similar features to"
    )
    similar_features_parser.add_argument(
        "--feature_set_id", 
        type=int, 
        default=None,
        help="Feature set ID to search in (if None, will use same set as feature_id)"
    )
    similar_features_parser.add_argument(
        "--top_n", 
        type=int, 
        default=None,
        help="Number of similar features to return"
    )
    similar_features_parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output file (JSON format)"
    )
    
    # Cluster features
    cluster_features_parser = subparsers.add_parser(
        "cluster-features", 
        help="Cluster features using K-means"
    )
    cluster_features_parser.add_argument(
        "--feature_set_id", 
        type=int, 
        required=True,
        help="Feature set ID"
    )
    cluster_features_parser.add_argument(
        "--n_clusters", 
        type=int, 
        default=None,
        help="Number of clusters"
    )
    cluster_features_parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output file (JSON format)"
    )
    
    # Visualize features
    visualize_features_parser = subparsers.add_parser(
        "visualize-features", 
        help="Visualize features using dimensionality reduction"
    )
    visualize_features_parser.add_argument(
        "--feature_set_id", 
        type=int, 
        required=True,
        help="Feature set ID"
    )
    visualize_features_parser.add_argument(
        "--method", 
        type=str, 
        choices=["pca", "tsne", "umap"],
        default=None,
        help="Dimensionality reduction method"
    )
    visualize_features_parser.add_argument(
        "--n_components", 
        type=int, 
        default=None,
        help="Number of components for dimensionality reduction"
    )
    visualize_features_parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output file (PNG format)"
    )
    
    # Generate feature descriptions
    describe_features_parser = subparsers.add_parser(
        "describe-features", 
        help="Generate descriptions for features based on top activating sequences"
    )
    describe_features_parser.add_argument(
        "--feature_set_id", 
        type=int, 
        required=True,
        help="Feature set ID"
    )
    describe_features_parser.add_argument(
        "--feature_ids", 
        type=str, 
        default=None,
        help="Comma-separated list of feature IDs (if None, will describe all features)"
    )
    describe_features_parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output file (JSON format)"
    )
    
    args = parser.parse_args()
    
    # Load settings
    settings = FeatureAnalysisSettings()
    
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
    if args.top_n is not None:
        settings.top_n = args.top_n
    if args.n_clusters is not None:
        settings.n_clusters = args.n_clusters
    if args.method is not None:
        settings.method = args.method
    if args.n_components is not None:
        settings.n_components = args.n_components
    
    # Command-specific overrides
    if args.command == "similar-features" and getattr(args, "top_n", None) is not None:
        settings.top_n = args.top_n
    elif args.command == "cluster-features" and getattr(args, "n_clusters", None) is not None:
        settings.n_clusters = args.n_clusters
    elif args.command == "visualize-features":
        if getattr(args, "method", None) is not None:
            settings.method = args.method
        if getattr(args, "n_components", None) is not None:
            settings.n_components = args.n_components
    
    return args, settings


def get_feature_vectors(
    session: Session,
    feature_set_id: int,
    feature_ids: Optional[List[int]] = None
) -> Tuple[List[Feature], np.ndarray]:
    """
    Get feature vectors from the database.
    
    Args:
        session: Database session
        feature_set_id: Feature set ID
        feature_ids: Optional list of feature IDs to filter
        
    Returns:
        Tuple of (features, feature_vectors)
    """
    stmt = select(Feature).where(Feature.feature_set_id == feature_set_id)
    
    if feature_ids:
        stmt = stmt.where(Feature.id.in_(feature_ids))
    
    features = session.exec(stmt).all()
    
    if not features:
        logger.error(f"No features found for feature set {feature_set_id}")
        sys.exit(1)
    
    # Extract feature vectors
    feature_vectors = []
    for feature in features:
        if feature.feature_vector is None:
            logger.error(f"Feature {feature.id} has no feature vector")
            sys.exit(1)
        feature_vectors.append(np.array(feature.feature_vector))
    
    feature_vectors = np.array(feature_vectors)
    
    logger.info(f"Loaded {len(features)} feature vectors with shape {feature_vectors.shape}")
    
    return features, feature_vectors


def find_similar_features(
    session: Session,
    feature_id: int,
    feature_set_id: Optional[int] = None,
    top_n: int = 5,
    output_file: Optional[str] = None
) -> None:
    """
    Find features similar to a given feature based on cosine similarity.
    
    Args:
        session: Database session
        feature_id: Feature ID to find similar features to
        feature_set_id: Feature set ID to search in (if None, will use same set as feature_id)
        top_n: Number of similar features to return
        output_file: Optional output file (JSON format)
    """
    # Get target feature
    target_feature = session.get(Feature, feature_id)
    if not target_feature:
        logger.error(f"Feature with ID {feature_id} not found")
        return
    
    # Get target feature vector
    if target_feature.feature_vector is None:
        logger.error(f"Feature {feature_id} has no feature vector")
        return
    
    target_vector = np.array(target_feature.feature_vector)
    
    # If feature_set_id is not provided, use the same as the target feature
    if feature_set_id is None:
        feature_set_id = target_feature.feature_set_id
    
    # Get feature set
    feature_set = session.get(FeatureSet, feature_set_id)
    if not feature_set:
        logger.error(f"Feature set with ID {feature_set_id} not found")
        return
    
    # Get other features
    stmt = select(Feature).where(
        Feature.feature_set_id == feature_set_id,
        Feature.id != feature_id
    )
    features = session.exec(stmt).all()
    
    if not features:
        logger.error(f"No other features found in feature set {feature_set_id}")
        return
    
    # Calculate cosine similarity
    feature_vectors = []
    for feature in features:
        if feature.feature_vector is None:
            logger.warning(f"Feature {feature.id} has no feature vector, skipping")
            continue
        feature_vectors.append(np.array(feature.feature_vector))
    
    if not feature_vectors:
        logger.error("No valid feature vectors found")
        return
    
    feature_vectors = np.array(feature_vectors)
    similarities = cosine_similarity([target_vector], feature_vectors)[0]
    
    # Sort by similarity
    indices = np.argsort(-similarities)
    top_indices = indices[:top_n]
    top_features = [features[i] for i in top_indices]
    top_similarities = similarities[top_indices]
    
    # Prepare table data
    data = []
    json_data = []
    for feature, similarity in zip(top_features, top_similarities):
        data.append([
            feature.id,
            feature.feature_idx,
            f"{similarity:.4f}",
            feature.description or "N/A"
        ])
        
        json_data.append({
            "id": feature.id,
            "feature_idx": feature.feature_idx,
            "similarity": float(similarity),
            "description": feature.description,
            "mean_activation": feature.mean_activation,
            "max_activation": feature.max_activation,
            "activation_frequency": feature.activation_frequency,
            "sparsity": feature.sparsity
        })
    
    # Print table
    print(f"Features similar to feature {feature_id} (index {target_feature.feature_idx}) in feature set {feature_set.name}")
    if target_feature.description:
        print(f"Description: {target_feature.description}")
    
    headers = ["ID", "Index", "Similarity", "Description"]
    print(tabulate(data, headers=headers, tablefmt="grid"))
    
    # Write to JSON if requested
    if output_file:
        output_data = {
            "target_feature": {
                "id": target_feature.id,
                "feature_idx": target_feature.feature_idx,
                "description": target_feature.description,
                "mean_activation": target_feature.mean_activation,
                "max_activation": target_feature.max_activation,
                "activation_frequency": target_feature.activation_frequency,
                "sparsity": target_feature.sparsity
            },
            "similar_features": json_data
        }
        
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Wrote similar features to {output_file}")


def cluster_features(
    session: Session,
    feature_set_id: int,
    n_clusters: int = 10,
    output_file: Optional[str] = None
) -> None:
    """
    Cluster features using K-means.
    
    Args:
        session: Database session
        feature_set_id: Feature set ID
        n_clusters: Number of clusters
        output_file: Optional output file (JSON format)
    """
    # Get feature set
    feature_set = session.get(FeatureSet, feature_set_id)
    if not feature_set:
        logger.error(f"Feature set with ID {feature_set_id} not found")
        return
    
    # Get features and vectors
    features, feature_vectors = get_feature_vectors(session, feature_set_id)
    
    # Run K-means
    logger.info(f"Running K-means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(feature_vectors)
    
    # Group features by cluster
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        
        clusters[label].append({
            "id": features[i].id,
            "feature_idx": features[i].feature_idx,
            "description": features[i].description,
            "mean_activation": features[i].mean_activation,
            "max_activation": features[i].max_activation,
            "activation_frequency": features[i].activation_frequency,
            "sparsity": features[i].sparsity
        })
    
    # Print cluster summary
    print(f"Clustering results for feature set {feature_set.name} ({feature_set_id})")
    print(f"Number of clusters: {n_clusters}")
    
    for cluster_id in sorted(clusters.keys()):
        print(f"\nCluster {cluster_id} ({len(clusters[cluster_id])} features)")
        
        # Show the top 5 features
        top_features = sorted(clusters[cluster_id], key=lambda f: f["mean_activation"], reverse=True)[:5]
        
        data = []
        for feature in top_features:
            data.append([
                feature["id"],
                feature["feature_idx"],
                f"{feature['mean_activation']:.4f}",
                feature["description"] or "N/A"
            ])
        
        headers = ["ID", "Index", "Mean Activation", "Description"]
        print(tabulate(data, headers=headers, tablefmt="plain"))
    
    # Write to JSON if requested
    if output_file:
        output_data = {
            "feature_set": {
                "id": feature_set.id,
                "name": feature_set.name,
                "model_name": feature_set.model_name,
                "layer_name": feature_set.layer_name
            },
            "n_clusters": n_clusters,
            "clusters": {str(k): v for k, v in clusters.items()}
        }
        
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Wrote clustering results to {output_file}")


def visualize_features(
    session: Session,
    feature_set_id: int,
    method: str = "tsne",
    n_components: int = 2,
    output_file: Optional[str] = None
) -> None:
    """
    Visualize features using dimensionality reduction.
    
    Args:
        session: Database session
        feature_set_id: Feature set ID
        method: Dimensionality reduction method ("pca" or "tsne")
        n_components: Number of components for dimensionality reduction
        output_file: Optional output file (PNG format)
    """
    # Get feature set
    feature_set = session.get(FeatureSet, feature_set_id)
    if not feature_set:
        logger.error(f"Feature set with ID {feature_set_id} not found")
        return
    
    # Get features and vectors
    features, feature_vectors = get_feature_vectors(session, feature_set_id)
    
    # Normalize feature vectors
    norm_vectors = feature_vectors / np.linalg.norm(feature_vectors, axis=1)[:, np.newaxis]
    
    # Apply dimensionality reduction
    logger.info(f"Running {method.upper()} with {n_components} components...")
    
    if method == "pca":
        reducer = PCA(n_components=n_components)
    elif method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=0)
        except ImportError:
            logger.error("UMAP not installed. Please install with: pip install umap-learn")
            return
    else:  # tsne
        reducer = TSNE(n_components=n_components, perplexity=min(30, len(features) - 1), random_state=0)
    
    reduced_vectors = reducer.fit_transform(norm_vectors)
    
    # Get feature properties for coloring
    feature_activations = np.array([f.mean_activation for f in features])
    feature_sparsities = np.array([f.sparsity for f in features])
    
    # Create a DataFrame for plotting
    df = pd.DataFrame({
        "ID": [f.id for f in features],
        "Index": [f.feature_idx for f in features],
        "x": reduced_vectors[:, 0],
        "y": reduced_vectors[:, 1] if n_components > 1 else np.zeros(len(features)),
        "z": reduced_vectors[:, 2] if n_components > 2 else np.zeros(len(features)),
        "Mean Activation": feature_activations,
        "Sparsity": feature_sparsities,
        "Description": [f.description or "" for f in features]
    })
    
    # Set up the figure
    if n_components == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            df["x"], df["y"], df["z"],
            c=df["Mean Activation"],
            cmap="viridis",
            alpha=0.8,
            s=50
        )
        
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
        
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot by mean activation
        scatter1 = ax1.scatter(
            df["x"], df["y"],
            c=df["Mean Activation"],
            cmap="viridis",
            alpha=0.8,
            s=50
        )
        ax1.set_title("Features colored by Mean Activation")
        ax1.set_xlabel("Component 1")
        ax1.set_ylabel("Component 2")
        plt.colorbar(scatter1, ax=ax1, label="Mean Activation")
        
        # Plot by sparsity
        scatter2 = ax2.scatter(
            df["x"], df["y"],
            c=df["Sparsity"],
            cmap="coolwarm",
            alpha=0.8,
            s=50
        )
        ax2.set_title("Features colored by Sparsity")
        ax2.set_xlabel("Component 1")
        ax2.set_ylabel("Component 2")
        plt.colorbar(scatter2, ax=ax2, label="Sparsity")
    
    plt.suptitle(f"Feature visualization for {feature_set.name} ({feature_set_id}) using {method.upper()}")
    plt.tight_layout()
    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Saved visualization to {output_file}")
    else:
        plt.show()


def generate_feature_descriptions(
    session: Session,
    feature_set_id: int,
    feature_ids: Optional[List[int]] = None,
    output_file: Optional[str] = None
) -> None:
    """
    Generate descriptions for features based on top activating sequences.
    This is a placeholder function where you would implement a strategy
    to automatically generate feature descriptions.
    
    Args:
        session: Database session
        feature_set_id: Feature set ID
        feature_ids: Optional list of feature IDs
        output_file: Optional output file (JSON format)
    """
    # Get feature set
    feature_set = session.get(FeatureSet, feature_set_id)
    if not feature_set:
        logger.error(f"Feature set with ID {feature_set_id} not found")
        return
    
    # Get features
    stmt = select(Feature).where(Feature.feature_set_id == feature_set_id)
    
    if feature_ids:
        stmt = stmt.where(Feature.id.in_(feature_ids))
    
    features = session.exec(stmt).all()
    
    if not features:
        logger.error(f"No features found for feature set {feature_set_id}")
        return
    
    logger.info(f"Generating descriptions for {len(features)} features")
    
    # This is where you would implement a strategy to automatically generate descriptions
    # For now, we'll just print the feature statistics
    
    data = []
    json_data = []
    for feature in features:
        data.append([
            feature.id,
            feature.feature_idx,
            f"{feature.mean_activation:.4f}",
            f"{feature.max_activation:.4f}",
            f"{feature.activation_frequency:.4f}",
            f"{feature.sparsity:.4f}",
            feature.description or "N/A"
        ])
        
        json_data.append({
            "id": feature.id,
            "feature_idx": feature.feature_idx,
            "mean_activation": feature.mean_activation,
            "max_activation": feature.max_activation,
            "activation_frequency": feature.activation_frequency,
            "sparsity": feature.sparsity,
            "description": feature.description,
            "suggested_description": f"Feature {feature.feature_idx} with activation {feature.mean_activation:.4f}"
        })
    
    # Print table
    print(f"Features in set {feature_set.name} ({feature_set_id})")
    headers = ["ID", "Index", "Mean Act.", "Max Act.", "Act. Freq.", "Sparsity", "Description"]
    print(tabulate(data, headers=headers, tablefmt="grid"))
    
    # Write to JSON if requested
    if output_file:
        output_data = {
            "feature_set": {
                "id": feature_set.id,
                "name": feature_set.name,
                "model_name": feature_set.model_name,
                "layer_name": feature_set.layer_name
            },
            "features": json_data
        }
        
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Wrote feature descriptions to {output_file}")
        
    logger.info(
        "Note: This is a placeholder function. To generate meaningful descriptions, "
        "you would need to analyze top activating sequences for each feature "
        "and implement a suitable algorithm or use an external model."
    )


def main() -> None:
    """Main function to run feature analysis."""
    args, settings = parse_args()
    
    if not args.command:
        logger.error("No command specified. Use --help to see available commands.")
        return
    
    if settings.database_url:
        os.environ["DATABASE_URL"] = settings.database_url
    
    engine = create_db_engine()
    
    with Session(engine) as session:
        if args.command == "similar-features":
            find_similar_features(
                session, args.feature_id, args.feature_set_id,
                settings.top_n, args.output
            )
        
        elif args.command == "cluster-features":
            cluster_features(
                session, args.feature_set_id, settings.n_clusters, args.output
            )
        
        elif args.command == "visualize-features":
            visualize_features(
                session, args.feature_set_id, settings.method,
                settings.n_components, args.output
            )
        
        elif args.command == "describe-features":
            feature_ids = None
            if args.feature_ids:
                feature_ids = [int(fid.strip()) for fid in args.feature_ids.split(",")]
            
            generate_feature_descriptions(
                session, args.feature_set_id, feature_ids, args.output
            )


if __name__ == "__main__":
    main() 