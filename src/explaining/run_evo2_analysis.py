#!/usr/bin/env python3
"""
Script to extract activations from Evo2 model, train a sparse autoencoder on them,
and analyze the learned features.

This implements a similar approach to that described in:
"Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np
import random
import time
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any
from tqdm import tqdm

from src.explaining.evo2_activations import ExplainableEvo
from src.explaining.sparse_autoencoder import SparseAutoencoder
from src.explaining.feature_analysis import FeatureAnalyzer


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract and analyze Evo2 activations")
    
    # Evo2 model config
    parser.add_argument("--model_name", type=str, default="evo2_7b", 
                        help="Name of Evo2 model to use ('evo2_7b' or 'evo2_40b')")
    parser.add_argument("--local_path", type=str, default=None,
                        help="Optional path to local model checkpoint")
    
    # Layer selection for analysis
    parser.add_argument("--layer_name", type=str, required=True,
                        help="Name of the layer to extract activations from")
    
    # Data options
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to file containing genomic sequences to analyze")
    parser.add_argument("--max_samples", type=int, default=10000,
                        help="Maximum number of sequences to process")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for processing")
    
    # Autoencoder options
    parser.add_argument("--expansion_factor", type=float, default=8.0,
                        help="Expansion factor for autoencoder (latent_dim = input_dim * expansion_factor)")
    parser.add_argument("--l1_coefficient", type=float, default=1e-3,
                        help="L1 sparsity coefficient for the autoencoder")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate for training")
    parser.add_argument("--tied_weights", action="store_true",
                        help="Use tied weights in the autoencoder")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--save_activations", action="store_true",
                        help="Save raw activations (warning: can be large)")
    parser.add_argument("--cuda", action="store_true",
                        help="Use CUDA if available")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose output")
    
    # Action to perform
    parser.add_argument("--mode", type=str, choices=["extract", "train", "analyze", "all"], default="all",
                        help="Action to perform: extract activations, train autoencoder, analyze features, or all")
    
    return parser.parse_args()


def load_sequences(data_file: str, max_samples: int = None) -> List[str]:
    """
    Load genomic sequences from a file.
    
    Args:
        data_file: Path to file containing sequences
        max_samples: Maximum number of sequences to load
        
    Returns:
        List of sequences
    """
    # Check file extension
    ext = os.path.splitext(data_file)[1].lower()
    
    sequences = []
    
    # Handle different file formats
    if ext == ".fasta" or ext == ".fa":
        # FASTA format
        from Bio import SeqIO
        records = list(SeqIO.parse(data_file, "fasta"))
        sequences = [str(record.seq) for record in records[:max_samples]]
    elif ext == ".txt":
        # Simple text file with one sequence per line
        with open(data_file, "r") as f:
            sequences = [line.strip() for line in f if line.strip()][:max_samples]
    elif ext == ".json":
        # JSON file
        with open(data_file, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                sequences = [seq for seq in data if isinstance(seq, str)][:max_samples]
            elif isinstance(data, dict) and "sequences" in data:
                sequences = data["sequences"][:max_samples]
            else:
                raise ValueError(f"Unsupported JSON format in {data_file}")
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    if max_samples is not None and max_samples < len(sequences):
        sequences = sequences[:max_samples]
        
    return sequences


def extract_activations(args: argparse.Namespace) -> Tuple[torch.Tensor, List[str]]:
    """
    Extract activations from Evo2 model.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple of (activations, sequences)
    """
    print(f"Extracting activations from {args.model_name} layer {args.layer_name}...")
    
    # Load sequences
    sequences = load_sequences(args.data_file, args.max_samples)
    print(f"Loaded {len(sequences)} sequences from {args.data_file}")
    
    # Print a sample sequence
    if args.verbose:
        print(f"Sample sequence: {sequences[0][:100]}...")
    
    # Set up extractor
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    extractor = ExplainableEvo(
        model_name=args.model_name,
        local_path=args.local_path,
        device=device
    )
    
    # Check if the requested layer is available
    available_layers = extractor.list_available_layers()
    if args.layer_name not in available_layers:
        print(f"WARNING: Layer {args.layer_name} not found in available layers.")
        print(f"Available layers: {available_layers}")
        closest_match = min(available_layers, key=lambda x: len(set(x) - set(args.layer_name)))
        use_layer = input(f"Would you like to use {closest_match} instead? (y/n): ")
        if use_layer.lower() == "y":
            args.layer_name = closest_match
        else:
            print("Exiting.")
            sys.exit(1)
    
    # Extract activations
    start_time = time.time()
    activations = extractor.collect_activations(
        sequences=sequences,
        layer_names=[args.layer_name],
        batch_size=args.batch_size,
        show_progress=True
    )
    end_time = time.time()
    
    # Get the activations for the requested layer
    raw_activations = activations[args.layer_name]
    
    print(f"Extraction completed in {end_time - start_time:.2f} seconds")
    print(f"Extracted activations shape: {raw_activations.shape}")
    
    # Save activations if requested
    if args.save_activations:
        output_dir = Path(args.output_dir) / "activations"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        activation_file = output_dir / f"{args.layer_name}_activations.pt"
        torch.save(raw_activations, activation_file)
        
        # Save sequences for reference
        sequence_file = output_dir / "sequences.json"
        with open(sequence_file, "w") as f:
            json.dump(sequences, f)
            
        print(f"Saved activations to {activation_file}")
        print(f"Saved sequences to {sequence_file}")
    
    return raw_activations, sequences


def train_autoencoder(
    raw_activations: torch.Tensor,
    args: argparse.Namespace
) -> SparseAutoencoder:
    """
    Train a sparse autoencoder on the extracted activations.
    
    Args:
        raw_activations: Raw neuron activations from the model
        args: Command line arguments
        
    Returns:
        Trained sparse autoencoder
    """
    print("Training sparse autoencoder...")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    
    # Determine input and latent dimensions
    input_dim = raw_activations.shape[1]
    latent_dim = int(input_dim * args.expansion_factor)
    
    print(f"Autoencoder dimensions: input_dim={input_dim}, latent_dim={latent_dim}, "
          f"expansion_factor={args.expansion_factor:.1f}x")
    
    # Create autoencoder
    autoencoder = SparseAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        l1_coefficient=args.l1_coefficient,
        tied_weights=args.tied_weights,
        device=device
    )
    
    # Split data into train/val
    val_size = min(int(len(raw_activations) * 0.1), 1000)  # 10% or 1000 max
    indices = torch.randperm(len(raw_activations))
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_activations = raw_activations[train_indices]
    val_activations = raw_activations[val_indices]
    
    print(f"Training on {len(train_activations)} samples, validating on {len(val_activations)} samples")
    
    # Create output directory for checkpoints
    checkpoint_dir = Path(args.output_dir) / "models" / f"{args.layer_name}_expansion{args.expansion_factor:.1f}x"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Train the autoencoder
    start_time = time.time()
    metrics = autoencoder.train_model(
        train_activations=train_activations,
        val_activations=val_activations,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir=checkpoint_dir,
        verbose=args.verbose
    )
    end_time = time.time()
    
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    
    # Plot and save training metrics
    autoencoder.plot_training_metrics(save_path=checkpoint_dir / "training_metrics.png")
    
    # Save final model
    final_model_path = checkpoint_dir / "final_model.pt"
    autoencoder.save(final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    return autoencoder


def analyze_features(
    autoencoder: SparseAutoencoder,
    raw_activations: torch.Tensor,
    sequences: List[str],
    args: argparse.Namespace
) -> None:
    """
    Analyze the features learned by the sparse autoencoder.
    
    Args:
        autoencoder: Trained sparse autoencoder
        raw_activations: Raw neuron activations from the model
        sequences: Sequences corresponding to the activations
        args: Command line arguments
    """
    print("Analyzing learned features...")
    
    # Create output directory for analysis results
    output_dir = Path(args.output_dir) / "analysis" / f"{args.layer_name}_expansion{args.expansion_factor:.1f}x"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    
    # Create feature analyzer
    analyzer = FeatureAnalyzer(autoencoder, raw_activations, sequences, device)
    
    # Compute and plot feature density
    print("Computing feature density...")
    analyzer.plot_feature_density_histogram(save_path=output_dir / "feature_density.png")
    
    # Find and analyze feature clusters
    print("Clustering features...")
    try:
        cluster_data = analyzer.find_feature_clusters()
        analyzer.plot_feature_clusters(
            cluster_data, 
            feature_stats=analyzer.feature_stats,
            save_path=output_dir / "feature_clusters.png"
        )
        
        # Also plot clusters colored by different statistics
        for stat_name in ['mean_activation', 'sparsity']:
            analyzer.plot_feature_clusters(
                cluster_data, 
                feature_stats=analyzer.feature_stats,
                color_by=stat_name,
                save_path=output_dir / f"feature_clusters_{stat_name}.png"
            )
    except Exception as e:
        print(f"Error clustering features: {e}")
    
    # Save analysis results
    print("Saving analysis results...")
    analyzer.save_analysis_results(output_dir, save_latents=(args.save_activations))
    
    # Analyze a few random features
    n_features = min(5, autoencoder.latent_dim)
    feature_indices = np.random.choice(autoencoder.latent_dim, n_features, replace=False)
    
    print(f"Analyzing {n_features} random features...")
    for i, feature_idx in enumerate(feature_indices):
        print(f"Analyzing feature {feature_idx}...")
        
        # Plot activation pattern
        analyzer.plot_feature_activation_pattern(
            feature_idx, 
            save_path=output_dir / f"feature_{feature_idx}_activations.png"
        )
        
        # Plot weight pattern
        analyzer.plot_feature_weight_pattern(
            feature_idx, 
            save_path=output_dir / f"feature_{feature_idx}_weights.png"
        )
    
    print(f"Analysis results saved to {output_dir}")


def main() -> None:
    """Main function to run the analysis pipeline."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save command-line arguments
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Extract activations
    if args.mode in ["extract", "all"]:
        raw_activations, sequences = extract_activations(args)
    else:
        # Load pre-extracted activations if not extracting
        activation_file = Path(args.output_dir) / "activations" / f"{args.layer_name}_activations.pt"
        sequence_file = Path(args.output_dir) / "activations" / "sequences.json"
        
        if not activation_file.exists():
            print(f"Activation file {activation_file} not found. Run with --mode extract first.")
            return
            
        print(f"Loading pre-extracted activations from {activation_file}")
        raw_activations = torch.load(activation_file)
        
        if sequence_file.exists():
            with open(sequence_file, "r") as f:
                sequences = json.load(f)
        else:
            sequences = None
    
    # Train autoencoder
    if args.mode in ["train", "all"]:
        autoencoder = train_autoencoder(raw_activations, args)
    elif args.mode == "analyze":
        # Load pre-trained autoencoder
        model_dir = Path(args.output_dir) / "models" / f"{args.layer_name}_expansion{args.expansion_factor:.1f}x"
        model_file = model_dir / "final_model.pt"
        
        if not model_file.exists():
            print(f"Model file {model_file} not found. Run with --mode train first.")
            return
            
        print(f"Loading pre-trained autoencoder from {model_file}")
        device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        autoencoder = SparseAutoencoder.load_from_checkpoint(model_file, device=device)
    
    # Analyze features
    if args.mode in ["analyze", "all"]:
        analyze_features(autoencoder, raw_activations, sequences, args)
    
    print("Done!")


if __name__ == "__main__":
    main() 