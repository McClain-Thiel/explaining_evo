import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import umap
import hdbscan
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import os
import json
import pandas as pd
from collections import defaultdict
import seaborn as sns
from tqdm import tqdm
import pickle
from sklearn.cluster import KMeans

from src.explaining.sparse_autoencoder import SparseAutoencoder


class FeatureAnalyzer:
    """
    Tool for analyzing and visualizing features learned by a sparse autoencoder.
    
    Inspired by the methods used in:
    "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"
    """
    
    def __init__(
        self,
        autoencoder: SparseAutoencoder,
        raw_activations: torch.Tensor,
        sequences: Optional[List[str]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the feature analyzer.
        
        Args:
            autoencoder: Trained sparse autoencoder
            raw_activations: Raw neuron activations from the model
            sequences: Optional list of sequences corresponding to activations
            device: Device to run computations on
        """
        self.autoencoder = autoencoder
        self.raw_activations = raw_activations
        self.sequences = sequences
        self.device = device
        
        # Compute latent activations
        self.latent_activations = self._compute_latent_activations(raw_activations)
        
        # Compute basic feature statistics
        self.feature_stats = self._compute_feature_stats()
        
    def _compute_latent_activations(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Compute latent activations for the given raw activations.
        
        Args:
            activations: Raw neuron activations
            
        Returns:
            Latent activations from the autoencoder
        """
        # Break into smaller batches to avoid OOM
        batch_size = 1024
        latents = []
        
        for i in range(0, len(activations), batch_size):
            batch = activations[i:i+batch_size].to(self.device)
            with torch.no_grad():
                latent = self.autoencoder.encode(batch)
                latents.append(latent.cpu())
                
        return torch.cat(latents, dim=0)
        
    def _compute_feature_stats(self) -> Dict[str, np.ndarray]:
        """
        Compute basic statistics for each feature.
        
        Returns:
            Dictionary of feature statistics
        """
        latent_np = self.latent_activations.numpy()
        
        # Compute statistics
        mean_act = np.mean(latent_np, axis=0)
        max_act = np.max(latent_np, axis=0)
        sparsity = np.mean(latent_np > 0, axis=0)
        std_act = np.std(latent_np, axis=0)
        
        return {
            'mean_activation': mean_act,
            'max_activation': max_act,
            'sparsity': sparsity,
            'std_activation': std_act,
        }
        
    def get_top_activating_examples(
        self, 
        feature_idx: int, 
        n_examples: int = 10
    ) -> Dict[str, Any]:
        """
        Get the top activating examples for a specific feature.
        
        Args:
            feature_idx: Index of the feature to analyze
            n_examples: Number of top examples to return
            
        Returns:
            Dictionary containing top examples and their activations
        """
        # Get feature activations
        feature_acts = self.latent_activations[:, feature_idx].numpy()
        
        # Get indices of top activating examples
        top_indices = np.argsort(feature_acts)[-n_examples:][::-1]
        
        # Get top activations
        top_activations = feature_acts[top_indices]
        
        # Get top examples (raw activations)
        top_raw_activations = self.raw_activations[top_indices]
        
        # Include sequences if available
        top_sequences = None
        if self.sequences is not None:
            top_sequences = [self.sequences[i] for i in top_indices]
        
        return {
            'indices': top_indices,
            'activations': top_activations,
            'raw_activations': top_raw_activations,
            'sequences': top_sequences
        }
        
    def compute_feature_density(self, threshold: float = 0.1) -> np.ndarray:
        """
        Compute the density of each feature (fraction of times it activates above threshold).
        
        Args:
            threshold: Activation threshold
            
        Returns:
            Array of feature densities
        """
        # Count how often each feature activates above threshold
        is_active = (self.latent_activations.numpy() > threshold).astype(float)
        density = np.mean(is_active, axis=0)
        
        return density
        
    def find_feature_clusters(
        self,
        n_components: int = 10,
        min_cluster_size: int = 5,
        metric: str = 'cosine',
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Cluster features based on their decoder weight vectors.
        
        Args:
            n_components: Number of UMAP dimensions for clustering
            min_cluster_size: Minimum cluster size for HDBSCAN
            metric: Distance metric to use
            random_state: Random seed
            
        Returns:
            Dictionary with cluster assignments and reduced feature vectors
        """
        # Get feature vectors (decoder weights)
        feature_vectors = self.autoencoder.get_feature_vectors().numpy()
        
        # Reduce dimensionality with UMAP
        reducer = umap.UMAP(
            n_components=n_components,
            metric=metric,
            random_state=random_state
        )
        reduced_vectors = reducer.fit_transform(feature_vectors)
        
        # Cluster with HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric=metric,
            gen_min_span_tree=True,
            cluster_selection_method='eom'
        )
        cluster_labels = clusterer.fit_predict(reduced_vectors)
        
        # Get 2D vectors for visualization
        reducer_2d = umap.UMAP(
            n_components=2,
            metric=metric,
            random_state=random_state
        )
        vectors_2d = reducer_2d.fit_transform(feature_vectors)
        
        return {
            'cluster_labels': cluster_labels,
            'reduced_vectors': reduced_vectors,
            'vectors_2d': vectors_2d,
            'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        }
        
    def plot_feature_clusters(
        self,
        cluster_data: Dict[str, Any],
        feature_stats: Optional[Dict[str, np.ndarray]] = None,
        color_by: str = 'cluster',
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot feature clusters in 2D space.
        
        Args:
            cluster_data: Output from find_feature_clusters
            feature_stats: Optional feature statistics for coloring
            color_by: What to color points by ('cluster', 'mean_act', 'sparsity', 'max_act')
            figsize: Figure size
            save_path: Path to save the figure
        """
        plt.figure(figsize=figsize)
        
        vectors_2d = cluster_data['vectors_2d']
        
        if color_by == 'cluster':
            # Color by cluster
            cluster_labels = cluster_data['cluster_labels']
            scatter = plt.scatter(
                vectors_2d[:, 0], 
                vectors_2d[:, 1], 
                c=cluster_labels, 
                cmap='tab20', 
                alpha=0.7,
                s=50
            )
            
            # Add legend for clusters
            n_clusters = cluster_data['n_clusters']
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=scatter.cmap(scatter.norm(i)),
                                         markersize=10, label=f'Cluster {i}')
                              for i in range(n_clusters)]
            plt.legend(handles=legend_elements, loc='upper right', ncol=2)
            
        elif feature_stats is not None and color_by in feature_stats:
            # Color by feature statistic
            values = feature_stats[color_by]
            scatter = plt.scatter(
                vectors_2d[:, 0], 
                vectors_2d[:, 1], 
                c=values, 
                cmap='viridis', 
                alpha=0.7,
                s=50
            )
            plt.colorbar(scatter, label=color_by)
            
        else:
            # Default coloring
            plt.scatter(
                vectors_2d[:, 0], 
                vectors_2d[:, 1], 
                alpha=0.7,
                s=50
            )
        
        plt.title(f'Feature Clusters (colored by {color_by})')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
        
    def plot_feature_activation_pattern(
        self, 
        feature_idx: int, 
        n_examples: int = 10,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot the pattern of activations for a specific feature.
        
        Args:
            feature_idx: Index of the feature to analyze
            n_examples: Number of top examples to include
            figsize: Figure size
            save_path: Path to save the figure
        """
        # Get top activating examples
        top_data = self.get_top_activating_examples(feature_idx, n_examples)
        
        plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        
        # Plot feature activations across all examples
        feature_acts = self.latent_activations[:, feature_idx].numpy()
        plt.hist(feature_acts[feature_acts > 0], bins=50, alpha=0.7)
        plt.title(f'Feature {feature_idx} Activation Distribution (non-zero only)')
        plt.xlabel('Activation Value')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        
        # Plot top activating examples
        plt.bar(range(len(top_data['activations'])), top_data['activations'])
        plt.title(f'Top {n_examples} Activations for Feature {feature_idx}')
        plt.xlabel('Example Rank')
        plt.ylabel('Activation Value')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
        
        # If sequences are available, print them with their activations
        if top_data['sequences'] is not None:
            print(f"Top activating sequences for Feature {feature_idx}:")
            for i, (seq, act) in enumerate(zip(top_data['sequences'], top_data['activations'])):
                print(f"{i+1}. Activation: {act:.4f}, Sequence: {seq}")
                
    def plot_feature_weight_pattern(
        self, 
        feature_idx: int,
        figsize: Tuple[int, int] = (10, 6),
        n_top_weights: int = 20,
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot the weights pattern for a specific feature.
        
        Args:
            feature_idx: Index of the feature to analyze
            figsize: Figure size
            n_top_weights: Number of top weights to display
            save_path: Path to save the figure
        """
        # Get feature vector (decoder weights)
        feature_vector = self.autoencoder.get_feature_vectors()[feature_idx].numpy()
        
        plt.figure(figsize=figsize)
        
        # Plot histogram of all weights
        plt.subplot(1, 2, 1)
        plt.hist(feature_vector, bins=50, alpha=0.7)
        plt.title(f'Feature {feature_idx} Weight Distribution')
        plt.xlabel('Weight Value')
        plt.ylabel('Count')
        
        # Plot top weights
        plt.subplot(1, 2, 2)
        
        # Get indices of top absolute weights
        top_indices = np.argsort(np.abs(feature_vector))[-n_top_weights:][::-1]
        top_weights = feature_vector[top_indices]
        
        # Color positive weights blue, negative weights red
        colors = ['blue' if w > 0 else 'red' for w in top_weights]
        
        plt.bar(range(len(top_weights)), top_weights, color=colors)
        plt.title(f'Top {n_top_weights} Weights for Feature {feature_idx}')
        plt.xlabel('Weight Index (sorted by magnitude)')
        plt.ylabel('Weight Value')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
        
    def plot_feature_density_histogram(
        self,
        threshold: float = 0.1,
        log_scale: bool = True,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot histogram of feature densities.
        
        Args:
            threshold: Activation threshold
            log_scale: Whether to use log scale for y-axis
            figsize: Figure size
            save_path: Path to save the figure
        """
        densities = self.compute_feature_density(threshold)
        
        plt.figure(figsize=figsize)
        plt.hist(densities, bins=50, alpha=0.7)
        plt.title('Feature Density Distribution')
        plt.xlabel('Density (fraction of examples where feature activates)')
        plt.ylabel('Count')
        
        if log_scale:
            plt.yscale('log')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
            
    def save_analysis_results(
        self,
        output_dir: Union[str, Path],
        save_latents: bool = False
    ) -> None:
        """
        Save analysis results to disk.
        
        Args:
            output_dir: Directory to save results to
            save_latents: Whether to save latent activations (can be large)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save feature statistics
        with open(output_dir / 'feature_stats.json', 'w') as f:
            # Convert numpy arrays to lists
            serializable_stats = {
                k: v.tolist() for k, v in self.feature_stats.items()
            }
            json.dump(serializable_stats, f)
            
        # Compute and save feature densities
        densities = self.compute_feature_density()
        np.save(output_dir / 'feature_densities.npy', densities)
        
        # Save latent activations if requested
        if save_latents:
            torch.save(self.latent_activations, output_dir / 'latent_activations.pt')
            
        # Try to compute and save feature clusters
        try:
            cluster_data = self.find_feature_clusters()
            
            # Save cluster labels
            np.save(output_dir / 'cluster_labels.npy', cluster_data['cluster_labels'])
            
            # Save 2D vectors for visualization
            np.save(output_dir / 'feature_vectors_2d.npy', cluster_data['vectors_2d'])
            
            # Create and save cluster visualization
            self.plot_feature_clusters(cluster_data, save_path=output_dir / 'feature_clusters.png')
            
        except Exception as e:
            print(f"Error computing clusters: {e}")
            
        # Save feature vectors
        feature_vectors = self.autoencoder.get_feature_vectors().numpy()
        np.save(output_dir / 'feature_vectors.npy', feature_vectors)
        
        print(f"Analysis results saved to {output_dir}")
        
    @classmethod
    def load_analysis_results(
        cls,
        input_dir: Union[str, Path],
        autoencoder: SparseAutoencoder,
        raw_activations: torch.Tensor,
        sequences: Optional[List[str]] = None
    ) -> 'FeatureAnalyzer':
        """
        Load analysis results from disk.
        
        Args:
            input_dir: Directory to load results from
            autoencoder: Trained sparse autoencoder
            raw_activations: Raw neuron activations from the model
            sequences: Optional list of sequences corresponding to activations
            
        Returns:
            FeatureAnalyzer instance with loaded results
        """
        input_dir = Path(input_dir)
        
        # Create analyzer instance
        analyzer = cls(autoencoder, raw_activations, sequences)
        
        # Load latent activations if available
        latents_path = input_dir / 'latent_activations.pt'
        if latents_path.exists():
            analyzer.latent_activations = torch.load(latents_path)
            
        # Load feature statistics if available
        stats_path = input_dir / 'feature_stats.json'
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats_dict = json.load(f)
                # Convert lists back to numpy arrays
                analyzer.feature_stats = {
                    k: np.array(v) for k, v in stats_dict.items()
                }
                
        return analyzer 