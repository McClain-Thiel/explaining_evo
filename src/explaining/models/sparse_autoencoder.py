import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import time
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm


class SparseAutoencoder(nn.Module):
    """
    Sparse autoencoder for learning monosemantic features from neural network activations.
    
    This implementation is based on the autoencoder described in:
    "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"
    """
    
    def __init__(
        self, 
        input_dim: int, 
        latent_dim: int = None,
        expansion_factor: float = 8.0,
        l1_coefficient: float = 1e-3,
        bias_decoder: bool = True,
        tied_weights: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the sparse autoencoder.
        
        Args:
            input_dim: Dimension of the input activations
            latent_dim: Dimension of the latent space (number of features to learn)
            expansion_factor: Factor by which to expand the input dimension if latent_dim not provided
            l1_coefficient: Coefficient for L1 sparsity penalty
            bias_decoder: Whether to use bias in the decoder
            tied_weights: Whether to tie the encoder and decoder weights
            device: Device to use for training
        """
        super().__init__()
        
        self.input_dim = input_dim
        
        # If latent_dim is not provided, calculate based on expansion factor
        if latent_dim is None:
            self.latent_dim = int(input_dim * expansion_factor)
        else:
            self.latent_dim = latent_dim
            self.expansion_factor = self.latent_dim / self.input_dim
        
        self.l1_coefficient = l1_coefficient
        self.bias_decoder = bias_decoder
        self.tied_weights = tied_weights
        self.device = device
        
        # Initialize encoder
        self.encoder = nn.Linear(input_dim, self.latent_dim, bias=True)
        
        # Initialize decoder (weights will be tied if requested)
        if tied_weights:
            self.decoder_weight = None  # Will use encoder.weight.T at forward pass
            self.decoder_bias = nn.Parameter(torch.zeros(input_dim)) if bias_decoder else None
        else:
            self.decoder = nn.Linear(self.latent_dim, input_dim, bias=bias_decoder)
            
        # Move model to device
        self.to(device)
        
        # Track feature sparsity and other metrics during training
        self.metrics = {
            'train_loss': [],
            'recon_loss': [],
            'l1_loss': [],
            'feature_sparsity': [],
            'dead_features': [],
            'saturated_features': []
        }
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (reconstructed_input, latent_activations)
        """
        # Encode
        latent = self.encoder(x)
        
        # Apply optional activation/nonlinearity to the latent space
        # (Following the monosemanticity paper, we use ReLU)
        latent = F.relu(latent)
        
        # Decode
        if self.tied_weights:
            # Use the transposed encoder weights
            x_recon = F.linear(
                latent, 
                self.encoder.weight.t(), 
                self.decoder_bias if self.bias_decoder else None
            )
        else:
            x_recon = self.decoder(latent)
            
        return x_recon, latent
        
    def loss_function(
        self, 
        x_recon: torch.Tensor, 
        x: torch.Tensor, 
        latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the loss for the sparse autoencoder.
        
        Args:
            x_recon: Reconstructed input
            x: Original input
            latent: Latent activations
            
        Returns:
            Tuple of (total_loss, reconstruction_loss, l1_loss)
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x)
        
        # L1 sparsity penalty
        l1_loss = self.l1_coefficient * torch.mean(torch.abs(latent))
        
        # Total loss
        total_loss = recon_loss + l1_loss
        
        return total_loss, recon_loss, l1_loss
        
    def train_model(
        self,
        train_activations: torch.Tensor,
        val_activations: Optional[torch.Tensor] = None,
        batch_size: int = 256,
        num_epochs: int = 100,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.0,
        patience: int = 10,
        min_delta: float = 1e-4,
        dead_feature_threshold: float = 1e-5,
        saturated_feature_threshold: float = 0.9,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        log_interval: int = 10,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the sparse autoencoder.
        
        Args:
            train_activations: Training activations
            val_activations: Validation activations (optional)
            batch_size: Batch size for training
            num_epochs: Number of epochs to train for
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            patience: Patience for early stopping
            min_delta: Minimum change in validation loss for early stopping
            dead_feature_threshold: Threshold for considering a feature "dead"
            saturated_feature_threshold: Threshold for considering a feature "saturated"
            checkpoint_dir: Directory to save model checkpoints
            log_interval: How often to log metrics during training
            verbose: Whether to print training progress
            
        Returns:
            Dictionary of training metrics
        """
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(train_activations)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
        )
        
        if val_activations is not None:
            val_dataset = torch.utils.data.TensorDataset(val_activations)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
            )
        
        # Set up for early stopping
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_epoch = 0
        
        # Set up for checkpointing
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Training
            self.train()
            train_loss = 0.0
            train_recon_loss = 0.0
            train_l1_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") if verbose else train_loader
            
            for batch_idx, (data,) in enumerate(progress_bar):
                data = data.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                recon_batch, latent = self(data)
                
                # Compute loss
                loss, recon_loss, l1_loss = self.loss_function(recon_batch, data, latent)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                train_recon_loss += recon_loss.item()
                train_l1_loss += l1_loss.item()
                
                if verbose and batch_idx % log_interval == 0:
                    progress_bar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'recon': f"{recon_loss.item():.4f}",
                        'l1': f"{l1_loss.item():.4f}"
                    })
            
            # Average losses over batches
            train_loss /= len(train_loader)
            train_recon_loss /= len(train_loader)
            train_l1_loss /= len(train_loader)
            
            # Compute feature sparsity metrics
            with torch.no_grad():
                # Sample a subset for feature statistics
                if len(train_loader.dataset) > 10000:
                    indices = torch.randperm(len(train_loader.dataset))[:10000]
                    sample_data = train_activations[indices].to(self.device)
                else:
                    sample_data = train_activations.to(self.device)
                
                # Forward pass on sample
                _, latent = self(sample_data)
                
                # Compute average activation per feature
                avg_activation = torch.mean(latent, dim=0).cpu().numpy()
                
                # Compute sparsity
                feature_sparsity = torch.mean((latent > 0).float()).item()
                
                # Count dead and saturated features
                dead_features = np.sum(avg_activation < dead_feature_threshold)
                saturated_features = np.sum(avg_activation > saturated_feature_threshold)
            
            # Update metrics
            self.metrics['train_loss'].append(train_loss)
            self.metrics['recon_loss'].append(train_recon_loss)
            self.metrics['l1_loss'].append(train_l1_loss)
            self.metrics['feature_sparsity'].append(feature_sparsity)
            self.metrics['dead_features'].append(dead_features)
            self.metrics['saturated_features'].append(saturated_features)
            
            # Validation
            val_loss = 0.0
            if val_activations is not None:
                self.eval()
                with torch.no_grad():
                    for batch_idx, (data,) in enumerate(val_loader):
                        data = data.to(self.device)
                        
                        # Forward pass
                        recon_batch, latent = self(data)
                        
                        # Compute loss
                        loss, _, _ = self.loss_function(recon_batch, data, latent)
                        
                        # Update metrics
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                
                # Check for improvement
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    best_epoch = epoch
                    
                    # Save checkpoint
                    if checkpoint_dir is not None:
                        self.save(checkpoint_dir / f"best_model.pt")
                else:
                    epochs_without_improvement += 1
            
            # Print epoch summary
            if verbose:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{num_epochs} completed in {elapsed:.0f}s - "
                      f"Loss: {train_loss:.4f}, Recon: {train_recon_loss:.4f}, L1: {train_l1_loss:.4f}, "
                      f"Sparsity: {feature_sparsity:.4f}, Dead: {dead_features}/{self.latent_dim}, "
                      f"Saturated: {saturated_features}/{self.latent_dim}")
                
                if val_activations is not None:
                    print(f"Validation Loss: {val_loss:.4f}")
                    
            # Save periodic checkpoint
            if checkpoint_dir is not None and (epoch + 1) % 10 == 0:
                self.save(checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt")
                
            # Early stopping
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
                break
        
        # Save final model
        if checkpoint_dir is not None:
            self.save(checkpoint_dir / "final_model.pt")
            
        # Load best model if validation was used
        if val_activations is not None and checkpoint_dir is not None:
            best_model_path = checkpoint_dir / "best_model.pt"
            if best_model_path.exists():
                self.load(best_model_path)
                print(f"Loaded best model from epoch {best_epoch+1}")
        
        return self.metrics
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input activations to sparse latent features.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Latent activations of shape (batch_size, latent_dim)
        """
        self.eval()
        with torch.no_grad():
            latent = self.encoder(x)
            latent = F.relu(latent)
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent features back to input space.
        
        Args:
            latent: Latent tensor of shape (batch_size, latent_dim)
            
        Returns:
            Reconstructed input of shape (batch_size, input_dim)
        """
        self.eval()
        with torch.no_grad():
            if self.tied_weights:
                x_recon = F.linear(
                    latent, 
                    self.encoder.weight.t(), 
                    self.decoder_bias if self.bias_decoder else None
                )
            else:
                x_recon = self.decoder(latent)
        return x_recon
    
    def get_feature_vectors(self) -> torch.Tensor:
        """
        Get the feature vectors (dictionary elements) learned by the autoencoder.
        
        Returns:
            Tensor of shape (latent_dim, input_dim) where each row is a feature vector
        """
        if self.tied_weights:
            return self.encoder.weight.t().cpu()
        else:
            return self.decoder.weight.cpu()
            
    def get_feature_stats(
        self, 
        activations: torch.Tensor, 
        batch_size: int = 1024
    ) -> Dict[str, np.ndarray]:
        """
        Compute statistics for each learned feature on a dataset.
        
        Args:
            activations: Activations to compute statistics on
            batch_size: Batch size for processing
            
        Returns:
            Dictionary of feature statistics
        """
        self.eval()
        
        dataset = torch.utils.data.TensorDataset(activations)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )
        
        # Collect all latent activations
        all_latents = []
        with torch.no_grad():
            for batch_idx, (data,) in enumerate(loader):
                data = data.to(self.device)
                _, latent = self(data)
                all_latents.append(latent.cpu())
        
        all_latents = torch.cat(all_latents, dim=0).numpy()
        
        # Compute statistics
        mean_activation = np.mean(all_latents, axis=0)
        max_activation = np.max(all_latents, axis=0)
        activation_freq = np.mean(all_latents > 0, axis=0)
        
        return {
            'mean_activation': mean_activation,
            'max_activation': max_activation,
            'activation_freq': activation_freq,
        }
    
    def find_dead_features(
        self, 
        activations: torch.Tensor, 
        threshold: float = 1e-5,
        batch_size: int = 1024
    ) -> np.ndarray:
        """
        Find dead features that rarely activate.
        
        Args:
            activations: Activations to compute statistics on
            threshold: Mean activation threshold below which a feature is considered dead
            batch_size: Batch size for processing
            
        Returns:
            Array of indices of dead features
        """
        stats = self.get_feature_stats(activations, batch_size)
        dead_idx = np.where(stats['mean_activation'] < threshold)[0]
        return dead_idx
    
    def find_most_active_inputs(
        self,
        feature_idx: int,
        activations: torch.Tensor, 
        n_samples: int = 10,
        batch_size: int = 1024
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find inputs that most strongly activate a specific feature.
        
        Args:
            feature_idx: Index of the feature to analyze
            activations: Dataset of activations
            n_samples: Number of top activating samples to return
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (top_activating_inputs, corresponding_feature_activations)
        """
        self.eval()
        
        dataset = torch.utils.data.TensorDataset(activations)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )
        
        # Keep track of top activations and corresponding inputs
        top_activations = torch.zeros(n_samples, device=self.device)
        top_inputs = torch.zeros((n_samples, self.input_dim), device=self.device)
        
        with torch.no_grad():
            for batch_idx, (data,) in enumerate(loader):
                data = data.to(self.device)
                
                # Get feature activations
                _, latent = self(data)
                feature_act = latent[:, feature_idx]
                
                # Update top activations
                if batch_idx == 0 and n_samples <= len(feature_act):
                    # Initialize with first batch
                    top_indices = torch.topk(feature_act, n_samples).indices
                    top_activations = feature_act[top_indices]
                    top_inputs = data[top_indices]
                else:
                    # Compare with current top activations
                    all_activations = torch.cat([top_activations, feature_act])
                    all_inputs = torch.cat([top_inputs, data])
                    
                    # Get top indices
                    top_indices = torch.topk(all_activations, n_samples).indices
                    
                    # Update top activations and inputs
                    top_activations = all_activations[top_indices]
                    top_inputs = all_inputs[top_indices]
        
        return top_inputs.cpu(), top_activations.cpu()
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'l1_coefficient': self.l1_coefficient,
            'bias_decoder': self.bias_decoder,
            'tied_weights': self.tied_weights,
            'metrics': self.metrics
        }
        
        torch.save(save_dict, path)
    
    def load(self, path: Union[str, Path]) -> None:
        """
        Load the model from a file.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model parameters
        self.input_dim = checkpoint['input_dim']
        self.latent_dim = checkpoint['latent_dim']
        self.l1_coefficient = checkpoint['l1_coefficient']
        self.bias_decoder = checkpoint['bias_decoder']
        self.tied_weights = checkpoint['tied_weights']
        
        # Load state dict
        self.load_state_dict(checkpoint['model_state_dict'])
        
        # Load metrics if available
        if 'metrics' in checkpoint:
            self.metrics = checkpoint['metrics']
    
    @classmethod
    def load_from_checkpoint(
        cls, 
        path: Union[str, Path],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> 'SparseAutoencoder':
        """
        Load a model from a checkpoint file.
        
        Args:
            path: Path to load the model from
            device: Device to load the model on
            
        Returns:
            Loaded SparseAutoencoder model
        """
        checkpoint = torch.load(path, map_location=device)
        
        # Create model with same parameters
        model = cls(
            input_dim=checkpoint['input_dim'],
            latent_dim=checkpoint['latent_dim'],
            l1_coefficient=checkpoint['l1_coefficient'],
            bias_decoder=checkpoint['bias_decoder'],
            tied_weights=checkpoint['tied_weights'],
            device=device
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load metrics if available
        if 'metrics' in checkpoint:
            model.metrics = checkpoint['metrics']
            
        return model
    
    def plot_training_metrics(
        self, 
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        Plot training metrics.
        
        Args:
            save_path: Path to save the plot to (optional)
            figsize: Figure size
        """
        if not self.metrics['train_loss']:
            print("No training metrics available")
            return
            
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        # Plot loss
        axs[0, 0].plot(self.metrics['train_loss'], label='Total Loss')
        axs[0, 0].plot(self.metrics['recon_loss'], label='Reconstruction Loss')
        axs[0, 0].plot(self.metrics['l1_loss'], label='L1 Loss')
        axs[0, 0].set_title('Training Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()
        
        # Plot feature sparsity
        axs[0, 1].plot(self.metrics['feature_sparsity'])
        axs[0, 1].set_title('Feature Sparsity')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Fraction of Non-Zero Activations')
        
        # Plot dead features
        axs[1, 0].plot(self.metrics['dead_features'])
        axs[1, 0].set_title('Dead Features')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Number of Dead Features')
        
        # Plot saturated features
        axs[1, 1].plot(self.metrics['saturated_features'])
        axs[1, 1].set_title('Saturated Features')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Number of Saturated Features')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show() 