import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import os
import tqdm
from pathlib import Path

from src.evo2.evo2.models import Evo2
from src.explaining.hooks.activation_hook import ActivationHook


class ExplainableEvo:
    """Class to extract activations from the Evo2 model for later use in sparse autoencoder training."""
    
    def __init__(
        self, 
        model_name: str = "evo2_7b", 
        local_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the activation extractor for Evo2.
        
        Args:
            model_name: Name of the Evo2 model to use
            local_path: Optional path to a local model checkpoint
            device: Device to run the model on
        """
        self.model = Evo2(model_name=model_name, local_path=local_path)
        self.device = device
        self.hooks = {}
        self.tokenizer = self.model.tokenizer
        
        # Get available layers for hooking
        self._identify_available_layers()
        
    def _identify_available_layers(self):
        """Identify the available layers in the model for hooking."""
        # Get the keys from the model's state dict
        self.available_layers = list(self.model.model.state_dict().keys())
        
        # Filter to include only weight matrices which are more likely to be layers
        self.available_layers = [layer.split('.weight')[0] for layer in self.available_layers 
                                if '.weight' in layer and not 'norm' in layer]
        
        # Remove duplicates
        self.available_layers = list(set(self.available_layers))
        
    def list_available_layers(self) -> List[str]:
        """Return a list of available layers for hooking."""
        return self.available_layers
        
    def add_hook(self, layer_name: str) -> None:
        """
        Add a hook to a specific layer in the model.
        
        Args:
            layer_name: Name of the layer to hook
        """
        hook = ActivationHook(layer_name)
        hook.register_hook(self.model.model)
        self.hooks[layer_name] = hook
        
    def remove_hook(self, layer_name: str) -> None:
        """
        Remove a hook from a specific layer.
        
        Args:
            layer_name: Name of the layer to remove the hook from
        """
        if layer_name in self.hooks:
            self.hooks[layer_name].remove_hook()
            del self.hooks[layer_name]
            
    def remove_all_hooks(self) -> None:
        """Remove all hooks from the model."""
        for layer_name in list(self.hooks.keys()):
            self.remove_hook(layer_name)
            
    def get_activations(self, layer_name: str) -> List[torch.Tensor]:
        """
        Get activations from a specific layer.
        
        Args:
            layer_name: Name of the layer to get activations from
            
        Returns:
            List of activation tensors
        """
        if layer_name not in self.hooks:
            raise ValueError(f"No hook registered for layer {layer_name}")
            
        return self.hooks[layer_name].get_activations()
        
    def get_flat_activations(self, layer_name: str) -> torch.Tensor:
        """
        Get flattened activations from a specific layer.
        
        Args:
            layer_name: Name of the layer to get activations from
            
        Returns:
            Tensor of shape (n_samples, n_neurons)
        """
        if layer_name not in self.hooks:
            raise ValueError(f"No hook registered for layer {layer_name}")
            
        return self.hooks[layer_name].get_flat_activations()
        
    def clear_activations(self, layer_name: Optional[str] = None) -> None:
        """
        Clear activations from a specific layer or all layers.
        
        Args:
            layer_name: Name of the layer to clear activations from, or None to clear all
        """
        if layer_name is None:
            for hook in self.hooks.values():
                hook.clear_activations()
        elif layer_name in self.hooks:
            self.hooks[layer_name].clear_activations()
        else:
            raise ValueError(f"No hook registered for layer {layer_name}")
            
    def collect_activations(
        self, 
        sequences: List[str], 
        layer_names: List[str],
        batch_size: int = 1,
        show_progress: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Collect activations from the model for a set of sequences.
        
        Args:
            sequences: List of DNA sequences to process
            layer_names: List of layer names to collect activations from
            batch_size: Batch size for processing
            show_progress: Whether to show a progress bar
            
        Returns:
            Dictionary mapping layer names to activation tensors
        """
        # Clear any existing activations
        self.remove_all_hooks()
        
        # Add hooks for each layer
        for layer_name in layer_names:
            self.add_hook(layer_name)
            
        # Process sequences in batches
        all_activations = {layer_name: [] for layer_name in layer_names}
        
        try:
            iterator = range(0, len(sequences), batch_size)
            if show_progress:
                iterator = tqdm.tqdm(iterator, desc="Collecting activations")
                
            for i in iterator:
                batch = sequences[i:i+batch_size]
                
                # Tokenize the sequences
                tokens = [self.tokenizer.encode(seq) for seq in batch]
                max_len = max(len(t) for t in tokens)
                
                # Pad sequences to the same length
                padded_tokens = [t + [0] * (max_len - len(t)) for t in tokens]
                input_ids = torch.tensor(padded_tokens, device=self.device)
                
                # Run the model forward pass
                with torch.no_grad():
                    # Call the model's forward method
                    self.model(input_ids)
                    
                # Collect activations for each layer
                for layer_name in layer_names:
                    act = self.hooks[layer_name].get_activations()
                    self.hooks[layer_name].clear_activations()
                    all_activations[layer_name].extend(act)
                    
            # Convert lists of tensors to single tensors
            for layer_name in layer_names:
                all_activations[layer_name] = torch.cat([
                    a.reshape(-1, a.shape[-1]) for a in all_activations[layer_name]
                ], dim=0)
                
            return all_activations
            
        finally:
            # Clean up hooks
            self.remove_all_hooks()
            
    def save_activations(
        self, 
        activations: Dict[str, torch.Tensor], 
        output_dir: Union[str, Path],
        filename_prefix: str = "activations"
    ) -> None:
        """
        Save activations to disk.
        
        Args:
            activations: Dictionary mapping layer names to activation tensors
            output_dir: Directory to save activations to
            filename_prefix: Prefix for saved files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for layer_name, act in activations.items():
            # Create a safe filename by replacing non-alphanumeric characters
            safe_layer_name = ''.join(c if c.isalnum() else '_' for c in layer_name)
            filename = f"{filename_prefix}_{safe_layer_name}.pt"
            torch.save(act, output_dir / filename)
            
    def load_activations(
        self, 
        input_dir: Union[str, Path], 
        layer_names: Optional[List[str]] = None,
        filename_prefix: str = "activations"
    ) -> Dict[str, torch.Tensor]:
        """
        Load activations from disk.
        
        Args:
            input_dir: Directory to load activations from
            layer_names: Optional list of layer names to load, or None to load all
            filename_prefix: Prefix for saved files
            
        Returns:
            Dictionary mapping layer names to activation tensors
        """
        input_dir = Path(input_dir)
        activations = {}
        
        if layer_names is None:
            # Load all activation files matching the prefix
            files = list(input_dir.glob(f"{filename_prefix}_*.pt"))
            for file in files:
                # Extract layer name from filename
                layer_name = file.stem.replace(f"{filename_prefix}_", "")
                activations[layer_name] = torch.load(file)
        else:
            # Load only the specified layers
            for layer_name in layer_names:
                safe_layer_name = ''.join(c if c.isalnum() else '_' for c in layer_name)
                filename = f"{filename_prefix}_{safe_layer_name}.pt"
                file_path = input_dir / filename
                if file_path.exists():
                    activations[layer_name] = torch.load(file_path)
                else:
                    raise FileNotFoundError(f"Activation file not found: {file_path}")
                    
        return activations 