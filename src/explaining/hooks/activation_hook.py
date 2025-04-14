import torch
import numpy as np
from typing import Dict, List, Optional, Union, Callable
import torch.nn as nn


class ActivationHook:
    def __init__(self, layer_name: str):
        self.layer_name = layer_name
        self.activations = []
        self.hook_handle = None
        
    def hook_fn(self, module, input, output):
        # If output is a tuple, take the first element
        if isinstance(output, tuple):
            output = output[0]
        self.activations.append(output.detach().clone())
    
    def register_hook(self, model: nn.Module) -> None:
        """Register the hook to the specified layer in the model"""
        layer = model.get_submodule(self.layer_name)
        self.hook_handle = layer.register_forward_hook(self.hook_fn)
        
    def remove_hook(self) -> None:
        """Remove the hook from the model"""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
            
    def clear_activations(self) -> None:
        """Clear stored activations"""
        self.activations = []
        
    def get_activations(self) -> List[torch.Tensor]:
        """Get the stored activations"""
        return self.activations
    
    def get_flat_activations(self) -> torch.Tensor:
        """Get activations flattened into a single tensor 
        of shape (n_samples, n_neurons)"""
        if not self.activations:
            return torch.tensor([])
            
        # Handle different activation shapes appropriately
        flat_activations = []
        for act in self.activations:
            # For 3D activations (batch_size, seq_len, hidden_dim)
            if len(act.shape) == 3:
                # Reshape to (batch_size * seq_len, hidden_dim)
                flat_act = act.reshape(-1, act.shape[-1])
                flat_activations.append(flat_act)
            # For 2D activations (batch_size, hidden_dim)
            elif len(act.shape) == 2:
                flat_activations.append(act)
            # For other shapes, raise an error
            else:
                raise ValueError(f"Unexpected activation shape: {act.shape}")
                
        return torch.cat(flat_activations, dim=0)
    
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hook()