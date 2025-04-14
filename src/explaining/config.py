"""
Configuration management for the Explaining Evo project.

This module provides Pydantic Settings classes for managing configuration across
the project, with support for environment variables and config files.
"""

import os
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from pydantic import BaseModel, Field, validator, root_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database connection settings."""
    url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/explainable_evo",
        description="Database connection URL"
    )
    echo: bool = Field(
        default=False,
        description="Echo SQL statements"
    )
    
    model_config = SettingsConfigDict(
        env_prefix="DB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class ModelSettings(BaseSettings):
    """Model settings for Evo2."""
    name: str = Field(
        default="evo2_7b",
        description="Name of Evo2 model to use ('evo2_7b' or 'evo2_40b')"
    )
    local_path: Optional[str] = Field(
        default=None,
        description="Optional path to local model checkpoint"
    )
    layer_name: str = Field(
        description="Name of the layer to extract activations from"
    )
    device: str = Field(
        default="auto",
        description="Device to use ('cpu', 'cuda', or 'auto')"
    )
    
    @validator("device")
    def validate_device(cls, v):
        import torch
        if v == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if v not in ["cpu", "cuda"]:
            raise ValueError(f"Device must be 'cpu', 'cuda', or 'auto', got {v}")
        if v == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return v
    
    model_config = SettingsConfigDict(
        env_prefix="MODEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class AutoencoderSettings(BaseSettings):
    """Autoencoder training and inference settings."""
    expansion_factor: float = Field(
        default=8.0,
        description="Expansion factor for autoencoder (latent_dim = input_dim * expansion_factor)"
    )
    l1_coefficient: float = Field(
        default=1e-3,
        description="L1 sparsity coefficient for the autoencoder"
    )
    tied_weights: bool = Field(
        default=False,
        description="Use tied weights in the autoencoder"
    )
    activation_threshold: float = Field(
        default=0.1,
        description="Minimum activation value to store"
    )
    max_features_per_sequence: int = Field(
        default=100,
        description="Maximum number of features to store per sequence"
    )
    epochs: int = Field(
        default=50,
        description="Number of training epochs"
    )
    learning_rate: float = Field(
        default=3e-4,
        description="Learning rate for autoencoder training"
    )
    checkpoint_path: Optional[str] = Field(
        default=None,
        description="Path to load/save autoencoder checkpoint"
    )
    
    model_config = SettingsConfigDict(
        env_prefix="AUTOENCODER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class ProcessingSettings(BaseSettings):
    """General processing settings."""
    batch_size: int = Field(
        default=16,
        description="Batch size for processing"
    )
    max_samples: Optional[int] = Field(
        default=None,
        description="Maximum number of samples to process"
    )
    output_dir: str = Field(
        default="results",
        description="Directory to save results"
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose output"
    )
    
    model_config = SettingsConfigDict(
        env_prefix="PROCESSING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class FeatureAnalysisConfig(BaseSettings):
    """Configuration for feature analysis."""
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    model: ModelSettings
    autoencoder: AutoencoderSettings = Field(default_factory=AutoencoderSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    feature_set_name: Optional[str] = Field(
        default=None,
        description="Name for the feature set (defaults to '{model_name}_{layer_name}')"
    )
    
    @root_validator
    def set_feature_set_name(cls, values):
        if not values.get("feature_set_name"):
            model = values.get("model")
            if model:
                values["feature_set_name"] = f"{model.name}_{model.layer_name}"
        return values
    
    @classmethod
    def from_file(cls, config_file: Union[str, Path], **overrides):
        """Load configuration from a file with optional overrides."""
        import json
        import yaml
        
        config_file = Path(config_file)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        # Load config from file based on extension
        if config_file.suffix.lower() in ['.yaml', '.yml']:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
        elif config_file.suffix.lower() == '.json':
            with open(config_file, 'r') as f:
                config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_file.suffix}")
        
        # Apply overrides
        for key, value in overrides.items():
            if '.' in key:
                # Handle nested override like "database.url"
                parts = key.split('.')
                section = config_data
                for part in parts[:-1]:
                    if part not in section:
                        section[part] = {}
                    section = section[part]
                section[parts[-1]] = value
            else:
                config_data[key] = value
        
        return cls(**config_data)
    
    def save_to_file(self, config_file: Union[str, Path], format: str = 'yaml'):
        """Save configuration to a file."""
        import json
        import yaml
        
        config_file = Path(config_file)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary
        config_dict = self.model_dump()
        
        # Save based on format
        if format.lower() == 'yaml':
            with open(config_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif format.lower() == 'json':
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class QuerySettings(BaseSettings):
    """Settings for database query operations."""
    database_url: Optional[str] = Field(
        default=None,
        description="Database connection URL (overrides env variable)"
    )
    min_activation: float = Field(
        default=0.1,
        description="Minimum activation threshold for features"
    )
    limit: int = Field(
        default=10,
        description="Maximum number of results to return"
    )
    
    model_config = SettingsConfigDict(
        env_prefix="QUERY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class FeatureAnalysisSettings(BaseSettings):
    """Settings for feature analysis operations."""
    database_url: Optional[str] = Field(
        default=None, 
        description="Database connection URL (overrides env variable)"
    )
    top_n: int = Field(
        default=5,
        description="Number of top features to return"
    )
    n_clusters: int = Field(
        default=10,
        description="Number of clusters for K-means"
    )
    method: str = Field(
        default="tsne",
        description="Dimensionality reduction method"
    )
    n_components: int = Field(
        default=2,
        description="Number of components for dimensionality reduction"
    )
    output_format: str = Field(
        default="png",
        description="Output format for visualizations"
    )
    
    @validator("method")
    def validate_method(cls, v):
        allowed = ["pca", "tsne", "umap"]
        if v.lower() not in allowed:
            raise ValueError(f"Method must be one of {allowed}, got {v}")
        return v.lower()
    
    model_config = SettingsConfigDict(
        env_prefix="FEATURE_ANALYSIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    ) 