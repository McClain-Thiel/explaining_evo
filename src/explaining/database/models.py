from typing import List, Optional, Dict, Any
from datetime import datetime
import json

from sqlmodel import Field, Relationship, SQLModel, Column, JSON


class Sequence(SQLModel, table=True):
    """
    Model representing a genomic sequence.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Sequence data
    sequence: str = Field(index=True)
    sequence_hash: str = Field(index=True, unique=True)  # Hash to uniquely identify sequences
    
    # Metadata
    metadata: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    source: Optional[str] = Field(default=None, index=True)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    features: List["SequenceFeature"] = Relationship(back_populates="sequence")
    
    def set_metadata(self, metadata_dict: Dict[str, Any]) -> None:
        """Set sequence metadata from a dictionary."""
        self.metadata = metadata_dict
    
    def get_metadata_value(self, key: str, default: Any = None) -> Any:
        """Get a specific metadata value."""
        if self.metadata is None:
            return default
        return self.metadata.get(key, default)


class FeatureSet(SQLModel, table=True):
    """
    Model representing a set of features extracted from a model layer.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Feature set information
    name: str = Field(index=True)
    model_name: str = Field(index=True)  # e.g., "evo2_7b"
    layer_name: str = Field(index=True)  # e.g., "blocks.6.mlp"
    
    # Autoencoder info
    expansion_factor: float
    latent_dim: int
    input_dim: int
    l1_coefficient: float
    
    # Configuration details
    config: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    features: List["Feature"] = Relationship(back_populates="feature_set")
    
    def set_config(self, config_dict: Dict[str, Any]) -> None:
        """Set config from a dictionary."""
        self.config = config_dict


class Feature(SQLModel, table=True):
    """
    Model representing a specific feature learned by the autoencoder.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Feature information
    feature_idx: int = Field(index=True)  # Index in the feature set
    feature_set_id: int = Field(foreign_key="featureset.id", index=True)
    
    # Feature statistics
    mean_activation: float
    max_activation: float
    activation_frequency: float
    sparsity: float
    
    # Feature vector (dictionary elements)
    # We store this as a JSON array of floats
    feature_vector: List[float] = Field(sa_column=Column(JSON))
    
    # Optional semantic description of the feature
    description: Optional[str] = None
    
    # Relationships
    feature_set: FeatureSet = Relationship(back_populates="features")
    sequence_features: List["SequenceFeature"] = Relationship(back_populates="feature")
    
    def get_top_weights(self, n: int = 10) -> List[tuple]:
        """
        Get top feature weights by magnitude.
        
        Returns:
            List of (index, weight) tuples sorted by absolute weight magnitude
        """
        if not self.feature_vector:
            return []
        
        weights = [(i, w) for i, w in enumerate(self.feature_vector)]
        return sorted(weights, key=lambda x: abs(x[1]), reverse=True)[:n]


class SequenceFeature(SQLModel, table=True):
    """
    Link model between Sequence and Feature, containing the activation value.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Foreign keys
    sequence_id: int = Field(foreign_key="sequence.id", index=True)
    feature_id: int = Field(foreign_key="feature.id", index=True)
    
    # Activation value for this feature on this sequence
    activation: float
    
    # Relationships
    sequence: Sequence = Relationship(back_populates="features")
    feature: Feature = Relationship(back_populates="sequence_features") 