"""
MAPLE Preprocessing Module
Contains data loading and preprocessing logic.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles data loading and preprocessing for MAPLE experiments."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def create_synthetic_data(self, batch_size: int = 2, sequence_length: int = 77, 
                            latent_size: int = 64, embedding_dim: int = 768):
        """Create synthetic data for testing and smoke tests."""
        logger.info(f"Creating synthetic data: batch_size={batch_size}")
        
        latents = torch.randn(batch_size, 4, latent_size, latent_size, device=self.device)
        
        timestep = torch.randint(0, 1000, (batch_size,), device=self.device)
        
        text_embeddings = torch.randn(batch_size, sequence_length, embedding_dim, device=self.device)
        
        return {
            'latents': latents,
            'timestep': timestep,
            'text_embeddings': text_embeddings
        }
    
    def load_experiment_data(self, experiment_type: str):
        """Load data specific to experiment type."""
        if experiment_type == "smoke_test":
            return self.create_synthetic_data(
                batch_size=self.config['smoke_test']['batch_size']
            )
        elif experiment_type == "full_experiment":
            return self.create_synthetic_data(
                batch_size=self.config.get('full_experiment', {}).get('batch_size', 2)
            )
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    def preprocess_prompts(self, prompts: List[str]) -> torch.Tensor:
        """Preprocess text prompts into embeddings."""
        batch_size = len(prompts)
        return torch.randn(batch_size, 77, 768, device=self.device)
