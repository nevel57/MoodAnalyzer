"""
Simple configuration for sentiment analysis.
"""

import os
from dataclasses import dataclass


@dataclass
class Config:
    # Model settings
    FAST_MODEL_PATH: str = "models/fast_model.joblib"

    # Routing settings
    MIN_CONFIDENCE: float = 0.7  # If confidence < this, use accurate model

    # Text processing
    MAX_TEXT_LENGTH: int = 1000

    @classmethod
    def from_env(cls):
        """Create config from environment variables."""
        return cls()


# Global config instance
config = Config.from_env()