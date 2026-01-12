# BDH Narrative Consistency Classifier Configuration
# Copyright 2026 - Kharagpur Data Science Hackathon

import dataclasses
from typing import Optional

@dataclasses.dataclass
class ModelConfig:
    """BDH Model hyperparameters"""
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 50257  # GPT-2 tokenizer vocab size

@dataclasses.dataclass
class TemporalScaleConfig:
    """Multi-scale temporal dynamics settings"""
    # Sentence level (fine-grained)
    sentence_window: int = 64
    sentence_stride: int = 32
    
    # Paragraph level (medium-grained)
    paragraph_window: int = 256
    paragraph_stride: int = 128
    
    # Chapter level (coarse-grained)
    chapter_window: int = 1024
    chapter_stride: int = 512

@dataclasses.dataclass
class HebbianConfig:
    """Hebbian learning checkpoint settings"""
    checkpoint_interval: int = 100  # Steps between checkpoints
    max_checkpoints: int = 50  # Maximum checkpoints to retain
    delta_compression: bool = True  # Use compressed deltas
    learning_rate: float = 0.01  # Hebbian learning rate
    decay_factor: float = 0.99  # Weight decay for stability

@dataclasses.dataclass
class PathwayConfig:
    """Pathway streaming configuration"""
    batch_size: int = 8
    buffer_size: int = 1000
    refresh_interval_ms: int = 100

@dataclasses.dataclass
class GeminiConfig:
    """LangGraph + Gemini settings"""
    model_name: str = "gemini-2.0-flash"  # Gemini 2.0 Flash (latest fast model)
    api_key: str = "" # User provided API key
    max_output_tokens: int = 1024
    temperature: float = 0.3
    top_p: float = 0.9
    top_k_evidence: int = 5  # Top-k supporting sentences

@dataclasses.dataclass
class TrainingConfig:
    """Training hyperparameters"""
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 4
    gradient_clip: float = 1.0
    warmup_steps: int = 100
    device: str = "cuda"  # or "cpu"

@dataclasses.dataclass
class Config:
    """Master configuration combining all settings"""
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    temporal: TemporalScaleConfig = dataclasses.field(default_factory=TemporalScaleConfig)
    hebbian: HebbianConfig = dataclasses.field(default_factory=HebbianConfig)
    pathway: PathwayConfig = dataclasses.field(default_factory=PathwayConfig)
    gemini: GeminiConfig = dataclasses.field(default_factory=GeminiConfig)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
    
    # Paths
    data_dir: str = "Dataset"
    books_dir: str = "Dataset/Books"
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"

def get_config() -> Config:
    """Get default configuration"""
    return Config()
