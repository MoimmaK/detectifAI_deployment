"""
Report Generation Configuration

Defines all configuration parameters for the report generation module,
including LLM settings, paths, and report formatting options.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path


@dataclass
class LLMConfig:
    """Configuration for the local LLM engine."""
    
    # Model selection - Qwen2.5-3B-Instruct recommended for speed + quality balance
    # Alternative: Phi-3-mini-4k-instruct (MIT license, slightly larger)
    model_name: str = "qwen2.5-3b-instruct-q4_k_m.gguf"
    
    # HuggingFace repo for downloading
    hf_repo: str = "Qwen/Qwen2.5-3B-Instruct-GGUF"
    hf_filename: str = "qwen2.5-3b-instruct-q4_k_m.gguf"  # ~2GB quantized
    
    # Alternative model (MIT license, more permissive)
    alt_model_name: str = "Phi-3-mini-4k-instruct-q4.gguf"
    alt_hf_repo: str = "microsoft/Phi-3-mini-4k-instruct-gguf"
    alt_hf_filename: str = "Phi-3-mini-4k-instruct-q4.gguf"  # ~2.3GB
    
    # Local model path
    models_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(__file__), 'models'
    ))
    
    # LLM inference parameters
    n_ctx: int = 4096  # Context window size
    n_threads: int = 4  # CPU threads (adjust based on your system)
    n_gpu_layers: int = 0  # Set > 0 if you have GPU with CUDA
    temperature: float = 0.3  # Slightly higher for faster generation
    top_p: float = 0.9
    max_tokens: int = 512  # Reduced for faster generation
    repeat_penalty: float = 1.1
    
    # Timeout settings
    timeout_seconds: int = 60  # Max time for LLM generation
    
    @property
    def model_path(self) -> str:
        """Get full path to model file."""
        return os.path.join(self.models_dir, self.model_name)
    
    @property
    def alt_model_path(self) -> str:
        """Get full path to alternative model file."""
        return os.path.join(self.models_dir, self.alt_model_name)


@dataclass
class ReportConfig:
    """Configuration for report generation and export."""
    
    # LLM configuration
    llm: LLMConfig = field(default_factory=LLMConfig)
    
    # Report output settings
    output_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'video_processing_outputs', 'reports'
    ))
    
    # Template paths
    templates_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(__file__), 'templates'
    ))
    prompts_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(__file__), 'prompts'
    ))
    
    # Report content settings
    include_executive_summary: bool = True
    include_timeline: bool = True
    include_evidence_images: bool = True
    include_observations: bool = True
    include_face_crops: bool = True
    max_images_per_event: int = 3
    max_events_in_report: int = 50
    
    # Image settings
    thumbnail_width: int = 400
    thumbnail_quality: int = 85
    
    # PDF settings
    pdf_page_size: str = "A4"
    pdf_margin_mm: int = 20
    
    # Report metadata
    organization_name: str = "DetectifAI Security System"
    report_classification: str = "CONFIDENTIAL"
    
    # MongoDB connection (uses existing DetectifAI config)
    use_database: bool = True
    
    def __post_init__(self):
        """Create necessary directories."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.llm.models_dir, exist_ok=True)
        os.makedirs(self.templates_dir, exist_ok=True)
        os.makedirs(self.prompts_dir, exist_ok=True)


# Default configuration instance
default_config = ReportConfig()


def get_report_config(**kwargs) -> ReportConfig:
    """
    Get report configuration with optional overrides.
    
    Args:
        **kwargs: Override any config parameter
        
    Returns:
        ReportConfig instance
    """
    config = ReportConfig()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.llm, key):
            setattr(config.llm, key, value)
    
    return config
