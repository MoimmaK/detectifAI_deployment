"""
Configuration settings for video captioning module
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CaptioningConfig:
    """Configuration for captioning service"""
    
    # Vision model settings
    vision_model_name: str = "Salesforce/blip-image-captioning-base"
    vision_device: str = "cpu"  # or "cuda" if available
    vision_batch_size: int = 4
    
    # LLM settings for sanitization
    llm_model_name: str = "microsoft/DialoGPT-medium"
    llm_device: str = "cpu"
    llm_max_tokens: int = 150
    llm_temperature: float = 0.1
    
    # Embedding settings
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_normalize: bool = True
    
    # Database settings
    db_connection_string: Optional[str] = None
    vector_db_path: Optional[str] = "./vector_store"
    
    # Processing settings
    max_concurrent_requests: int = 10
    enable_async_processing: bool = True
    log_rejected_captions: bool = True
    
    # Safety prompt template
    safety_prompt_template: str = """You are a surveillance captioning assistant. Rewrite the following caption to be neutral, objective, and safe.

Rules:
- Do NOT mention gender, race, skin color, clothing, age, or physical appearance.
- Do NOT make identity assumptions.
- Only describe observable actions, movements, interactions, and objects.
- Keep the caption concise (1–2 sentences).

Caption: {raw_caption}

Rewritten caption:"""