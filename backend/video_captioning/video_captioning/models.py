"""
Data models for video captioning module
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import numpy as np
from PIL import Image


@dataclass
class Frame:
    """Represents a video frame with metadata"""
    frame_id: str
    timestamp: datetime
    video_id: str
    image: Image.Image
    
    def __post_init__(self):
        if not isinstance(self.image, Image.Image):
            raise ValueError("image must be a PIL Image object")


@dataclass
class CaptionRecord:
    """Represents a processed caption with embeddings"""
    caption_id: str
    video_id: str
    frame_id: str
    timestamp: datetime
    raw_caption: str
    sanitized_caption: str
    embedding: np.ndarray
    created_at: datetime
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage"""
        return {
            'caption_id': self.caption_id,
            'video_id': self.video_id,
            'frame_id': self.frame_id,
            'timestamp': self.timestamp.isoformat(),
            'raw_caption': self.raw_caption,
            'sanitized_caption': self.sanitized_caption,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class ProcessingResult:
    """Result of frame processing operation"""
    success: bool
    caption_records: List[CaptionRecord]
    errors: List[str]
    processing_time: float