"""
Video Captioning Module for Surveillance System

This module provides vision-language captioning capabilities for video frames,
including caption generation, sanitization, embedding, and storage.
"""

from .captioning_service import CaptioningService
from .models import Frame, CaptionRecord
from .config import CaptioningConfig

__version__ = "1.0.0"
__all__ = ["CaptioningService", "Frame", "CaptionRecord", "CaptioningConfig"]