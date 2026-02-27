"""
Video Captioning Module - Parent Package
"""

# This allows importing from video_captioning
try:
    from .video_captioning.captioning_service import CaptioningService
    from .video_captioning.models import Frame, CaptionRecord
    from .video_captioning.config import CaptioningConfig
    
    __all__ = ["CaptioningService", "Frame", "CaptionRecord", "CaptioningConfig"]
except ImportError as e:
    # Fallback for direct imports
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'video_captioning'))
    
    from captioning_service import CaptioningService
    from models import Frame, CaptionRecord
    from config import CaptioningConfig
    
    __all__ = ["CaptioningService", "Frame", "CaptionRecord", "CaptioningConfig"]
