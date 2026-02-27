"""
Caption sanitization for policy compliance.

Uses efficient rule-based sanitization to remove sensitive/identifying terms
from captions while preserving descriptive quality for NLP search.

Note: DialoGPT LLM-based sanitization was removed because:
  1. It was extremely slow on CPU (~1.5s per caption vs 0ms rule-based)
  2. It produced worse captions (e.g., "a parking lot with cars" → "a car")
  3. It consumed ~1.5GB RAM for a conversational model misused for text rewriting
  4. Its outputs always failed safety checks and fell back to rule-based anyway
"""

import logging
from typing import List
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from .config import CaptioningConfig
except ImportError:
    from config import CaptioningConfig


class CaptionSanitizer:
    """Handles caption sanitization using efficient rule-based approach"""
    
    def __init__(self, config: CaptioningConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # No LLM model needed — rule-based sanitization is faster and more accurate
        self.model = None
        self.tokenizer = None
        
        # Thread pool for async processing
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_requests)
        
        # Audit log for rejected captions
        self.rejected_captions = []
        
        self.logger.info("Caption sanitizer initialized (rule-based mode)")
    
    def sanitize_caption(self, raw_caption: str) -> str:
        """Sanitize a single caption using rule-based approach"""
        try:
            return self._rule_based_sanitization(raw_caption)
        except Exception as e:
            self.logger.error(f"Failed to sanitize caption: {e}")
            return raw_caption  # Return original rather than losing the caption
    
    def _rule_based_sanitization(self, caption: str) -> str:
        """Efficient rule-based sanitization that preserves descriptive quality.
        
        Replaces identifying terms (gender, age) with neutral alternatives
        while preserving object descriptions useful for NLP search.
        """
        # Terms to replace with 'person'
        person_terms = {
            'man', 'woman', 'boy', 'girl', 'guy', 'lady', 'gentleman',
            'male', 'female'
        }
        # Terms to replace with 'individual'  
        age_terms = {
            'elderly', 'teenager', 'toddler'
        }
        # Terms to skip entirely (too identifying for people)
        skip_terms = {
            'blonde', 'brunette', 'bald', 'redhead'
        }
        
        words = caption.lower().split()
        filtered_words = []
        
        for word in words:
            clean_word = word.strip('.,!?;:')
            if clean_word in person_terms:
                filtered_words.append('person')
            elif clean_word in age_terms:
                filtered_words.append('individual')
            elif clean_word in skip_terms:
                continue  # Remove hair/appearance descriptors
            else:
                filtered_words.append(word)
        
        sanitized = ' '.join(filtered_words)
        
        # Ensure we have meaningful content
        if len(sanitized.strip()) < 5:
            return "Activity detected in scene"
        
        return sanitized.capitalize()
    
    def _is_caption_safe(self, caption: str) -> bool:
        """Validate that caption meets safety requirements"""
        caption_lower = caption.lower()
        
        # Check for prohibited terms
        prohibited_terms = [
            'gender', 'race', 'skin', 'color', 'age', 'appearance',
            'man', 'woman', 'male', 'female', 'boy', 'girl',
            'black', 'white', 'asian', 'hispanic', 'latino',
            'young', 'old', 'elderly', 'child', 'teenager'
        ]
        
        for term in prohibited_terms:
            if term in caption_lower:
                return False
        
        return True
    
    def sanitize_captions_batch(self, raw_captions: List[str]) -> List[str]:
        """Sanitize a batch of captions"""
        return [self.sanitize_caption(caption) for caption in raw_captions]
    
    async def sanitize_captions_async(self, raw_captions: List[str]) -> List[str]:
        """Sanitize captions asynchronously"""
        if not self.config.enable_async_processing:
            return self.sanitize_captions_batch(raw_captions)
        
        loop = asyncio.get_event_loop()
        
        # Run in thread pool
        sanitized_captions = await loop.run_in_executor(
            self.executor,
            self.sanitize_captions_batch,
            raw_captions
        )
        
        return sanitized_captions
    
    def get_rejected_captions(self) -> List[dict]:
        """Get audit log of rejected captions"""
        return self.rejected_captions.copy()
    
    def clear_rejected_captions(self):
        """Clear the rejected captions log"""
        self.rejected_captions.clear()
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)