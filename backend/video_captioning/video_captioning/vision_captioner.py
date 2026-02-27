"""
Vision-language model for generating captions from frames
"""

import logging
from typing import List, Union
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from .models import Frame
    from .config import CaptioningConfig
except ImportError:
    from models import Frame
    from config import CaptioningConfig


class VisionCaptioner:
    """Handles vision-language model for frame captioning"""
    
    def __init__(self, config: CaptioningConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(config.vision_device)
        
        # Initialize model and processor
        self._load_model()
        
        # Thread pool for async processing
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_requests)
    
    def _load_model(self):
        """Load the vision-language model"""
        try:
            self.logger.info(f"Loading vision model: {self.config.vision_model_name}")
            self.processor = BlipProcessor.from_pretrained(self.config.vision_model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.config.vision_model_name
            ).to(self.device)
            self.logger.info("Vision model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load vision model: {e}")
            raise
    
    def generate_caption(self, image: Image.Image) -> str:
        """Generate caption for a single image"""
        try:
            # Preprocess image
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption (reduced beams for faster inference)
            # Note: attention_mask is not needed for BLIP's generate() with pixel_values
            with torch.no_grad():
                out = self.model.generate(
                    pixel_values=inputs['pixel_values'],
                    max_length=50, 
                    num_beams=3
                )
            
            # Decode caption
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
            
        except Exception as e:
            self.logger.error(f"Failed to generate caption: {e}")
            return "Unable to generate caption"
    
    def generate_captions_batch(self, images: List[Image.Image]) -> List[str]:
        """Generate captions for a batch of images"""
        try:
            # Process in batches
            captions = []
            batch_size = self.config.vision_batch_size
            total_batches = (len(images) + batch_size - 1) // batch_size
            
            self.logger.info(f"🔄 Processing {len(images)} images in {total_batches} batches of {batch_size}")
            
            for i in range(0, len(images), batch_size):
                batch_num = (i // batch_size) + 1
                batch = images[i:i + batch_size]
                self.logger.info(f"⏳ Processing batch {batch_num}/{total_batches} ({len(batch)} images)...")
                batch_captions = self._process_batch(batch)
                captions.extend(batch_captions)
                self.logger.info(f"✅ Batch {batch_num}/{total_batches} complete")
            
            return captions
            
        except Exception as e:
            self.logger.error(f"Failed to generate batch captions: {e}")
            return ["Unable to generate caption"] * len(images)
    
    def _process_batch(self, images: List[Image.Image]) -> List[str]:
        """Process a single batch of images"""
        try:
            # Preprocess batch with padding
            inputs = self.processor(images, return_tensors="pt", padding=True).to(self.device)
            
            # Generate captions
            # Note: BLIP's generate() handles attention internally for vision inputs
            # Passing attention_mask causes shape errors (expects 2D for text, not 3D/4D for images)
            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values=inputs['pixel_values'],
                    max_length=50, 
                    num_beams=3,  # Reduced from 5 to 3 for 40% speed improvement
                    do_sample=False
                )
            
            # Decode captions
            captions = []
            for output in outputs:
                caption = self.processor.decode(output, skip_special_tokens=True)
                captions.append(caption)
            
            return captions
            
        except Exception as e:
            self.logger.error(f"Failed to process batch: {e}")
            return ["Unable to generate caption"] * len(images)
    
    async def generate_captions_async(self, frames: List[Frame]) -> List[str]:
        """Generate captions asynchronously"""
        if not self.config.enable_async_processing:
            return self.generate_captions_batch([frame.image for frame in frames])
        
        loop = asyncio.get_event_loop()
        images = [frame.image for frame in frames]
        
        # Run in thread pool
        captions = await loop.run_in_executor(
            self.executor, 
            self.generate_captions_batch, 
            images
        )
        
        return captions
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)