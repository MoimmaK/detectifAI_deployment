"""
Main captioning service that orchestrates the entire pipeline
"""

import logging
import asyncio
import uuid
from datetime import datetime
from typing import List
import time

try:
    from .models import Frame, CaptionRecord, ProcessingResult
    from .config import CaptioningConfig
    from .vision_captioner import VisionCaptioner
    from .caption_sanitizer import CaptionSanitizer
    from .embedding_generator import EmbeddingGenerator
    from .mongodb_storage import MongoDBCaptionStorage
except ImportError:
    from models import Frame, CaptionRecord, ProcessingResult
    from config import CaptioningConfig
    from vision_captioner import VisionCaptioner
    from caption_sanitizer import CaptionSanitizer
    from embedding_generator import EmbeddingGenerator
    from mongodb_storage import MongoDBCaptionStorage


class CaptioningService:
    """Main service for video frame captioning pipeline"""
    
    def __init__(self, config: CaptioningConfig = None, db_manager=None):
        self.config = config or CaptioningConfig()
        self.logger = logging.getLogger(__name__)
        self.db_manager = db_manager
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize all pipeline components"""
        try:
            self.logger.info("Initializing captioning service components...")
            
            # Initialize vision captioner
            self.logger.info("Loading vision captioner...")
            self.vision_captioner = VisionCaptioner(self.config)
            self.logger.info("✅ Vision captioner loaded")
            
            # Initialize caption sanitizer
            self.logger.info("Loading caption sanitizer...")
            self.caption_sanitizer = CaptionSanitizer(self.config)
            self.logger.info("✅ Caption sanitizer loaded")
            
            # Initialize embedding generator
            self.logger.info("Loading embedding generator...")
            self.embedding_generator = EmbeddingGenerator(self.config)
            self.logger.info("✅ Embedding generator loaded")
            
            # Initialize MongoDB storage with FAISS
            self.logger.info("Initializing MongoDB storage...")
            self.storage = MongoDBCaptionStorage(self.config, db_manager=self.db_manager)
            self.logger.info("✅ MongoDB storage initialized")
            
            self.logger.info("✅ All components initialized successfully (MongoDB + FAISS)")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def process_frames(self, frames: List[Frame]) -> ProcessingResult:
        """Process a batch of frames through the complete pipeline"""
        start_time = time.time()
        errors = []
        caption_records = []
        
        try:
            self.logger.info(f"Processing {len(frames)} frames")
            
            # Step 1: Generate raw captions
            self.logger.debug("Generating raw captions...")
            raw_captions = self.vision_captioner.generate_captions_batch(
                [frame.image for frame in frames]
            )
            
            # Step 2: Sanitize captions
            self.logger.debug("Sanitizing captions...")
            sanitized_captions = self.caption_sanitizer.sanitize_captions_batch(
                raw_captions
            )
            
            # Step 3: Generate embeddings
            self.logger.debug("Generating embeddings...")
            embeddings = self.embedding_generator.generate_embeddings_batch(
                sanitized_captions
            )
            
            # Step 4: Create caption records
            self.logger.debug("Creating caption records...")
            for i, frame in enumerate(frames):
                try:
                    record = CaptionRecord(
                        caption_id=str(uuid.uuid4()),
                        video_id=frame.video_id,
                        frame_id=frame.frame_id,
                        timestamp=frame.timestamp,
                        raw_caption=raw_captions[i],
                        sanitized_caption=sanitized_captions[i],
                        embedding=embeddings[i],
                        created_at=datetime.now()
                    )
                    caption_records.append(record)
                    
                except Exception as e:
                    error_msg = f"Failed to create record for frame {frame.frame_id}: {e}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)
            
            # Step 5: Store records
            if caption_records:
                self.logger.debug("Storing caption records...")
                stored_count = self.storage.store_caption_records_batch(caption_records)
                
                if stored_count != len(caption_records):
                    error_msg = f"Only stored {stored_count}/{len(caption_records)} records"
                    self.logger.warning(error_msg)
                    errors.append(error_msg)
            
            # Log rejected captions if enabled
            if self.config.log_rejected_captions:
                rejected = self.caption_sanitizer.get_rejected_captions()
                for rejection in rejected:
                    self.storage.log_rejected_caption(
                        rejection['raw'],
                        rejection['sanitized'],
                        rejection['reason']
                    )
            
            processing_time = time.time() - start_time
            
            self.logger.info(
                f"Processed {len(frames)} frames in {processing_time:.2f}s, "
                f"created {len(caption_records)} records, {len(errors)} errors"
            )
            
            return ProcessingResult(
                success=len(errors) == 0,
                caption_records=caption_records,
                errors=errors,
                processing_time=processing_time
            )
            
        except Exception as e:
            error_msg = f"Pipeline processing failed: {e}"
            self.logger.error(error_msg)
            errors.append(error_msg)
            
            return ProcessingResult(
                success=False,
                caption_records=caption_records,
                errors=errors,
                processing_time=time.time() - start_time
            )
    
    async def process_frames_async(self, frames: List[Frame]) -> ProcessingResult:
        """Process frames asynchronously"""
        start_time = time.time()
        errors = []
        caption_records = []
        
        try:
            self.logger.info(f"Processing {len(frames)} frames asynchronously")
            
            # Run all steps concurrently where possible
            tasks = []
            
            # Step 1: Generate raw captions
            caption_task = self.vision_captioner.generate_captions_async(frames)
            tasks.append(caption_task)
            
            # Wait for captions to complete before sanitization
            raw_captions = await caption_task
            
            # Step 2: Sanitize captions
            sanitize_task = self.caption_sanitizer.sanitize_captions_async(raw_captions)
            sanitized_captions = await sanitize_task
            
            # Step 3: Generate embeddings
            embedding_task = self.embedding_generator.generate_embeddings_async(
                sanitized_captions
            )
            embeddings = await embedding_task
            
            # Step 4: Create caption records
            for i, frame in enumerate(frames):
                try:
                    record = CaptionRecord(
                        caption_id=str(uuid.uuid4()),
                        video_id=frame.video_id,
                        frame_id=frame.frame_id,
                        timestamp=frame.timestamp,
                        raw_caption=raw_captions[i],
                        sanitized_caption=sanitized_captions[i],
                        embedding=embeddings[i],
                        created_at=datetime.now()
                    )
                    caption_records.append(record)
                    
                except Exception as e:
                    error_msg = f"Failed to create record for frame {frame.frame_id}: {e}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)
            
            # Step 5: Store records
            if caption_records:
                stored_count = self.storage.store_caption_records_batch(caption_records)
                
                if stored_count != len(caption_records):
                    error_msg = f"Only stored {stored_count}/{len(caption_records)} records"
                    self.logger.warning(error_msg)
                    errors.append(error_msg)
            
            processing_time = time.time() - start_time
            
            self.logger.info(
                f"Async processed {len(frames)} frames in {processing_time:.2f}s"
            )
            
            return ProcessingResult(
                success=len(errors) == 0,
                caption_records=caption_records,
                errors=errors,
                processing_time=processing_time
            )
            
        except Exception as e:
            error_msg = f"Async pipeline processing failed: {e}"
            self.logger.error(error_msg)
            errors.append(error_msg)
            
            return ProcessingResult(
                success=False,
                caption_records=caption_records,
                errors=errors,
                processing_time=time.time() - start_time
            )
    
    def search_captions(self, query: str, top_k: int = 5) -> List[dict]:
        """Search for similar captions using semantic search"""
        try:
            # Generate embedding for query
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            # Search for similar captions
            results = self.storage.search_similar_captions(query_embedding, top_k)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search captions: {e}")
            return []
    
    def get_video_captions(self, video_id: str) -> List[dict]:
        """Get all captions for a specific video"""
        return self.storage.get_captions_by_video(video_id)
    
    def get_statistics(self) -> dict:
        """Get service statistics"""
        stats = self.storage.get_statistics()
        
        # Add component information
        stats.update({
            'embedding_dimension': self.embedding_generator.get_embedding_dimension(),
            'vision_model': self.config.vision_model_name,
            'embedding_model': self.config.embedding_model_name,
            'async_enabled': self.config.enable_async_processing
        })
        
        return stats
    
    def get_rejected_captions(self) -> List[dict]:
        """Get audit log of rejected captions"""
        return self.caption_sanitizer.get_rejected_captions()
    
    def clear_rejected_captions(self):
        """Clear the rejected captions audit log"""
        self.caption_sanitizer.clear_rejected_captions()
    
    def close(self):
        """Close service and cleanup resources"""
        try:
            # Only close if storage exists and we're not in the middle of processing
            if hasattr(self, 'storage') and self.storage is not None:
                self.storage.close()
                self.logger.info("Captioning service closed")
        except Exception as e:
            self.logger.error(f"Failed to close service: {e}")
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            # Check if Python is shutting down
            import sys
            if sys.meta_path is not None and hasattr(self, 'storage'):
                self.close()
        except:
            # Silently ignore errors during shutdown
            pass