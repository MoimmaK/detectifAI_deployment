"""
Video Captioning Integrator for DetectifAI

This module integrates video captioning into the video processing pipeline.
It generates neutral, policy-safe captions from keyframes and stores them with semantic embeddings.
"""

import os
import sys
import logging
from typing import List, Dict, Any
from datetime import datetime
from PIL import Image
import cv2

# Import from video_captioning package
try:
    # Try direct import from package
    from video_captioning import CaptioningService, Frame, CaptioningConfig
except ImportError:
    # Fallback to explicit path
    from video_captioning.video_captioning.captioning_service import CaptioningService
    from video_captioning.video_captioning.models import Frame
    from video_captioning.video_captioning.config import CaptioningConfig

logger = logging.getLogger(__name__)


class VideoCaptioningIntegrator:
    """Integration layer between video captioning and DetectifAI pipeline"""
    
    def __init__(self, config, db_manager=None):
        self.config = config
        self.db_manager = db_manager
        self.enabled = getattr(config, 'enable_video_captioning', False)
        
        logger.info(f"🎬 Initializing VideoCaptioningIntegrator - enabled: {self.enabled}")
        
        # Initialize captioning service if enabled
        self.captioning_service = None
        
        if self.enabled:
            try:
                # Create captioning configuration from DetectifAI config
                captioning_config = CaptioningConfig(
                    vision_model_name=getattr(config, 'captioning_vision_model', "Salesforce/blip-image-captioning-base"),
                    vision_device=getattr(config, 'captioning_device', "cpu"),
                    vision_batch_size=getattr(config, 'captioning_batch_size', 4),
                    embedding_model_name=getattr(config, 'captioning_embedding_model', "sentence-transformers/all-MiniLM-L6-v2"),
                    db_connection_string=getattr(config, 'captioning_db_path', None),
                    vector_db_path=getattr(config, 'captioning_vector_db_path', "./video_captioning_store"),
                    enable_async_processing=getattr(config, 'captioning_async', True),
                    log_rejected_captions=True
                )
                
                # Initialize with MongoDB support
                self.captioning_service = CaptioningService(captioning_config, db_manager=db_manager)
                logger.info("✅ Video captioning service initialized successfully (MongoDB + FAISS)")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize video captioning service: {e}")
                self.enabled = False
        else:
            logger.info("Video captioning disabled in config")
    
    def _download_keyframe_from_minio(self, bucket, minio_path, local_path):
        """Download a keyframe from MinIO to local path"""
        try:
            if not self.db_manager or not self.db_manager.minio_client:
                logger.error("MinIO client not available")
                return False
            
            # Add timeout to prevent hanging
            import socket
            original_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(30)  # 30 second timeout
            
            try:
                self.db_manager.minio_client.fget_object(bucket, minio_path, local_path)
                logger.debug(f"✅ Downloaded {minio_path} to {local_path}")
                return True
            finally:
                socket.setdefaulttimeout(original_timeout)
                
        except Exception as e:
            logger.error(f"❌ Failed to download {minio_path} from MinIO: {e}")
            return False
    
    def process_keyframes_with_captioning(self, keyframes: List, video_id: str = None) -> Dict[str, Any]:
        """
        Process keyframes to generate captions
        
        Args:
            keyframes: List of KeyframeResult objects
            video_id: Optional video identifier
            
        Returns:
            Dictionary containing captioning results
        """
        if not self.enabled or not self.captioning_service:
            logger.info("🚫 Video captioning disabled, skipping...")
            return {
                'enabled': False,
                'total_captions': 0,
                'captions': []
            }
        
        logger.info(f"🎬 Starting video captioning on {len(keyframes)} keyframes")
        
        # Add overall timeout for the entire captioning process
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Video captioning exceeded maximum time limit")
        
        # Set 5 minute timeout for entire captioning process
        # Note: signal.alarm only works on Unix, so we'll use a different approach
        start_time = datetime.now()
        max_processing_time = 300  # 5 minutes in seconds
        
        # Create temporary directory for downloaded keyframes
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="keyframes_")
        logger.info(f"📁 Created temporary directory for keyframes: {temp_dir}")
        
        try:
            # Get keyframe bucket from db_manager
            keyframe_bucket = None
            if self.db_manager and hasattr(self.db_manager, 'keyframe_repo'):
                keyframe_bucket = self.db_manager.keyframe_repo.bucket
            elif self.db_manager:
                # Fallback: try to get from config or use default
                keyframe_bucket = getattr(self.db_manager, 'keyframe_bucket', 'detectifai-keyframes')
            else:
                keyframe_bucket = 'detectifai-keyframes'  # Default bucket name
            
            logger.info(f"🪣 Using MinIO bucket: {keyframe_bucket}")
            
            # Convert keyframes to Frame objects
            frames = []
            downloaded_files = []  # Track files for cleanup
            max_keyframes_to_process = 10  # Reduced limit for faster processing
            
            logger.info(f"📊 Processing up to {min(len(keyframes), max_keyframes_to_process)} keyframes (limited for performance)")
            
            for idx, keyframe in enumerate(keyframes[:max_keyframes_to_process]):  # Limit processing
                try:
                    # Debug: Log keyframe structure
                    logger.debug(f"Processing keyframe {idx}: type={type(keyframe)}")
                    
                    # Try different keyframe structures
                    frame_path = None
                    timestamp = None
                    frame_index = idx
                    minio_path = None
                    minio_bucket_override = None
                    
                    # Check for different attribute names
                    if hasattr(keyframe, 'frame_path'):
                        frame_path = keyframe.frame_path
                    elif hasattr(keyframe, 'path'):
                        frame_path = keyframe.path
                    elif hasattr(keyframe, 'frame_data') and hasattr(keyframe.frame_data, 'frame_path'):
                        frame_path = keyframe.frame_data.frame_path
                    
                    # Check for MinIO metadata in keyframe object (added by database_video_service)
                    if hasattr(keyframe, 'minio_path'):
                        minio_path = keyframe.minio_path
                        minio_bucket_override = getattr(keyframe, 'minio_bucket', None)
                    elif hasattr(keyframe, 'frame_data'):
                        if hasattr(keyframe.frame_data, 'minio_path'):
                            minio_path = keyframe.frame_data.minio_path
                            minio_bucket_override = getattr(keyframe.frame_data, 'minio_bucket', None)
                    
                    # Get timestamp
                    if hasattr(keyframe, 'timestamp'):
                        timestamp = keyframe.timestamp
                    elif hasattr(keyframe, 'frame_data') and hasattr(keyframe.frame_data, 'timestamp'):
                        timestamp = keyframe.frame_data.timestamp
                    else:
                        timestamp = 0.0
                    
                    # Get frame index
                    if hasattr(keyframe, 'frame_index'):
                        frame_index = keyframe.frame_index
                    elif hasattr(keyframe, 'frame_number'):
                        frame_index = keyframe.frame_number
                    elif hasattr(keyframe, 'frame_data') and hasattr(keyframe.frame_data, 'frame_number'):
                        frame_index = keyframe.frame_data.frame_number
                    
                    # Check if frame_path is a MinIO path (doesn't exist locally)
                    if frame_path and not os.path.exists(frame_path):
                        logger.debug(f"⚠️ Frame path doesn't exist locally: {frame_path}")
                        
                        # Use MinIO path from keyframe metadata if available
                        if not minio_path and video_id:
                            # Fallback: construct MinIO path from video_id and frame_index
                            minio_path = f"{video_id}/keyframes/frame_{frame_index:06d}.jpg"
                        
                        if minio_path:
                            logger.debug(f"🔍 Attempting to download from MinIO: {minio_path}")
                            
                            # Use bucket from keyframe metadata or default
                            bucket_to_use = minio_bucket_override or keyframe_bucket
                            
                            # Download from MinIO to temp directory
                            local_temp_path = os.path.join(temp_dir, f"frame_{frame_index:06d}.jpg")
                            
                            if self._download_keyframe_from_minio(bucket_to_use, minio_path, local_temp_path):
                                frame_path = local_temp_path
                                downloaded_files.append(local_temp_path)
                                logger.debug(f"✅ Downloaded keyframe to: {frame_path}")
                            else:
                                logger.warning(f"❌ Failed to download keyframe from MinIO: {minio_path}")
                                continue
                        else:
                            logger.warning(f"⚠️ No MinIO path available and no video_id to construct path")
                            continue
                    
                    # Load image from keyframe path
                    if frame_path and os.path.exists(frame_path):
                        logger.debug(f"📸 Loading image from: {frame_path}")
                        pil_image = Image.open(frame_path)
                        
                        # Create Frame object
                        frame = Frame(
                            frame_id=f"frame_{frame_index:06d}",
                            timestamp=datetime.fromtimestamp(timestamp) if timestamp else datetime.now(),
                            video_id=video_id or "unknown",
                            image=pil_image
                        )
                        
                        frames.append(frame)
                        logger.debug(f"✅ Successfully converted keyframe {idx}")
                    else:
                        logger.warning(f"⚠️ Keyframe {idx} has no valid frame_path or file doesn't exist: {frame_path}")
                        
                except Exception as e:
                    logger.error(f"❌ Error converting keyframe {idx}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
            
            if not frames:
                logger.warning("⚠️ No frames could be converted for captioning")
                logger.warning(f"Keyframe sample: {keyframes[0] if keyframes else 'No keyframes'}")
                return {
                    'enabled': True,
                    'total_captions': 0,
                    'captions': [],
                    'errors': ['No frames could be converted - check keyframe structure or MinIO access']
                }
            
            logger.info(f"📝 Processing {len(frames)} frames for captioning...")
            logger.info(f"⏱️  Time elapsed: {(datetime.now() - start_time).total_seconds():.1f}s")
            
            # Check if we've exceeded time limit before processing
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > max_processing_time:
                logger.error(f"❌ Exceeded time limit before caption generation: {elapsed:.1f}s")
                return {
                    'enabled': True,
                    'total_captions': 0,
                    'captions': [],
                    'errors': [f'Timeout: Exceeded {max_processing_time}s before caption generation']
                }
            
            # Process frames through captioning pipeline with error handling
            try:
                logger.info("🤖 Calling captioning service to process frames...")
                result = self.captioning_service.process_frames(frames)
                logger.info(f"✅ Captioning service completed in {(datetime.now() - start_time).total_seconds():.1f}s")
            except Exception as caption_error:
                logger.error(f"❌ Caption generation failed: {caption_error}")
                import traceback
                logger.error(traceback.format_exc())
                return {
                    'enabled': True,
                    'total_captions': 0,
                    'captions': [],
                    'errors': [f'Caption generation error: {str(caption_error)}']
                }
            
            # Extract caption records and print debugging info
            captions = []
            logger.info("=" * 80)
            logger.info("🎬 VIDEO CAPTIONING RESULTS - KEYFRAME CAPTIONS")
            logger.info("=" * 80)
            
            for idx, record in enumerate(result.caption_records, 1):
                caption_data = {
                    'caption_id': record.caption_id,
                    'frame_id': record.frame_id,
                    'timestamp': record.timestamp.isoformat(),
                    'raw_caption': record.raw_caption,
                    'sanitized_caption': record.sanitized_caption,
                    'created_at': record.created_at.isoformat()
                }
                captions.append(caption_data)
                
                # DEBUG: Print caption for each keyframe
                logger.info(f"\n📸 Keyframe #{idx} - {record.frame_id}")
                logger.info(f"   ⏱️  Timestamp: {record.timestamp}")
                logger.info(f"   🔤 Raw Caption: {record.raw_caption}")
                logger.info(f"   ✨ Sanitized Caption: {record.sanitized_caption}")
                logger.info(f"   🆔 Caption ID: {record.caption_id}")
                
                # Also print to console for immediate visibility
                print(f"\n{'='*60}")
                print(f"📸 Keyframe #{idx}: {record.frame_id}")
                print(f"⏱️  Time: {record.timestamp}")
                print(f"🔤 Caption: {record.sanitized_caption}")
                print(f"{'='*60}")
            
            logger.info("\n" + "=" * 80)
            logger.info(f"✅ Video captioning complete: {len(captions)} captions generated and saved to MongoDB")
            logger.info(f"💾 Embeddings saved to FAISS vector database")
            logger.info("=" * 80)
            
            return {
                'enabled': True,
                'total_captions': len(captions),
                'captions': captions,
                'processing_time': result.processing_time,
                'errors': result.errors
            }
            
        except Exception as e:
            logger.error(f"❌ Video captioning failed: {e}", exc_info=True)
            return {
                'enabled': True,
                'total_captions': 0,
                'captions': [],
                'errors': [str(e)]
            }
        finally:
            # Cleanup: Remove temporary directory and downloaded files
            try:
                import shutil
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    logger.info(f"🧹 Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup temporary directory: {e}")
    
    def search_captions(self, query: str, video_id: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search captions using semantic similarity
        
        Args:
            query: Search query text
            video_id: Optional video ID to filter results
            top_k: Number of results to return
            
        Returns:
            List of matching caption records with similarity scores
        """
        if not self.enabled or not self.captioning_service:
            return []
        
        try:
            results = self.captioning_service.search_captions(query, top_k=top_k)
            
            # Filter by video_id if provided
            if video_id:
                results = [r for r in results if r.get('video_id') == video_id]
            
            return results
            
        except Exception as e:
            logger.error(f"Caption search failed: {e}")
            return []
    
    def get_video_captions(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Get all captions for a specific video
        
        Args:
            video_id: Video identifier
            
        Returns:
            List of caption records
        """
        if not self.enabled or not self.captioning_service:
            return []
        
        try:
            return self.captioning_service.get_video_captions(video_id)
        except Exception as e:
            logger.error(f"Failed to get video captions: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get captioning service statistics"""
        if not self.enabled or not self.captioning_service:
            return {'enabled': False}
        
        try:
            stats = self.captioning_service.get_statistics()
            stats['enabled'] = True
            return stats
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {'enabled': True, 'error': str(e)}
