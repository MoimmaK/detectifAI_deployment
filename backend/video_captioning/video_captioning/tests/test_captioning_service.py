"""
Tests for the main captioning service
"""

import pytest
import asyncio
from datetime import datetime
from PIL import Image
import tempfile
import os

from video_captioning import CaptioningService, Frame, CaptioningConfig


@pytest.fixture
def temp_config():
    """Create a temporary configuration for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = CaptioningConfig(
            db_connection_string=os.path.join(temp_dir, "test.db"),
            vector_db_path=os.path.join(temp_dir, "vector_store"),
            vision_batch_size=2,
            enable_async_processing=True,
            log_rejected_captions=True
        )
        yield config


@pytest.fixture
def sample_frames():
    """Create sample frames for testing"""
    frames = []
    for i in range(3):
        image = Image.new('RGB', (224, 224), color=(100 + i*50, 150, 200))
        frame = Frame(
            frame_id=f"test_frame_{i:04d}",
            timestamp=datetime.now(),
            video_id="test_video_001",
            image=image
        )
        frames.append(frame)
    return frames


class TestCaptioningService:
    """Test cases for CaptioningService"""
    
    def test_service_initialization(self, temp_config):
        """Test service initializes correctly"""
        service = CaptioningService(temp_config)
        assert service.config == temp_config
        assert service.vision_captioner is not None
        assert service.caption_sanitizer is not None
        assert service.embedding_generator is not None
        assert service.storage is not None
        service.close()
    
    def test_process_frames_sync(self, temp_config, sample_frames):
        """Test synchronous frame processing"""
        service = CaptioningService(temp_config)
        
        result = service.process_frames(sample_frames)
        
        assert result.success
        assert len(result.caption_records) == len(sample_frames)
        assert result.processing_time > 0
        assert len(result.errors) == 0
        
        # Check record structure
        for record in result.caption_records:
            assert record.caption_id is not None
            assert record.video_id == "test_video_001"
            assert record.raw_caption is not None
            assert record.sanitized_caption is not None
            assert record.embedding is not None
            assert record.embedding.shape[0] > 0  # Has embedding dimension
        
        service.close()
    
    @pytest.mark.asyncio
    async def test_process_frames_async(self, temp_config, sample_frames):
        """Test asynchronous frame processing"""
        service = CaptioningService(temp_config)
        
        result = await service.process_frames_async(sample_frames)
        
        assert result.success
        assert len(result.caption_records) == len(sample_frames)
        assert result.processing_time > 0
        
        service.close()
    
    def test_search_captions(self, temp_config, sample_frames):
        """Test caption search functionality"""
        service = CaptioningService(temp_config)
        
        # First process some frames
        result = service.process_frames(sample_frames)
        assert result.success
        
        # Then search
        search_results = service.search_captions("test query", top_k=2)
        
        # Should return results (even if similarity is low)
        assert isinstance(search_results, list)
        assert len(search_results) <= 2
        
        service.close()
    
    def test_get_video_captions(self, temp_config, sample_frames):
        """Test retrieving captions by video ID"""
        service = CaptioningService(temp_config)
        
        # Process frames
        result = service.process_frames(sample_frames)
        assert result.success
        
        # Get captions for video
        video_captions = service.get_video_captions("test_video_001")
        
        assert len(video_captions) == len(sample_frames)
        for caption in video_captions:
            assert caption['video_id'] == "test_video_001"
            assert 'sanitized_caption' in caption
            assert 'timestamp' in caption
        
        service.close()
    
    def test_get_statistics(self, temp_config, sample_frames):
        """Test statistics retrieval"""
        service = CaptioningService(temp_config)
        
        # Process frames
        result = service.process_frames(sample_frames)
        assert result.success
        
        # Get statistics
        stats = service.get_statistics()
        
        assert 'total_captions' in stats
        assert 'unique_videos' in stats
        assert 'embedding_dimension' in stats
        assert 'vision_model' in stats
        assert stats['total_captions'] == len(sample_frames)
        assert stats['unique_videos'] == 1
        
        service.close()
    
    def test_empty_frames_list(self, temp_config):
        """Test processing empty frames list"""
        service = CaptioningService(temp_config)
        
        result = service.process_frames([])
        
        assert result.success
        assert len(result.caption_records) == 0
        assert len(result.errors) == 0
        
        service.close()
    
    def test_invalid_frame_handling(self, temp_config):
        """Test handling of invalid frames"""
        service = CaptioningService(temp_config)
        
        # Create frame with invalid image
        try:
            invalid_frame = Frame(
                frame_id="invalid",
                timestamp=datetime.now(),
                video_id="test",
                image="not_an_image"  # This should cause validation error
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        service.close()
    
    def test_service_cleanup(self, temp_config):
        """Test service cleanup"""
        service = CaptioningService(temp_config)
        
        # Service should close without errors
        service.close()
        
        # Multiple closes should be safe
        service.close()


if __name__ == "__main__":
    pytest.main([__file__])