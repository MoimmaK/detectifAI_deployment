"""
Example usage of the video captioning module
"""

import asyncio
from datetime import datetime
from PIL import Image
import logging

from video_captioning import CaptioningService, Frame, CaptioningConfig


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_sample_frames():
    """Create sample frames for testing"""
    # Create dummy images (in real usage, these would be actual video frames)
    frames = []
    
    for i in range(3):
        # Create a simple test image
        image = Image.new('RGB', (224, 224), color=(100 + i*50, 150, 200))
        
        frame = Frame(
            frame_id=f"frame_{i:04d}",
            timestamp=datetime.now(),
            video_id="test_video_001",
            image=image
        )
        frames.append(frame)
    
    return frames


def main():
    """Main example function"""
    setup_logging()
    
    # Create configuration
    config = CaptioningConfig(
        vision_model_name="Salesforce/blip-image-captioning-base",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        enable_async_processing=True,
        log_rejected_captions=True
    )
    
    # Initialize service
    service = CaptioningService(config)
    
    # Create sample frames
    frames = create_sample_frames()
    
    print(f"Processing {len(frames)} frames...")
    
    # Process frames synchronously
    result = service.process_frames(frames)
    
    print(f"Processing completed:")
    print(f"  Success: {result.success}")
    print(f"  Records created: {len(result.caption_records)}")
    print(f"  Processing time: {result.processing_time:.2f}s")
    print(f"  Errors: {len(result.errors)}")
    
    if result.errors:
        for error in result.errors:
            print(f"    - {error}")
    
    # Display results
    for record in result.caption_records:
        print(f"\nFrame {record.frame_id}:")
        print(f"  Raw caption: {record.raw_caption}")
        print(f"  Sanitized: {record.sanitized_caption}")
        print(f"  Embedding shape: {record.embedding.shape}")
    
    # Test search functionality
    print("\n--- Testing Search ---")
    search_results = service.search_captions("person walking", top_k=3)
    print(f"Found {len(search_results)} similar captions")
    
    # Get statistics
    stats = service.get_statistics()
    print(f"\n--- Statistics ---")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    service.close()


async def async_example():
    """Example of async processing"""
    setup_logging()
    
    config = CaptioningConfig(enable_async_processing=True)
    service = CaptioningService(config)
    
    frames = create_sample_frames()
    
    print("Processing frames asynchronously...")
    result = await service.process_frames_async(frames)
    
    print(f"Async processing completed in {result.processing_time:.2f}s")
    
    service.close()


if __name__ == "__main__":
    # Run synchronous example
    main()
    
    # Run async example
    print("\n" + "="*50)
    print("Running async example...")
    asyncio.run(async_example())