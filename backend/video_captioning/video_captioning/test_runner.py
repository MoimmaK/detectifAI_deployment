"""
Simple test runner for video captioning module
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from PIL import Image
import cv2
import logging

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from captioning_service import CaptioningService
from models import Frame
from config import CaptioningConfig


def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def extract_sample_frames(video_path, max_frames=5):
    """Extract a few sample frames from video for testing"""
    frames = []
    
    try:
        print(f"Opening video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return frames
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")
        
        # Extract frames at regular intervals
        frame_interval = max(1, total_frames // max_frames)
        
        frame_count = 0
        extracted = 0
        
        while extracted < max_frames:
            ret, cv_frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                # Resize for efficiency (optional)
                pil_image = pil_image.resize((640, 480), Image.Resampling.LANCZOS)
                
                # Calculate timestamp
                timestamp_seconds = frame_count / fps if fps > 0 else extracted
                timestamp = datetime.now().replace(
                    second=int(timestamp_seconds) % 60,
                    microsecond=int((timestamp_seconds % 1) * 1000000)
                )
                
                frame = Frame(
                    frame_id=f"frame_{frame_count:06d}",
                    timestamp=timestamp,
                    video_id=Path(video_path).stem,
                    image=pil_image
                )
                
                frames.append(frame)
                extracted += 1
                print(f"Extracted frame {extracted}/{max_frames} at {timestamp_seconds:.2f}s")
            
            frame_count += 1
        
        cap.release()
        print(f"Successfully extracted {len(frames)} frames")
        
    except Exception as e:
        print(f"Error extracting frames: {e}")
    
    return frames


def test_video_captioning(video_path):
    """Test the captioning module with a video file"""
    print(f"\n{'='*60}")
    print(f"TESTING VIDEO CAPTIONING MODULE")
    print(f"Video: {video_path}")
    print(f"{'='*60}")
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    try:
        # Create configuration
        config = CaptioningConfig(
            vision_model_name="Salesforce/blip-image-captioning-base",
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            vision_device="cpu",  # Use CPU for compatibility
            embedding_device="cpu",
            vision_batch_size=2,  # Small batch for testing
            enable_async_processing=False,  # Sync for simplicity
            log_rejected_captions=True
        )
        
        print("Initializing captioning service...")
        service = CaptioningService(config)
        print("✓ Service initialized successfully")
        
        # Extract frames
        print("\nExtracting frames from video...")
        frames = extract_sample_frames(video_path, max_frames=3)
        
        if not frames:
            print("No frames extracted. Exiting.")
            return
        
        # Process frames
        print(f"\nProcessing {len(frames)} frames...")
        result = service.process_frames(frames)
        
        # Display results
        print(f"\n{'='*40}")
        print("PROCESSING RESULTS")
        print(f"{'='*40}")
        print(f"Success: {result.success}")
        print(f"Processing time: {result.processing_time:.2f} seconds")
        print(f"Records created: {len(result.caption_records)}")
        print(f"Errors: {len(result.errors)}")
        
        if result.errors:
            print("\nErrors encountered:")
            for error in result.errors:
                print(f"  - {error}")
        
        # Show captions
        if result.caption_records:
            print(f"\n{'='*40}")
            print("GENERATED CAPTIONS")
            print(f"{'='*40}")
            
            for i, record in enumerate(result.caption_records, 1):
                print(f"\nFrame {i} ({record.frame_id}):")
                print(f"  Timestamp: {record.timestamp}")
                print(f"  Raw caption: {record.raw_caption}")
                print(f"  Safe caption: {record.sanitized_caption}")
                print(f"  Embedding shape: {record.embedding.shape}")
        
        # Test search functionality
        print(f"\n{'='*40}")
        print("TESTING SEARCH")
        print(f"{'='*40}")
        
        search_queries = ["person", "movement", "activity", "scene"]
        
        for query in search_queries:
            print(f"\nSearching for: '{query}'")
            results = service.search_captions(query, top_k=3)
            
            if results:
                for j, result_item in enumerate(results, 1):
                    similarity = result_item.get('similarity', 0)
                    print(f"  {j}. {result_item['sanitized_caption']} (similarity: {similarity:.3f})")
            else:
                print("  No results found")
        
        # Show statistics
        stats = service.get_statistics()
        print(f"\n{'='*40}")
        print("STATISTICS")
        print(f"{'='*40}")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Check for rejected captions
        rejected = service.get_rejected_captions()
        if rejected:
            print(f"\nRejected captions: {len(rejected)}")
            for rejection in rejected:
                print(f"  Raw: {rejection['raw']}")
                print(f"  Reason: {rejection['reason']}")
        
        print(f"\n{'='*60}")
        print("TEST COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        
        service.close()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run tests"""
    setup_logging()
    
    # List of available test videos
    test_videos = [
        "../backend/fight_0002.mp4",
        "../backend/fire.mp4", 
        "../backend/rob.mp4",
        "../backend/fire+weapon.mp4"
    ]
    
    print("Available test videos:")
    available_videos = []
    for i, video in enumerate(test_videos, 1):
        if os.path.exists(video):
            available_videos.append(video)
            print(f"  {i}. {video}")
        else:
            print(f"  {i}. {video} (NOT FOUND)")
    
    if not available_videos:
        print("No test videos found!")
        return
    
    # Test the first available video
    test_video = available_videos[0]
    print(f"\nTesting with: {test_video}")
    
    test_video_captioning(test_video)


if __name__ == "__main__":
    main()