"""
Quick test without heavy model downloads - uses mock data
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from PIL import Image
import cv2

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models import Frame


def create_test_frames_from_video(video_path, num_frames=3):
    """Create test frames from video without processing"""
    frames = []
    
    try:
        print(f"Reading video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open {video_path}")
            return frames
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video: {total_frames} frames, {fps:.1f} FPS")
        
        # Extract frames at intervals
        interval = max(1, total_frames // num_frames)
        
        for i in range(num_frames):
            frame_pos = i * interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            
            ret, cv_frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB and PIL
            rgb_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            pil_image = pil_image.resize((320, 240))  # Smaller for testing
            
            frame = Frame(
                frame_id=f"test_frame_{i:03d}",
                timestamp=datetime.now(),
                video_id=Path(video_path).stem,
                image=pil_image
            )
            
            frames.append(frame)
            print(f"✓ Extracted frame {i+1}/{num_frames}")
        
        cap.release()
        
    except Exception as e:
        print(f"Error: {e}")
    
    return frames


def test_basic_functionality():
    """Test basic module functionality without heavy models"""
    print("="*50)
    print("QUICK TEST - Video Captioning Module")
    print("="*50)
    
    # Find a test video
    test_videos = [
        "../backend/fight_0002.mp4",
        "../backend/fire.mp4",
        "../backend/rob.mp4"
    ]
    
    video_path = None
    for video in test_videos:
        if os.path.exists(video):
            video_path = video
            break
    
    if not video_path:
        print("No test videos found!")
        print("Available videos should be in ../backend/")
        return
    
    print(f"Using video: {video_path}")
    
    # Test 1: Frame extraction
    print("\n1. Testing frame extraction...")
    frames = create_test_frames_from_video(video_path, num_frames=2)
    
    if frames:
        print(f"✓ Successfully extracted {len(frames)} frames")
        for frame in frames:
            print(f"  - {frame.frame_id}: {frame.image.size} pixels")
    else:
        print("✗ Failed to extract frames")
        return
    
    # Test 2: Basic model imports
    print("\n2. Testing module imports...")
    try:
        from config import CaptioningConfig
        config = CaptioningConfig()
        print("✓ Configuration loaded")
        
        from models import CaptionRecord
        print("✓ Models imported")
        
        from storage import CaptionStorage
        print("✓ Storage module imported")
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        return
    
    # Test 3: Mock caption processing
    print("\n3. Testing mock caption processing...")
    try:
        import uuid
        import numpy as np
        
        # Create mock caption records
        mock_records = []
        for frame in frames:
            record = CaptionRecord(
                caption_id=str(uuid.uuid4()),
                video_id=frame.video_id,
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                raw_caption=f"Mock raw caption for {frame.frame_id}",
                sanitized_caption=f"Person performing activity in scene {frame.frame_id[-1]}",
                embedding=np.random.rand(384),  # Mock embedding
                created_at=datetime.now()
            )
            mock_records.append(record)
        
        print(f"✓ Created {len(mock_records)} mock caption records")
        
        for record in mock_records:
            print(f"  - {record.frame_id}: {record.sanitized_caption}")
        
    except Exception as e:
        print(f"✗ Mock processing error: {e}")
        return
    
    # Test 4: Storage test
    print("\n4. Testing storage...")
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config = CaptioningConfig(
                db_connection_string=os.path.join(temp_dir, "test.db"),
                vector_db_path=os.path.join(temp_dir, "vectors")
            )
            
            storage = CaptionStorage(test_config)
            
            # Store mock records
            stored = storage.store_caption_records_batch(mock_records)
            print(f"✓ Stored {stored} records in database")
            
            # Test retrieval
            video_captions = storage.get_captions_by_video(frames[0].video_id)
            print(f"✓ Retrieved {len(video_captions)} captions for video")
            
            storage.close()
        
    except Exception as e:
        print(f"✗ Storage test error: {e}")
        return
    
    print("\n" + "="*50)
    print("✅ QUICK TEST PASSED!")
    print("="*50)
    print("\nNext steps:")
    print("1. Install full requirements: python install_requirements.py")
    print("2. Run full test: python test_runner.py")
    print("3. Or run example: python example_usage.py")


if __name__ == "__main__":
    test_basic_functionality()