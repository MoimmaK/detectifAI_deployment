"""
Simple test without OpenCV dependency
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test basic module imports"""
    print("Testing module imports...")
    
    try:
        from models import Frame, CaptionRecord
        print("✓ Models imported successfully")
        
        from config import CaptioningConfig
        config = CaptioningConfig()
        print("✓ Configuration loaded")
        
        from storage import CaptionStorage
        print("✓ Storage module imported")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_with_dummy_data():
    """Test with dummy image data"""
    print("\nTesting with dummy data...")
    
    try:
        from PIL import Image
        from models import Frame, CaptionRecord
        from config import CaptioningConfig
        import uuid
        import tempfile
        
        # Create dummy image
        dummy_image = Image.new('RGB', (320, 240), color=(100, 150, 200))
        
        # Create frame
        frame = Frame(
            frame_id="test_frame_001",
            timestamp=datetime.now(),
            video_id="test_video",
            image=dummy_image
        )
        
        print(f"✓ Created test frame: {frame.frame_id}")
        print(f"  Image size: {frame.image.size}")
        print(f"  Video ID: {frame.video_id}")
        
        # Test storage with dummy data
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CaptioningConfig(
                db_connection_string=os.path.join(temp_dir, "test.db"),
                vector_db_path=os.path.join(temp_dir, "vectors")
            )
            
            from storage import CaptionStorage
            storage = CaptionStorage(config)
            
            # Create dummy caption record
            import numpy as np
            record = CaptionRecord(
                caption_id=str(uuid.uuid4()),
                video_id=frame.video_id,
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                raw_caption="A test scene with colors",
                sanitized_caption="Scene with various objects",
                embedding=np.random.rand(384),
                created_at=datetime.now()
            )
            
            # Store record
            success = storage.store_caption_record(record)
            print(f"✓ Stored caption record: {success}")
            
            # Retrieve record
            retrieved = storage.get_caption_by_id(record.caption_id)
            print(f"✓ Retrieved record: {retrieved is not None}")
            
            # Get statistics
            stats = storage.get_statistics()
            print(f"✓ Statistics: {stats}")
            
            storage.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_video_files():
    """Check what video files are available"""
    print("\nChecking for video files...")
    
    video_paths = [
        "../backend/fight_0002.mp4",
        "../backend/fire.mp4",
        "../backend/rob.mp4",
        "../backend/fire+weapon.mp4"
    ]
    
    found_videos = []
    for video_path in video_paths:
        if os.path.exists(video_path):
            size = os.path.getsize(video_path)
            print(f"✓ Found: {video_path} ({size/1024/1024:.1f} MB)")
            found_videos.append(video_path)
        else:
            print(f"✗ Not found: {video_path}")
    
    return found_videos


def main():
    """Run simple tests"""
    print("="*50)
    print("SIMPLE TEST - Video Captioning Module")
    print("="*50)
    
    # Test 1: Basic imports
    if not test_imports():
        print("Basic imports failed. Check your Python environment.")
        return
    
    # Test 2: Dummy data processing
    if not test_with_dummy_data():
        print("Dummy data test failed.")
        return
    
    # Test 3: Check video files
    videos = check_video_files()
    
    print("\n" + "="*50)
    print("✅ SIMPLE TEST COMPLETED!")
    print("="*50)
    
    if videos:
        print(f"\nFound {len(videos)} video files ready for testing.")
        print("\nNext steps:")
        print("1. Fix NumPy compatibility: pip install 'numpy<2'")
        print("2. Install OpenCV: pip install opencv-python")
        print("3. Run full test: python test_runner.py")
    else:
        print("\nNo video files found in ../backend/")
        print("Make sure video files are in the backend directory.")


if __name__ == "__main__":
    main()