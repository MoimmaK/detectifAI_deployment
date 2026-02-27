"""
Simple test for video captioning without requiring MongoDB videos
Tests the captioning service directly with a local image
"""

import sys
import os
import logging
from PIL import Image
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

print("=" * 80)
print("🎬 SIMPLE VIDEO CAPTIONING TEST")
print("=" * 80)

try:
    from config import get_security_focused_config
    from database.config import DatabaseManager
    from video_captioning_integrator import VideoCaptioningIntegrator
    
    # Initialize config with captioning enabled
    print("\n📦 Initializing configuration...")
    config = get_security_focused_config()
    config.enable_video_captioning = True
    
    # Initialize database manager
    print("🔌 Connecting to database...")
    db_manager = DatabaseManager()
    
    # Initialize video captioning integrator
    print("🎬 Initializing video captioning integrator...")
    captioning_integrator = VideoCaptioningIntegrator(config, db_manager=db_manager)
    
    if not captioning_integrator.enabled:
        print("❌ Video captioning is not enabled!")
        sys.exit(1)
    
    print("✅ Video captioning integrator initialized")
    
    # Find a test image (look for common image files)
    print("\n🔍 Looking for test images...")
    test_images = []
    
    # Check common locations
    image_locations = [
        "download.jpeg",
        "images.jpeg",
        "test_image.jpg",
        "test_image.png"
    ]
    
    for img_path in image_locations:
        if os.path.exists(img_path):
            test_images.append(img_path)
            print(f"   ✅ Found: {img_path}")
    
    if not test_images:
        print("\n❌ No test images found")
        print("💡 Please place a test image (jpg/png) in the backend directory")
        print("   Or specify a path to an image file")
        sys.exit(1)
    
    # Use the first available image
    test_image_path = test_images[0]
    print(f"\n✅ Using test image: {test_image_path}")
    
    # Create a mock keyframe object
    print("\n🔨 Creating mock keyframe object...")
    
    class MockFrameData:
        def __init__(self, frame_number, timestamp, frame_path):
            self.frame_number = frame_number
            self.timestamp = timestamp
            self.frame_path = frame_path
    
    class MockKeyframe:
        def __init__(self, frame_data):
            self.frame_data = frame_data
    
    # Create mock keyframe
    frame_data = MockFrameData(
        frame_number=0,
        timestamp=0.0,
        frame_path=test_image_path
    )
    mock_keyframe = MockKeyframe(frame_data)
    
    print("✅ Created mock keyframe")
    
    # Test captioning
    print("\n🎬 Testing video captioning...")
    print("=" * 80)
    
    result = captioning_integrator.process_keyframes_with_captioning(
        [mock_keyframe],
        video_id="test_video_simple"
    )
    
    print("\n" + "=" * 80)
    print("📊 CAPTIONING RESULTS")
    print("=" * 80)
    print(f"Enabled: {result.get('enabled')}")
    print(f"Total captions: {result.get('total_captions')}")
    print(f"Processing time: {result.get('processing_time', 'N/A')}")
    print(f"Errors: {result.get('errors', [])}")
    
    if result.get('captions'):
        print(f"\n✅ Successfully generated {len(result['captions'])} caption(s)!")
        print("\nCaption details:")
        for i, caption in enumerate(result['captions'], 1):
            print(f"\n{i}. Frame: {caption['frame_id']}")
            print(f"   Raw Caption: {caption['raw_caption']}")
            print(f"   Sanitized Caption: {caption['sanitized_caption']}")
            print(f"   Timestamp: {caption['timestamp']}")
            print(f"   Caption ID: {caption['caption_id']}")
    else:
        print("\n⚠️ No captions were generated")
        if result.get('errors'):
            print("Errors encountered:")
            for error in result['errors']:
                print(f"  - {error}")
    
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)
    print("\n💡 This test verifies that the captioning service works correctly.")
    print("   For MinIO integration testing, use: python test_minio_captioning.py")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
