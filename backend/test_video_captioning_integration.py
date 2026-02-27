"""
Test script for video captioning integration with MongoDB and FAISS
"""

import os
import sys
import logging
from datetime import datetime
from PIL import Image

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_video_captioning_integration():
    """Test video captioning with MongoDB and FAISS integration"""
    
    print("=" * 80)
    print("🎬 VIDEO CAPTIONING INTEGRATION TEST")
    print("=" * 80)
    
    try:
        # Import required modules
        from config import get_security_focused_config
        from video_captioning_integrator import VideoCaptioningIntegrator
        from database.config import DatabaseManager
        
        # Initialize database manager
        print("\n📦 Initializing database manager...")
        db_manager = DatabaseManager()
        
        # Test database connections
        print("🔌 Testing database connections...")
        if db_manager.test_connections():
            print("✅ Database connections successful")
        else:
            print("❌ Database connection failed")
            return
        
        # Initialize config with video captioning enabled
        print("\n⚙️  Initializing configuration...")
        config = get_security_focused_config()
        config.enable_video_captioning = True
        config.captioning_device = "cpu"  # Use CPU for testing
        config.captioning_batch_size = 2
        
        # Initialize video captioning integrator
        print("\n🎬 Initializing video captioning integrator...")
        captioning_integrator = VideoCaptioningIntegrator(config, db_manager=db_manager)
        
        if not captioning_integrator.enabled:
            print("❌ Video captioning not enabled")
            return
        
        print("✅ Video captioning integrator initialized")
        
        # Create test frames
        print("\n📸 Creating test frames...")
        from video_captioning.video_captioning.models import Frame
        
        # Create dummy images for testing
        test_frames = []
        for i in range(3):
            # Create a simple test image
            img = Image.new('RGB', (640, 480), color=(i*50, 100, 150))
            
            frame = Frame(
                frame_id=f"test_frame_{i:03d}",
                timestamp=datetime.now(),
                video_id="test_video_001",
                image=img
            )
            test_frames.append(frame)
        
        print(f"✅ Created {len(test_frames)} test frames")
        
        # Process frames with captioning
        print("\n🎬 Processing frames with video captioning...")
        print("=" * 80)
        
        result = captioning_integrator.process_keyframes_with_captioning(
            test_frames,
            video_id="test_video_001"
        )
        
        print("\n" + "=" * 80)
        print("📊 CAPTIONING RESULTS")
        print("=" * 80)
        print(f"Enabled: {result.get('enabled')}")
        print(f"Total Captions: {result.get('total_captions')}")
        print(f"Processing Time: {result.get('processing_time', 0):.2f}s")
        print(f"Errors: {len(result.get('errors', []))}")
        
        if result.get('captions'):
            print("\n📝 Generated Captions:")
            for idx, caption in enumerate(result['captions'], 1):
                print(f"\n  Caption {idx}:")
                print(f"    Frame ID: {caption.get('frame_id')}")
                print(f"    Raw: {caption.get('raw_caption')}")
                print(f"    Sanitized: {caption.get('sanitized_caption')}")
        
        # Test caption search
        print("\n" + "=" * 80)
        print("🔍 TESTING CAPTION SEARCH")
        print("=" * 80)
        
        search_results = captioning_integrator.search_captions(
            query="person walking",
            video_id="test_video_001",
            top_k=3
        )
        
        print(f"Found {len(search_results)} matching captions")
        for idx, result in enumerate(search_results, 1):
            print(f"\n  Result {idx}:")
            print(f"    Caption: {result.get('sanitized_caption')}")
            print(f"    Similarity: {result.get('similarity', 0):.4f}")
        
        # Get statistics
        print("\n" + "=" * 80)
        print("📊 CAPTIONING STATISTICS")
        print("=" * 80)
        
        stats = captioning_integrator.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n" + "=" * 80)
        print("✅ VIDEO CAPTIONING INTEGRATION TEST COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_video_captioning_integration()
