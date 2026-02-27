"""
Debug script to test video captioning initialization and processing
"""

import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

print("=" * 80)
print("🔍 DEBUGGING VIDEO CAPTIONING")
print("=" * 80)

try:
    print("\n1. Testing imports...")
    from config import get_security_focused_config
    from main_pipeline import CompleteVideoProcessingPipeline
    from database.config import DatabaseManager
    print("✅ Imports successful")
    
    print("\n2. Initializing database manager...")
    db_manager = DatabaseManager()
    print("✅ Database manager initialized")
    
    print("\n3. Creating config with captioning enabled...")
    config = get_security_focused_config()
    config.enable_video_captioning = True
    config.captioning_device = "cpu"
    config.captioning_batch_size = 1
    print(f"✅ Config created - captioning enabled: {config.enable_video_captioning}")
    
    print("\n4. Initializing pipeline...")
    pipeline = CompleteVideoProcessingPipeline(config, db_manager=db_manager)
    print(f"✅ Pipeline initialized")
    
    print("\n5. Checking video captioning component...")
    if pipeline.video_captioning:
        print(f"✅ Video captioning component exists")
        print(f"   Enabled: {pipeline.video_captioning.enabled}")
        print(f"   Service: {pipeline.video_captioning.captioning_service}")
        
        if pipeline.video_captioning.captioning_service:
            print(f"   ✅ Captioning service is initialized")
            
            # Test with a dummy frame
            print("\n6. Testing with dummy frame...")
            from PIL import Image
            from datetime import datetime
            from video_captioning.video_captioning.models import Frame
            
            # Create a test image
            test_image = Image.new('RGB', (640, 480), color=(100, 150, 200))
            
            # Create a test frame
            test_frame = Frame(
                frame_id="test_frame_001",
                timestamp=datetime.now(),
                video_id="test_video",
                image=test_image
            )
            
            print("   Processing test frame...")
            result = pipeline.video_captioning.captioning_service.process_frames([test_frame])
            
            print(f"\n   ✅ Processing complete!")
            print(f"   Success: {result.success}")
            print(f"   Captions generated: {len(result.caption_records)}")
            print(f"   Processing time: {result.processing_time:.2f}s")
            print(f"   Errors: {result.errors}")
            
            if result.caption_records:
                for idx, record in enumerate(result.caption_records, 1):
                    print(f"\n   Caption {idx}:")
                    print(f"      Raw: {record.raw_caption}")
                    print(f"      Sanitized: {record.sanitized_caption}")
        else:
            print(f"   ❌ Captioning service is None")
    else:
        print(f"❌ Video captioning component is None")
        print(f"   Config enable_video_captioning: {config.enable_video_captioning}")
    
    print("\n" + "=" * 80)
    print("✅ DEBUG COMPLETE")
    print("=" * 80)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
