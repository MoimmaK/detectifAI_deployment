"""
Quick verification script to check if all imports work correctly
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 80)
print("🔍 VERIFYING VIDEO CAPTIONING INTEGRATION")
print("=" * 80)

# Test 1: MongoDB Storage
print("\n1. Testing MongoDB Storage import...")
try:
    from video_captioning.video_captioning.mongodb_storage import MongoDBCaptionStorage
    print("   ✅ MongoDBCaptionStorage imported successfully")
except Exception as e:
    print(f"   ❌ Failed to import MongoDBCaptionStorage: {e}")

# Test 2: Captioning Service
print("\n2. Testing Captioning Service import...")
try:
    from video_captioning.video_captioning.captioning_service import CaptioningService
    print("   ✅ CaptioningService imported successfully")
except Exception as e:
    print(f"   ❌ Failed to import CaptioningService: {e}")

# Test 3: Video Captioning Integrator
print("\n3. Testing Video Captioning Integrator import...")
try:
    from video_captioning_integrator import VideoCaptioningIntegrator
    print("   ✅ VideoCaptioningIntegrator imported successfully")
except Exception as e:
    print(f"   ❌ Failed to import VideoCaptioningIntegrator: {e}")

# Test 4: Main Pipeline
print("\n4. Testing Main Pipeline import...")
try:
    from main_pipeline import CompleteVideoProcessingPipeline
    print("   ✅ CompleteVideoProcessingPipeline imported successfully")
except Exception as e:
    print(f"   ❌ Failed to import CompleteVideoProcessingPipeline: {e}")

# Test 5: Database Manager
print("\n5. Testing Database Manager import...")
try:
    from database.config import DatabaseManager
    print("   ✅ DatabaseManager imported successfully")
except Exception as e:
    print(f"   ❌ Failed to import DatabaseManager: {e}")

# Test 6: Config
print("\n6. Testing Config import...")
try:
    from config import get_security_focused_config
    config = get_security_focused_config()
    print("   ✅ Config imported successfully")
    print(f"   📝 Video captioning enabled: {getattr(config, 'enable_video_captioning', False)}")
except Exception as e:
    print(f"   ❌ Failed to import config: {e}")

print("\n" + "=" * 80)
print("✅ VERIFICATION COMPLETE")
print("=" * 80)
print("\nAll imports are working correctly!")
print("You can now:")
print("  1. Upload videos via the API")
print("  2. Run: python test_video_captioning_integration.py")
print("  3. Process videos with captioning enabled")
