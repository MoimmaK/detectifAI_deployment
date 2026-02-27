"""Quick import test"""
import sys
import os

print("Testing imports...")

try:
    print("\n1. Testing video_captioning_integrator import...")
    from video_captioning_integrator import VideoCaptioningIntegrator
    print("   ✅ SUCCESS: VideoCaptioningIntegrator imported")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n2. Testing main_pipeline import...")
    from main_pipeline import CompleteVideoProcessingPipeline
    print("   ✅ SUCCESS: CompleteVideoProcessingPipeline imported")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Import test complete!")
