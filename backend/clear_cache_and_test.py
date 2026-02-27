"""Clear Python cache and test imports"""
import os
import shutil
import sys

print("🧹 Clearing Python cache...")

# Clear __pycache__ directories
for root, dirs, files in os.walk('.'):
    if '__pycache__' in dirs:
        pycache_path = os.path.join(root, '__pycache__')
        try:
            shutil.rmtree(pycache_path)
            print(f"   Removed: {pycache_path}")
        except Exception as e:
            print(f"   Failed to remove {pycache_path}: {e}")

# Clear .pyc files
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.pyc'):
            pyc_path = os.path.join(root, file)
            try:
                os.remove(pyc_path)
                print(f"   Removed: {pyc_path}")
            except Exception as e:
                print(f"   Failed to remove {pyc_path}: {e}")

print("\n✅ Cache cleared!")

print("\n" + "="*80)
print("🧪 Testing imports...")
print("="*80)

try:
    print("\n1. Testing video_captioning package...")
    from video_captioning import CaptioningService, Frame, CaptioningConfig
    print("   ✅ SUCCESS: video_captioning package imported")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n2. Testing video_captioning_integrator...")
    from video_captioning_integrator import VideoCaptioningIntegrator
    print("   ✅ SUCCESS: VideoCaptioningIntegrator imported")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n3. Testing main_pipeline...")
    from main_pipeline import CompleteVideoProcessingPipeline
    print("   ✅ SUCCESS: CompleteVideoProcessingPipeline imported")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("✅ All tests complete!")
print("="*80)
