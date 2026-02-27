"""
Test script to verify caption generation fix
"""

import sys
sys.path.insert(0, '.')

from video_captioning.video_captioning.vision_captioner import VisionCaptioner
from video_captioning.video_captioning.config import CaptioningConfig
from PIL import Image
import numpy as np

def test_single_caption():
    """Test single image captioning"""
    print("=" * 50)
    print("TEST 1: Single Image Captioning")
    print("=" * 50)
    
    config = CaptioningConfig()
    captioner = VisionCaptioner(config)
    
    # Create test image
    test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    # Generate caption
    result = captioner.generate_caption(test_img)
    print(f"✅ Single caption: {result}")
    print()
    
    return result != "Unable to generate caption"

def test_batch_captions():
    """Test batch captioning"""
    print("=" * 50)
    print("TEST 2: Batch Image Captioning")
    print("=" * 50)
    
    config = CaptioningConfig()
    captioner = VisionCaptioner(config)
    
    # Create multiple test images
    test_imgs = [
        Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        for _ in range(5)
    ]
    
    # Generate batch captions
    results = captioner.generate_captions_batch(test_imgs)
    
    print(f"✅ Generated {len(results)} captions:")
    for i, caption in enumerate(results, 1):
        print(f"   {i}. {caption}")
    print()
    
    return all(c != "Unable to generate caption" for c in results)

def test_large_batch():
    """Test large batch (multiple sub-batches)"""
    print("=" * 50)
    print("TEST 3: Large Batch (Multiple Sub-batches)")
    print("=" * 50)
    
    config = CaptioningConfig()
    captioner = VisionCaptioner(config)
    
    # Create 12 images (will be split into 2 batches of 8 and 4)
    test_imgs = [
        Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        for _ in range(12)
    ]
    
    # Generate batch captions
    results = captioner.generate_captions_batch(test_imgs)
    
    print(f"✅ Generated {len(results)} captions from 12 images")
    print(f"   Batch size: {config.vision_batch_size}")
    print(f"   Expected batches: 2 (8 + 4)")
    print()
    
    return len(results) == 12 and all(c != "Unable to generate caption" for c in results)

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("CAPTION GENERATION FIX VERIFICATION")
    print("=" * 50)
    print()
    
    try:
        # Run tests
        test1_pass = test_single_caption()
        test2_pass = test_batch_captions()
        test3_pass = test_large_batch()
        
        # Summary
        print("=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        print(f"Single Caption:     {'✅ PASS' if test1_pass else '❌ FAIL'}")
        print(f"Batch Caption:      {'✅ PASS' if test2_pass else '❌ FAIL'}")
        print(f"Large Batch:        {'✅ PASS' if test3_pass else '❌ FAIL'}")
        print()
        
        if all([test1_pass, test2_pass, test3_pass]):
            print("🎉 ALL TESTS PASSED!")
            print("Caption generation is working correctly.")
        else:
            print("⚠️  SOME TESTS FAILED")
            print("Please check the logs above for errors.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
