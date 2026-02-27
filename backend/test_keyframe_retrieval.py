"""
Test if keyframes can be retrieved from MinIO for video captioning
"""

import sys
import os
import tempfile
from database.config import DatabaseManager

print("=" * 80)
print("🔍 TESTING KEYFRAME RETRIEVAL FOR VIDEO CAPTIONING")
print("=" * 80)

try:
    # Connect to database
    print("\n🔌 Connecting to database...")
    db_manager = DatabaseManager()
    
    # Get the video
    video_id = "video_20260212_143538_fc066b61"
    print(f"\n📹 Looking for video: {video_id}")
    
    video_file_collection = db_manager.db.video_file
    video = video_file_collection.find_one({"video_id": video_id})
    
    if not video:
        print(f"❌ Video not found: {video_id}")
        sys.exit(1)
    
    print(f"✅ Found video: {video_id}")
    
    # Get keyframe metadata
    print("\n📊 Checking keyframe metadata...")
    
    if 'meta_data' not in video or 'keyframe_info' not in video['meta_data']:
        print("❌ No keyframe_info in video metadata")
        sys.exit(1)
    
    keyframe_info = video['meta_data']['keyframe_info']
    print(f"✅ Found {len(keyframe_info)} keyframes in metadata")
    
    # Show first keyframe
    if keyframe_info:
        print("\n📸 First keyframe metadata:")
        first_kf = keyframe_info[0]
        for key, value in first_kf.items():
            print(f"   {key}: {value}")
    
    # Test downloading keyframes from MinIO
    print("\n🔍 Testing keyframe download from MinIO...")
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="test_keyframes_")
    print(f"📁 Created temp directory: {temp_dir}")
    
    success_count = 0
    fail_count = 0
    
    # Test first 5 keyframes
    for idx, kf_meta in enumerate(keyframe_info[:5]):
        frame_number = kf_meta.get('frame_number', idx)
        minio_path = kf_meta.get('minio_path')
        minio_bucket = kf_meta.get('minio_bucket', 'detectifai-keyframes')
        
        print(f"\n📥 Downloading keyframe {idx + 1}/5:")
        print(f"   Bucket: {minio_bucket}")
        print(f"   Path: {minio_path}")
        
        # Download to temp directory
        local_path = os.path.join(temp_dir, f"frame_{frame_number:06d}.jpg")
        
        try:
            db_manager.minio_client.fget_object(minio_bucket, minio_path, local_path)
            
            # Check if file exists and has size
            if os.path.exists(local_path):
                file_size = os.path.getsize(local_path)
                print(f"   ✅ Downloaded successfully: {file_size} bytes")
                success_count += 1
                
                # Try to open as image
                from PIL import Image
                img = Image.open(local_path)
                print(f"   ✅ Valid image: {img.size}")
            else:
                print(f"   ❌ File not found after download")
                fail_count += 1
                
        except Exception as e:
            print(f"   ❌ Download failed: {e}")
            fail_count += 1
    
    # Cleanup
    print(f"\n🧹 Cleaning up temp directory...")
    import shutil
    shutil.rmtree(temp_dir)
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 KEYFRAME RETRIEVAL TEST RESULTS")
    print("=" * 80)
    print(f"✅ Successful downloads: {success_count}/5")
    print(f"❌ Failed downloads: {fail_count}/5")
    
    if success_count == 5:
        print("\n✅ ALL KEYFRAMES CAN BE RETRIEVED!")
        print("   The video captioning should work.")
        print("   The hang might be caused by:")
        print("   - BLIP model loading taking too long")
        print("   - Memory issues during caption generation")
        print("   - Timeout issues in the captioning service")
    elif success_count > 0:
        print(f"\n⚠️  PARTIAL SUCCESS: {success_count}/5 keyframes retrieved")
        print("   Some keyframes are accessible, but not all")
    else:
        print("\n❌ NO KEYFRAMES COULD BE RETRIEVED")
        print("   This is why video captioning is hanging!")
    
    print("=" * 80)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
