"""
Test script to check keyframe structure from database service
(Same flow as actual video processing)
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
print("🔍 TESTING KEYFRAME STRUCTURE FROM DATABASE SERVICE")
print("=" * 80)

try:
    from config import get_security_focused_config
    from database.config import DatabaseManager
    from database_video_service import DatabaseIntegratedVideoService
    
    # Initialize database service
    print("\n📦 Initializing database service...")
    config = get_security_focused_config()
    config.enable_video_captioning = True
    db_service = DatabaseIntegratedVideoService(config)
    
    print("✅ Database service initialized")
    
    # Get a recent video from MongoDB
    print("\n🔍 Looking for recent videos in MongoDB...")
    videos = list(db_service.video_repo.collection.find().sort("upload_date", -1).limit(5))
    
    if not videos:
        print("❌ No videos found in MongoDB")
        print("Please upload a video first through the API")
        sys.exit(1)
    
    print(f"\n📹 Found {len(videos)} recent videos:")
    for idx, video in enumerate(videos, 1):
        print(f"   {idx}. {video.get('video_id')} - {video.get('upload_date')}")
    
    # Use the most recent video
    video = videos[0]
    video_id = video['video_id']
    print(f"\n✅ Using video: {video_id}")
    
    # Check if video has keyframes in metadata
    keyframe_info = video.get('meta_data', {}).get('keyframe_info', [])
    print(f"\n📊 Video has {len(keyframe_info)} keyframes in metadata")
    
    if keyframe_info:
        print("\n🔍 Sample keyframe metadata:")
        sample = keyframe_info[0]
        print(f"   Frame number: {sample.get('frame_number')}")
        print(f"   Timestamp: {sample.get('timestamp')}")
        print(f"   MinIO path: {sample.get('minio_path')}")
        print(f"   MinIO bucket: {sample.get('minio_bucket')}")
    
    # Now extract keyframes using the video processor (same as pipeline)
    print("\n🎬 Extracting keyframes using video processor...")
    
    # Get video file path from MinIO or local
    minio_path = video.get('minio_object_key')
    if minio_path:
        print(f"   Video in MinIO: {minio_path}")
        # Download from MinIO temporarily
        import tempfile
        temp_video_path = tempfile.mktemp(suffix='.mp4')
        try:
            db_service.video_repo.minio.fget_object(
                db_service.video_repo.video_bucket,
                minio_path,
                temp_video_path
            )
            video_path = temp_video_path
            print(f"   ✅ Downloaded to: {video_path}")
        except Exception as e:
            print(f"   ❌ Failed to download from MinIO: {e}")
            # Try local path
            video_path = video.get('file_path')
            if not video_path or not os.path.exists(video_path):
                print(f"   ❌ Local path doesn't exist: {video_path}")
                print("\n💡 Tip: The video is in MinIO but couldn't be downloaded.")
                print("   Let's use the keyframes that are already extracted!")
                
                # Use existing keyframes from MinIO metadata
                print("\n🔄 Using keyframes from MinIO metadata instead...")
                keyframe_info = video.get('meta_data', {}).get('keyframe_info', [])
                
                if keyframe_info:
                    print(f"   Found {len(keyframe_info)} keyframes in metadata")
                    print("\n📊 Keyframe structure from metadata:")
                    sample = keyframe_info[0]
                    for key, value in sample.items():
                        print(f"      {key}: {value}")
                    
                    print("\n✅ These keyframes are stored in MinIO")
                    print(f"   Bucket: {sample.get('minio_bucket')}")
                    print(f"   Path pattern: {sample.get('minio_path')}")
                    
                    print("\n" + "=" * 80)
                    print("⚠️  ISSUE IDENTIFIED:")
                    print("=" * 80)
                    print("The keyframes are stored in MinIO, not as local files!")
                    print("The video_captioning_integrator expects local file paths.")
                    print("\nSOLUTION: The integrator needs to:")
                    print("1. Download keyframes from MinIO, OR")
                    print("2. Accept keyframe images directly instead of file paths")
                    print("=" * 80)
                    
                sys.exit(0)
    else:
        video_path = video.get('file_path')
        print(f"   Using local path: {video_path}")
    
    # Extract keyframes
    keyframes = db_service.video_processor.extract_keyframes(video_path)
    print(f"\n✅ Extracted {len(keyframes)} keyframes")
    
    if keyframes:
        print("\n📊 Analyzing first keyframe structure:")
        kf = keyframes[0]
        print(f"   Type: {type(kf)}")
        print(f"   Class name: {kf.__class__.__name__}")
        
        # List all attributes
        attrs = [attr for attr in dir(kf) if not attr.startswith('_')]
        print(f"   Attributes ({len(attrs)}): {attrs[:20]}")  # Show first 20
        
        # Check for frame_path
        print("\n🔍 Checking for frame_path:")
        if hasattr(kf, 'frame_path'):
            print(f"   ✅ kf.frame_path: {kf.frame_path}")
            print(f"   File exists: {os.path.exists(kf.frame_path)}")
        else:
            print(f"   ❌ No kf.frame_path")
        
        # Check for frame_data
        print("\n🔍 Checking for frame_data:")
        if hasattr(kf, 'frame_data'):
            print(f"   ✅ Has frame_data: {type(kf.frame_data)}")
            fd_attrs = [attr for attr in dir(kf.frame_data) if not attr.startswith('_')]
            print(f"   frame_data attributes: {fd_attrs[:20]}")
            
            if hasattr(kf.frame_data, 'frame_path'):
                print(f"   ✅ frame_data.frame_path: {kf.frame_data.frame_path}")
                print(f"   File exists: {os.path.exists(kf.frame_data.frame_path)}")
            else:
                print(f"   ❌ No frame_data.frame_path")
        else:
            print(f"   ❌ No frame_data")
        
        # Check for timestamp
        print("\n🔍 Checking for timestamp:")
        if hasattr(kf, 'timestamp'):
            print(f"   ✅ kf.timestamp: {kf.timestamp}")
        elif hasattr(kf, 'frame_data') and hasattr(kf.frame_data, 'timestamp'):
            print(f"   ✅ frame_data.timestamp: {kf.frame_data.timestamp}")
        else:
            print(f"   ❌ No timestamp")
        
        # Check for frame_index/frame_number
        print("\n🔍 Checking for frame_index/frame_number:")
        if hasattr(kf, 'frame_index'):
            print(f"   ✅ kf.frame_index: {kf.frame_index}")
        elif hasattr(kf, 'frame_number'):
            print(f"   ✅ kf.frame_number: {kf.frame_number}")
        elif hasattr(kf, 'frame_data'):
            if hasattr(kf.frame_data, 'frame_number'):
                print(f"   ✅ frame_data.frame_number: {kf.frame_data.frame_number}")
            elif hasattr(kf.frame_data, 'frame_index'):
                print(f"   ✅ frame_data.frame_index: {kf.frame_data.frame_index}")
            else:
                print(f"   ❌ No frame_index/frame_number in frame_data")
        else:
            print(f"   ❌ No frame_index/frame_number")
        
        # Try to access the actual values
        print("\n🔍 Trying to access actual values:")
        try:
            # Try different ways to get frame_path
            frame_path = None
            if hasattr(kf, 'frame_path'):
                frame_path = kf.frame_path
            elif hasattr(kf, 'frame_data') and hasattr(kf.frame_data, 'frame_path'):
                frame_path = kf.frame_data.frame_path
            
            if frame_path:
                print(f"   ✅ Got frame_path: {frame_path}")
                print(f"   File exists: {os.path.exists(frame_path)}")
                
                # Try to open the image
                from PIL import Image
                img = Image.open(frame_path)
                print(f"   ✅ Successfully opened image: {img.size}")
            else:
                print(f"   ❌ Could not get frame_path")
                
        except Exception as e:
            print(f"   ❌ Error accessing values: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nThis shows the exact keyframe structure that video_captioning_integrator receives")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
