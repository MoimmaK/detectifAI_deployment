"""
Test script to verify video captioning with MinIO keyframe download
"""

import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

print("=" * 80)
print("🎬 TESTING VIDEO CAPTIONING WITH MINIO KEYFRAME DOWNLOAD")
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
    
    # Get a recent video from MongoDB
    print("\n🔍 Looking for recent videos in MongoDB...")
    print(f"   Database: {db_manager.db.name}")
    
    # Check available collections
    collection_names = db_manager.db.list_collection_names()
    print(f"   Available collections: {collection_names[:15]}")
    
    # Try video_file collection first (correct collection name)
    video_file_collection = db_manager.db.video_file
    print(f"\n   Checking 'video_file' collection...")
    videos = list(video_file_collection.find().sort("upload_date", -1).limit(5))
    
    if not videos:
        # Fallback to 'videos' collection
        print(f"   No videos in 'video_file', checking 'videos' collection...")
        videos_collection = db_manager.db.videos
        videos = list(videos_collection.find().sort("upload_date", -1).limit(5))
    
    if not videos:
        print("\n❌ No videos found in MongoDB")
        print("   Checked collections: 'video_file' and 'videos'")
        
        # Check total counts
        video_file_count = video_file_collection.count_documents({})
        videos_count = db_manager.db.videos.count_documents({})
        print(f"\n📊 Collection counts:")
        print(f"   video_file: {video_file_count} documents")
        print(f"   videos: {videos_count} documents")
        
        if video_file_count == 0 and videos_count == 0:
            print("\n💡 No videos found. Please upload a video first through the API:")
            print("   1. Start the backend: python app.py")
            print("   2. Upload a video through the frontend")
            print("   3. Run this test again")
            sys.exit(1)
    
    print(f"\n📹 Found {len(videos)} videos:")
    for idx, video in enumerate(videos, 1):
        video_id = video.get('video_id', 'unknown')
        upload_date = video.get('upload_date', 'unknown')
        print(f"   {idx}. {video_id} - {upload_date}")
        
        # Check if video has keyframe info
        keyframe_count = 0
        if 'meta_data' in video:
            keyframe_info = video.get('meta_data', {}).get('keyframe_info', [])
            keyframe_count = len(keyframe_info)
        elif 'keyframe_info' in video:
            keyframe_count = len(video.get('keyframe_info', []))
        
        print(f"      Keyframes: {keyframe_count}")
    
    # Use the most recent video
    video = videos[0]
    video_id = video.get('video_id')
    
    if not video_id:
        print("\n❌ Video document doesn't have 'video_id' field")
        print(f"   Available fields: {list(video.keys())}")
        sys.exit(1)
    
    print(f"\n✅ Using video: {video_id}")
    
    # Get keyframe metadata from the video document itself
    print("\n🔍 Looking for keyframes in video metadata...")
    keyframe_info = []
    
    if 'meta_data' in video and 'keyframe_info' in video['meta_data']:
        keyframe_info = video['meta_data']['keyframe_info']
        print(f"✅ Found keyframe_info in video meta_data: {len(keyframe_info)} keyframes")
        
        # Show sample keyframe
        if keyframe_info:
            sample_kf = keyframe_info[0]
            print(f"\n📊 Sample keyframe structure:")
            for key, value in sample_kf.items():
                print(f"   {key}: {value}")
    
    elif 'meta_data' in video and 'keyframes_minio_paths' in video['meta_data']:
        # Reconstruct from minio paths
        minio_paths = video['meta_data']['keyframes_minio_paths']
        print(f"✅ Found keyframes_minio_paths: {len(minio_paths)} paths")
        import re
        for idx, path in enumerate(minio_paths):
            match = re.search(r'frame_(\d+)', path)
            frame_num = int(match.group(1)) if match else idx
            keyframe_info.append({
                'frame_number': frame_num,
                'timestamp': 0.0,
                'minio_path': path,
                'minio_bucket': video.get('keyframe_bucket', 'detectifai-keyframes')
            })
    
    if not keyframe_info:
        print("\n❌ No keyframe metadata found in video document")
        print(f"   Video fields: {list(video.keys())}")
        if 'meta_data' in video:
            print(f"   meta_data fields: {list(video['meta_data'].keys())}")
        print("\n💡 The video may not have been fully processed yet.")
        sys.exit(1)
    
    print(f"\n✅ Prepared {len(keyframe_info)} keyframes for testing")
    
    # Create mock keyframe objects that simulate what the pipeline provides
    print("\n🔨 Creating mock keyframe objects...")
    
    class MockFrameData:
        def __init__(self, frame_number, timestamp, frame_path):
            self.frame_number = frame_number
            self.timestamp = timestamp
            self.frame_path = frame_path  # This will be a MinIO path that doesn't exist locally
    
    class MockKeyframe:
        def __init__(self, frame_data):
            self.frame_data = frame_data
    
    # Create mock keyframes from video's keyframe metadata (use first 5 only)
    mock_keyframes = []
    for kf_meta in keyframe_info[:5]:
        frame_data = MockFrameData(
            frame_number=kf_meta.get('frame_number', 0),
            timestamp=kf_meta.get('timestamp', 0.0),
            frame_path=f"/nonexistent/path/frame_{kf_meta.get('frame_number', 0):06d}.jpg"  # Fake local path
        )
        mock_keyframes.append(MockKeyframe(frame_data))
    
    print(f"✅ Created {len(mock_keyframes)} mock keyframes")
    
    # Show expected MinIO paths
    print("\n📍 Expected MinIO paths (from video metadata):")
    print(f"   Bucket: detectifai-keyframes")
    for kf_meta in keyframe_info[:3]:
        minio_path = kf_meta.get('minio_path', f"{video_id}/keyframes/frame_{kf_meta.get('frame_number', 0):06d}.jpg")
        print(f"   Object: {minio_path}")
    
    print("\n✅ These keyframes should exist in MinIO!")
    print("=" * 80)
    # Test captioning with MinIO download
    print("\n🎬 Testing video captioning with MinIO download...")
    print("=" * 80)
    
    result = captioning_integrator.process_keyframes_with_captioning(
        mock_keyframes,
        video_id=video_id
    )
    
    print("\n" + "=" * 80)
    print("📊 CAPTIONING RESULTS")
    print("=" * 80)
    print(f"Enabled: {result.get('enabled')}")
    print(f"Total captions: {result.get('total_captions')}")
    print(f"Processing time: {result.get('processing_time', 'N/A')}")
    print(f"Errors: {result.get('errors', [])}")
    
    if result.get('captions'):
        print(f"\n✅ Successfully generated {len(result['captions'])} captions!")
        print("\nSample captions:")
        for i, caption in enumerate(result['captions'][:3], 1):
            print(f"\n{i}. Frame: {caption['frame_id']}")
            print(f"   Caption: {caption['sanitized_caption']}")
            print(f"   Timestamp: {caption['timestamp']}")
    else:
        print("\n⚠️ No captions were generated")
        if result.get('errors'):
            print("Errors encountered:")
            for error in result['errors']:
                print(f"  - {error}")
    
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
