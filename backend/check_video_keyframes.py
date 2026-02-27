"""
Check where keyframes are actually stored for a video
"""

import sys
from database.config import DatabaseManager

print("=" * 80)
print("🔍 CHECKING VIDEO KEYFRAME STORAGE")
print("=" * 80)

try:
    # Connect to database
    print("\n🔌 Connecting to database...")
    db_manager = DatabaseManager()
    
    # Get the video
    video_id = "video_20260210_134805_fd668d1c"
    print(f"\n📹 Looking for video: {video_id}")
    
    video_file_collection = db_manager.db.video_file
    video = video_file_collection.find_one({"video_id": video_id})
    
    if not video:
        print(f"❌ Video not found: {video_id}")
        sys.exit(1)
    
    print(f"✅ Found video: {video_id}")
    
    # Check meta_data
    print("\n📊 Video document structure:")
    print(f"   Fields: {list(video.keys())}")
    
    if 'meta_data' in video:
        print(f"\n📦 meta_data fields: {list(video['meta_data'].keys())}")
        
        # Check for keyframe info
        if 'keyframe_info' in video['meta_data']:
            keyframe_info = video['meta_data']['keyframe_info']
            print(f"\n✅ Found keyframe_info: {len(keyframe_info)} keyframes")
            
            # Show first keyframe
            if keyframe_info:
                print("\n📸 First keyframe structure:")
                first_kf = keyframe_info[0]
                for key, value in first_kf.items():
                    print(f"   {key}: {value}")
                
                # Check if minio_path exists
                if 'minio_path' in first_kf:
                    print(f"\n🪣 Keyframes are stored in MinIO!")
                    print(f"   Bucket: {first_kf.get('minio_bucket', 'unknown')}")
                    print(f"   Path: {first_kf['minio_path']}")
                    
                    # Try to list objects in MinIO
                    print(f"\n🔍 Checking if keyframes exist in MinIO...")
                    bucket = first_kf.get('minio_bucket', 'detectifai-keyframes')
                    
                    try:
                        # List objects with the video_id prefix
                        objects = db_manager.minio_client.list_objects(
                            bucket,
                            prefix=f"{video_id}/keyframes/",
                            recursive=True
                        )
                        
                        object_list = list(objects)
                        print(f"   Found {len(object_list)} objects in MinIO")
                        
                        if object_list:
                            print("\n📋 First 5 objects:")
                            for obj in object_list[:5]:
                                print(f"      {obj.object_name}")
                        else:
                            print("\n❌ No objects found in MinIO with prefix:")
                            print(f"      {bucket}/{video_id}/keyframes/")
                            
                            # Try to list all objects in the bucket
                            print(f"\n🔍 Listing all objects in bucket: {bucket}")
                            all_objects = db_manager.minio_client.list_objects(
                                bucket,
                                recursive=True
                            )
                            all_list = list(all_objects)
                            print(f"   Total objects in bucket: {len(all_list)}")
                            
                            if all_list:
                                print("\n📋 First 10 objects in bucket:")
                                for obj in all_list[:10]:
                                    print(f"      {obj.object_name}")
                    
                    except Exception as e:
                        print(f"   ❌ Error listing MinIO objects: {e}")
                
                else:
                    print(f"\n⚠️  No minio_path in keyframe info")
                    print(f"   Keyframes might be stored locally or not uploaded")
        
        else:
            print(f"\n❌ No keyframe_info in meta_data")
            
            # Check for other keyframe-related fields
            if 'keyframes_minio_paths' in video['meta_data']:
                paths = video['meta_data']['keyframes_minio_paths']
                print(f"\n✅ Found keyframes_minio_paths: {len(paths)} paths")
                print(f"   First path: {paths[0] if paths else 'none'}")
    
    else:
        print(f"\n❌ No meta_data in video document")
    
    print("\n" + "=" * 80)
    print("✅ CHECK COMPLETE")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
