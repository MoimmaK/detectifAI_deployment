"""
Check if a video has been compressed and where it's stored
"""

import sys
from database.config import DatabaseManager

print("=" * 80)
print("🔍 CHECKING VIDEO COMPRESSION STATUS")
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
    
    # Check for compression info
    print("\n📊 Checking compression status...")
    
    if 'meta_data' in video:
        meta_data = video['meta_data']
        
        # Check for compression_info
        if 'compression_info' in meta_data:
            comp_info = meta_data['compression_info']
            print("\n✅ Compression info found:")
            for key, value in comp_info.items():
                print(f"   {key}: {value}")
        else:
            print("\n❌ No compression_info in meta_data")
        
        # Check for minio_compressed_path
        if 'minio_compressed_path' in meta_data:
            print(f"\n✅ MinIO compressed path: {meta_data['minio_compressed_path']}")
        else:
            print("\n❌ No minio_compressed_path in meta_data")
        
        # Check processing status
        if 'processing_status' in meta_data:
            print(f"\n📊 Processing status: {meta_data['processing_status']}")
            print(f"   Progress: {meta_data.get('processing_progress', 'unknown')}%")
            print(f"   Message: {meta_data.get('processing_message', 'none')}")
    else:
        print("\n❌ No meta_data in video document")
    
    # Check if compressed video exists in MinIO
    print("\n🔍 Checking MinIO for compressed video...")
    minio_path = f"compressed/{video_id}/video.mp4"
    bucket = db_manager.config.minio_video_bucket
    
    try:
        # Try to get object info
        stat = db_manager.minio_client.stat_object(bucket, minio_path)
        print(f"\n✅ Compressed video EXISTS in MinIO!")
        print(f"   Bucket: {bucket}")
        print(f"   Path: {minio_path}")
        print(f"   Size: {stat.size / (1024*1024):.2f} MB")
        print(f"   Last modified: {stat.last_modified}")
        
        # Try to generate presigned URL
        from datetime import timedelta
        url = db_manager.minio_client.presigned_get_object(
            bucket,
            minio_path,
            expires=timedelta(hours=1)
        )
        print(f"\n✅ Presigned URL generated:")
        print(f"   {url[:100]}...")
        
    except Exception as e:
        print(f"\n❌ Compressed video NOT FOUND in MinIO")
        print(f"   Bucket: {bucket}")
        print(f"   Path: {minio_path}")
        print(f"   Error: {e}")
        
        # List what's actually in the bucket for this video
        print(f"\n🔍 Listing all objects for video {video_id}...")
        try:
            objects = db_manager.minio_client.list_objects(
                bucket,
                prefix=f"{video_id}/",
                recursive=True
            )
            object_list = list(objects)
            
            if object_list:
                print(f"   Found {len(object_list)} objects:")
                for obj in object_list[:10]:
                    print(f"      {obj.object_name}")
            else:
                print(f"   No objects found with prefix: {video_id}/")
                
                # Try compressed prefix
                print(f"\n🔍 Trying compressed prefix...")
                objects = db_manager.minio_client.list_objects(
                    bucket,
                    prefix=f"compressed/{video_id}/",
                    recursive=True
                )
                object_list = list(objects)
                
                if object_list:
                    print(f"   Found {len(object_list)} objects:")
                    for obj in object_list:
                        print(f"      {obj.object_name}")
                else:
                    print(f"   No objects found with prefix: compressed/{video_id}/")
        except Exception as list_error:
            print(f"   Error listing objects: {list_error}")
    
    print("\n" + "=" * 80)
    print("✅ CHECK COMPLETE")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
