"""
Debug video processing to find why compression is failing
"""

import sys
import os
import logging
from database.config import DatabaseManager
from config import get_security_focused_config

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

print("=" * 80)
print("🔍 DEBUGGING VIDEO PROCESSING & COMPRESSION")
print("=" * 80)

try:
    # Connect to database
    print("\n🔌 Connecting to database...")
    db_manager = DatabaseManager()
    
    # Get the stuck video
    video_id = "video_20260212_143538_fc066b61"
    print(f"\n📹 Analyzing video: {video_id}")
    
    video_file_collection = db_manager.db.video_file
    video = video_file_collection.find_one({"video_id": video_id})
    
    if not video:
        print(f"❌ Video not found: {video_id}")
        sys.exit(1)
    
    print(f"✅ Found video: {video_id}")
    
    # Check processing status
    print("\n" + "=" * 80)
    print("📊 PROCESSING STATUS")
    print("=" * 80)
    
    meta_data = video.get('meta_data', {})
    
    status = meta_data.get('processing_status', 'unknown')
    progress = meta_data.get('processing_progress', 0)
    message = meta_data.get('processing_message', 'none')
    
    print(f"Status: {status}")
    print(f"Progress: {progress}%")
    print(f"Message: {message}")
    
    # Check what steps completed
    print("\n" + "=" * 80)
    print("✅ COMPLETED STEPS")
    print("=" * 80)
    
    # Keyframes
    keyframe_count = meta_data.get('keyframe_count', 0)
    print(f"1. Keyframes: {keyframe_count} extracted")
    
    # Object detection
    detection_count = meta_data.get('detection_count', 0)
    print(f"2. Object Detection: {detection_count} detections")
    
    # Behavior analysis
    behavior_count = meta_data.get('behavior_count', 0)
    print(f"3. Behavior Analysis: {behavior_count} behaviors")
    
    # Facial recognition
    face_count = meta_data.get('face_count', 0)
    print(f"4. Facial Recognition: {face_count} faces")
    
    # Video captioning
    caption_count = meta_data.get('total_captions', 0)
    print(f"5. Video Captioning: {caption_count} captions")
    
    # Compression
    compression_info = meta_data.get('compression_info')
    if compression_info:
        print(f"6. Compression: ✅ COMPLETED")
        print(f"   Original size: {compression_info.get('original_size_bytes', 0) / (1024*1024):.2f} MB")
        print(f"   Compressed size: {compression_info.get('compressed_size_bytes', 0) / (1024*1024):.2f} MB")
        print(f"   Ratio: {compression_info.get('compression_ratio', 0):.1f}%")
    else:
        print(f"6. Compression: ❌ NOT STARTED")
    
    # Check configuration
    print("\n" + "=" * 80)
    print("⚙️  CONFIGURATION CHECK")
    print("=" * 80)
    
    config = get_security_focused_config()
    
    print(f"enable_object_detection: {config.enable_object_detection}")
    print(f"enable_facial_recognition: {config.enable_facial_recognition}")
    print(f"enable_behavior_analysis: {getattr(config, 'enable_behavior_analysis', False)}")
    print(f"enable_video_captioning: {getattr(config, 'enable_video_captioning', False)}")
    print(f"generate_compressed_video: {config.generate_compressed_video}")
    
    # Identify the problem
    print("\n" + "=" * 80)
    print("🔍 PROBLEM ANALYSIS")
    print("=" * 80)
    
    if progress < 95:
        print(f"\n❌ ISSUE: Video stuck at {progress}%")
        print(f"   Message: {message}")
        print(f"\n📍 Current step: {message}")
        
        if "captioning" in message.lower():
            print("\n⚠️  VIDEO CAPTIONING IS BLOCKING THE PIPELINE!")
            print("\nPossible causes:")
            print("1. BLIP model loading is hanging")
            print("2. Caption generation is taking too long")
            print("3. Memory issues during processing")
            print("4. Error not being caught properly")
            print("\nSolution:")
            print("- Temporarily disable video captioning")
            print("- Or fix the captioning timeout/error handling")
            
        elif "behavior" in message.lower():
            print("\n⚠️  BEHAVIOR ANALYSIS IS BLOCKING!")
            
        elif "facial" in message.lower():
            print("\n⚠️  FACIAL RECOGNITION IS BLOCKING!")
            
        elif "detection" in message.lower():
            print("\n⚠️  OBJECT DETECTION IS BLOCKING!")
        
        print(f"\n💡 Compression runs at 95% progress")
        print(f"   Current progress: {progress}%")
        print(f"   Need to advance: {95 - progress}%")
        
    elif not compression_info:
        print("\n❌ ISSUE: Reached 95% but compression didn't run")
        print("\nPossible causes:")
        print("1. generate_compressed_video is False")
        print("2. Compression service failed silently")
        print("3. FFmpeg not available")
        print("4. MinIO upload failed")
        
    else:
        print("\n✅ Video processed successfully!")
        print("   Compression completed")
    
    # Check if video file exists
    print("\n" + "=" * 80)
    print("📁 VIDEO FILE CHECK")
    print("=" * 80)
    
    minio_path = video.get('minio_object_key')
    bucket = video.get('minio_bucket', 'detectifai-videos')
    
    if minio_path:
        print(f"\nOriginal video:")
        print(f"  Bucket: {bucket}")
        print(f"  Path: {minio_path}")
        
        try:
            stat = db_manager.minio_client.stat_object(bucket, minio_path)
            print(f"  ✅ EXISTS: {stat.size / (1024*1024):.2f} MB")
        except Exception as e:
            print(f"  ❌ NOT FOUND: {e}")
    
    # Check FFmpeg availability
    print("\n" + "=" * 80)
    print("🔧 FFMPEG CHECK")
    print("=" * 80)
    
    import subprocess
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✅ FFmpeg available: {version_line}")
        else:
            print(f"❌ FFmpeg error: {result.stderr}")
    except FileNotFoundError:
        print("❌ FFmpeg NOT FOUND")
        print("   Compression will use OpenCV (slower, lower quality)")
    except Exception as e:
        print(f"❌ FFmpeg check failed: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)
    
    if progress < 95:
        print(f"\n❌ ROOT CAUSE: Video stuck at {progress}% - '{message}'")
        print(f"   Compression never started (runs at 95%)")
        print(f"\n🔧 FIX: Resolve the blocking step to allow processing to continue")
    elif not compression_info:
        print(f"\n❌ ROOT CAUSE: Reached 95% but compression failed")
        print(f"\n🔧 FIX: Check compression service logs and FFmpeg availability")
    else:
        print(f"\n✅ Video processed successfully")
        print(f"   Check MinIO for compressed video at: compressed/{video_id}/video.mp4")
    
    print("\n" + "=" * 80)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
