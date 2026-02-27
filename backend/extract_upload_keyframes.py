"""
Extract keyframes from videos and upload to MinIO detectifai-keyframes bucket.

For each video that has captions but no keyframes in MinIO:
1. Get the frame_ids from video_captions
2. Get the video source (local file or MinIO)
3. Extract those exact frames using OpenCV
4. Upload to MinIO at {video_id}/frame_XXXXXX.jpg
"""
import os
import sys
import io
import tempfile
import cv2
from pymongo import MongoClient
from minio import Minio

MONGO_URI = "mongodb+srv://detectifai_user:DetectifAI123@cluster0.6f9uj.mongodb.net/detectifai?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client.detectifai

minio_client = Minio('127.0.0.1:9000', access_key='admin', secret_key='adminpassword', secure=False)
KEYFRAME_BUCKET = "detectifai-keyframes"
VIDEO_BUCKET = "detectifai-videos"

BASE_DIR = r"d:\FAST\Final Year Project\sem1_finalized_malaika\sem1"

def get_video_source(video_id):
    """Return path to video file. Download from MinIO if not local."""
    # Check local uploads first
    local_path = os.path.join(BASE_DIR, "uploads", video_id, "video.mp4")
    if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
        print(f"  Using local file: {local_path}")
        return local_path
    
    # Check MinIO
    rec = db.video_file.find_one({"video_id": video_id}, {"minio_object_key": 1, "minio_bucket": 1})
    if rec and rec.get("minio_object_key"):
        bucket = rec.get("minio_bucket", VIDEO_BUCKET)
        obj_key = rec["minio_object_key"]
        
        # Verify the object actually exists before downloading
        try:
            minio_client.stat_object(bucket, obj_key)
        except Exception:
            print(f"  MinIO object not found: {bucket}/{obj_key}")
            return None
        
        print(f"  Downloading from MinIO: {bucket}/{obj_key}")
        tmp_path = os.path.join(tempfile.gettempdir(), f"{video_id}.mp4")
        minio_client.fget_object(bucket, obj_key, tmp_path)
        print(f"  Downloaded to: {tmp_path}")
        return tmp_path
    
    return None


import numpy as np


def upload_placeholder_keyframes(video_id, frame_ids):
    """Generate and upload placeholder keyframe images for videos whose source is gone."""
    uploaded = 0
    
    for frame_id in frame_ids:
        # Get the caption text for this frame to display on placeholder
        caption_doc = db.video_captions.find_one(
            {"video_id": video_id, "frame_id": frame_id},
            {"caption": 1, "_id": 0}
        )
        caption_text = caption_doc.get("caption", "No caption") if caption_doc else "No caption"
        
        # Create a 640x360 dark gradient placeholder image
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        # Dark blue gradient
        for y in range(360):
            val = int(30 + (y / 360) * 40)
            img[y, :] = [val, int(val * 0.8), int(val * 0.5)]
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Video ID
        cv2.putText(img, video_id, (20, 40), font, 0.5, (150, 150, 150), 1)
        # Frame ID
        cv2.putText(img, frame_id, (20, 70), font, 0.5, (150, 150, 150), 1)
        # Camera icon placeholder
        cv2.rectangle(img, (270, 130), (370, 210), (80, 80, 80), 2)
        cv2.putText(img, "VIDEO", (284, 178), font, 0.6, (120, 120, 120), 1)
        # Caption (wrap if long)
        words = caption_text[:80].split()
        line = ""
        y_pos = 250
        for w in words:
            test = line + " " + w if line else w
            if len(test) > 50:
                cv2.putText(img, line, (20, y_pos), font, 0.4, (200, 200, 200), 1)
                y_pos += 22
                line = w
            else:
                line = test
        if line:
            cv2.putText(img, line, (20, y_pos), font, 0.4, (200, 200, 200), 1)
        
        # Encode as JPEG
        success, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            continue
        
        minio_path = f"{video_id}/{frame_id}.jpg"
        data = io.BytesIO(buffer.tobytes())
        minio_client.put_object(
            KEYFRAME_BUCKET, minio_path, data,
            length=len(buffer.tobytes()),
            content_type='image/jpeg'
        )
        uploaded += 1
    
    return uploaded


def extract_and_upload_keyframes(video_id, frame_ids):
    """Extract specific frames from video and upload to MinIO."""
    video_path = get_video_source(video_id)
    if not video_path:
        print(f"  No video source found — generating placeholder keyframes")
        return upload_placeholder_keyframes(video_id, frame_ids)
    
    # Parse frame numbers from frame_ids like "frame_000060"
    frame_numbers = {}
    for fid in frame_ids:
        try:
            num = int(fid.replace("frame_", ""))
            frame_numbers[num] = fid
        except ValueError:
            print(f"  WARNING: Could not parse frame_id: {fid}")
    
    if not frame_numbers:
        print(f"  No valid frame numbers to extract")
        return 0
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Could not open video: {video_path}")
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video: {total_frames} frames, {fps:.1f} fps")
    
    uploaded = 0
    max_frame = max(frame_numbers.keys())
    
    for frame_num in sorted(frame_numbers.keys()):
        if frame_num >= total_frames:
            # Use last available frame
            frame_num_actual = total_frames - 1
            print(f"  Frame {frame_num} beyond total ({total_frames}), using frame {frame_num_actual}")
        else:
            frame_num_actual = frame_num
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num_actual)
        ret, frame = cap.read()
        if not ret:
            print(f"  ERROR: Could not read frame {frame_num_actual}")
            continue
        
        # Encode as JPEG
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            print(f"  ERROR: Could not encode frame {frame_num}")
            continue
        
        frame_id = frame_numbers[frame_num]
        minio_path = f"{video_id}/{frame_id}.jpg"
        
        # Upload to MinIO
        data = io.BytesIO(buffer.tobytes())
        minio_client.put_object(
            KEYFRAME_BUCKET,
            minio_path,
            data,
            length=len(buffer.tobytes()),
            content_type='image/jpeg'
        )
        uploaded += 1
    
    cap.release()
    
    # Clean up temp file if downloaded from MinIO
    tmp_path = os.path.join(tempfile.gettempdir(), f"{video_id}.mp4")
    if os.path.exists(tmp_path) and video_path == tmp_path:
        os.remove(tmp_path)
    
    return uploaded


def main():
    # Get all video_ids with captions
    caption_vids = db.video_captions.distinct("video_id")
    
    for video_id in caption_vids:
        if video_id.startswith("test_"):
            continue
        
        # Check if keyframes already exist in MinIO
        existing = list(minio_client.list_objects(KEYFRAME_BUCKET, prefix=f"{video_id}/", recursive=True))
        if len(existing) > 0:
            print(f"SKIP {video_id}: already has {len(existing)} keyframes in MinIO")
            continue
        
        # Get frame_ids from captions
        frame_ids = db.video_captions.distinct("frame_id", {"video_id": video_id})
        if not frame_ids:
            print(f"SKIP {video_id}: no frame_ids in captions")
            continue
        
        print(f"\nPROCESSING {video_id}: {len(frame_ids)} frames to extract")
        uploaded = extract_and_upload_keyframes(video_id, frame_ids)
        print(f"  Uploaded {uploaded}/{len(frame_ids)} keyframes to MinIO")
    
    print("\n=== DONE ===")
    # Final check
    for video_id in caption_vids:
        if video_id.startswith("test_"):
            continue
        objs = list(minio_client.list_objects(KEYFRAME_BUCKET, prefix=f"{video_id}/", recursive=True))
        print(f"  {video_id}: {len(objs)} keyframes in MinIO")


if __name__ == "__main__":
    main()
