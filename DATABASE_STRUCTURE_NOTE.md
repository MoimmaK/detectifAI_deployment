# Database Structure Analysis

## Current Database Structure

Based on your MongoDB data, here's what we found:

### Collections

1. **`video_file`** - Uploaded videos
   - Contains video metadata
   - Example: `video_20251115_233346_1c1f20d9`
   - MinIO path: `original/video_20251115_233346_1c1f20d9/video.mp4`
   - Bucket: `detectifai-videos`

2. **`keyframes`** - Extracted keyframes
   - Contains keyframe data from **live streams**
   - camera_id: `webcam_01`
   - MinIO path: `live/webcam_01/20251120_125611_681861.jpg`
   - These are NOT from uploaded videos!

### The Issue

The keyframes in your database are from **live stream processing**, not from uploaded video processing. They have:
- `camera_id` instead of `video_id`
- MinIO paths like `live/webcam_01/TIMESTAMP.jpg`
- Different structure than expected

The video captioning integrator expects keyframes from uploaded videos with paths like:
- `{video_id}/keyframes/frame_000000.jpg`
- Stored in `detectifai-keyframes` bucket

## Why the Test Fails

The `test_minio_captioning.py` script looks for:
1. Videos in `video_file` collection ✅ (found)
2. Keyframes with format `{video_id}/keyframes/frame_XXXXXX.jpg` ❌ (not found)

Your keyframes are from live streams, not uploaded videos.

## Solution: Upload a Video Through the Pipeline

To test video captioning properly, you need to:

### Option 1: Upload a Video (Recommended for Full Test)

1. **Enable video captioning** in `backend/app.py`:
   ```python
   config.enable_video_captioning = True  # Line ~220
   ```

2. **Start the backend**:
   ```bash
   cd backend
   python app.py
   ```

3. **Upload a video** through the frontend
   - This will trigger the full pipeline
   - Keyframes will be extracted and uploaded to MinIO
   - Keyframes will have the correct format: `{video_id}/keyframes/frame_XXXXXX.jpg`

4. **Check the logs** for caption generation

### Option 2: Simple Test (No Upload Required)

Test the captioning service directly:

```bash
cd backend
python test_captioning_simple.py
```

This tests the captioning service without needing videos in the database.

## Expected Video Processing Flow

When you upload a video through the API:

1. **Video uploaded** → Stored in MinIO `detectifai-videos` bucket
2. **Video record created** → Stored in `video_file` collection
3. **Keyframes extracted** → Uploaded to MinIO `detectifai-keyframes` bucket
   - Path format: `{video_id}/keyframes/frame_000000.jpg`
   - Metadata stored in video document
4. **Video captioning** (if enabled)
   - Downloads keyframes from MinIO
   - Generates captions
   - Saves to MongoDB `video_captions` collection
   - Creates embeddings in FAISS

## Current vs Expected Keyframe Structure

### Current (Live Stream)
```json
{
  "camera_id": "webcam_01",
  "minio_path": "live/webcam_01/20251120_125611_681861.jpg",
  "timestamp": 1.018,
  "frame_index": 15
}
```

### Expected (Uploaded Video)
```json
{
  "video_id": "video_20251115_233346_1c1f20d9",
  "minio_path": "video_20251115_233346_1c1f20d9/keyframes/frame_000000.jpg",
  "minio_bucket": "detectifai-keyframes",
  "timestamp": 0.0,
  "frame_number": 0
}
```

## Recommendation

**Use the simple test first** to verify captioning works:

```bash
cd backend
python test_captioning_simple.py
```

This will confirm the captioning service is working correctly. Then upload a video through the frontend to test the full integration with MinIO keyframes.

## Why Live Stream Keyframes Won't Work

The video captioning integrator is designed for uploaded videos and expects:
- Keyframes linked to a `video_id`
- MinIO paths in format: `{video_id}/keyframes/frame_XXXXXX.jpg`
- Keyframes stored in `detectifai-keyframes` bucket

Live stream keyframes have a different structure and are processed differently.

---

**Next Step**: Run `python backend/test_captioning_simple.py` to verify the captioning service works!
