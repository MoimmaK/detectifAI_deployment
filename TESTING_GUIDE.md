# Video Captioning Testing Guide

## Current Situation

The test script shows:
```
❌ No videos found in MongoDB
Please upload a video first through the API
```

This means the MongoDB `videos` collection is empty. You need to upload a video first.

## Testing Options

### Option 1: Simple Test (No Video Upload Required) ⭐ RECOMMENDED

Test the captioning service directly with a local image:

```bash
cd backend
python test_captioning_simple.py
```

**What it does:**
- Uses an existing image file (download.jpeg or images.jpeg)
- Tests the captioning service directly
- Verifies BLIP model works
- No MongoDB video required

**Expected output:**
```
✅ Video captioning integrator initialized
✅ Using test image: download.jpeg
🎬 Testing video captioning...

📊 CAPTIONING RESULTS
✅ Successfully generated 1 caption(s)!

Caption: a person standing in front of a building
```

### Option 2: Full Integration Test (Requires Video Upload)

Test with actual MinIO keyframes:

**Step 1: Upload a video**
```bash
# Start the backend
cd backend
python app.py

# Then upload a video through the frontend
# Wait for processing to complete
```

**Step 2: Run the test**
```bash
cd backend
python test_minio_captioning.py
```

**What it does:**
- Finds a video in MongoDB
- Downloads keyframes from MinIO
- Generates captions
- Saves to MongoDB + FAISS

### Option 3: Enable Captioning in Production

**Step 1: Edit `backend/app.py` (line ~220)**
```python
config.enable_video_captioning = True  # Change from False
```

**Step 2: Restart backend**
```bash
cd backend
python app.py
```

**Step 3: Upload a video through frontend**

**Step 4: Check logs for caption generation**
```
🎬 Starting video captioning on 12 keyframes
📁 Created temporary directory for keyframes
🪣 Using MinIO bucket: detectifai-keyframes
🔍 Attempting to download from MinIO
✅ Downloaded keyframe to temp directory
📝 Processing frames for captioning...
✅ Video captioning complete: 12 captions generated
🧹 Cleaned up temporary directory
```

## Troubleshooting

### "No videos found in MongoDB"

**Cause:** The videos collection is empty

**Solution:** Upload a video first:
1. Start backend: `python app.py`
2. Open frontend in browser
3. Upload a video
4. Wait for processing to complete
5. Run test again

### "No test images found"

**Cause:** The simple test can't find a test image

**Solution:** Place a test image in the backend directory:
```bash
# Copy any jpg/png image to backend directory
cp /path/to/image.jpg backend/test_image.jpg
```

### "MinIO client not available"

**Cause:** Database connection issue

**Solution:** Check `backend/.env` file:
```env
MONGODB_URI=mongodb://localhost:27017/
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=your_access_key
MINIO_SECRET_KEY=your_secret_key
```

### "Failed to download keyframe from MinIO"

**Cause:** Keyframes not in MinIO or wrong bucket

**Solution:**
1. Verify bucket exists: `detectifai-keyframes`
2. Check keyframes were uploaded during video processing
3. Verify MinIO credentials in `.env`

### Memory crash

**Cause:** Insufficient RAM for BLIP model

**Solution:**
- Ensure 8GB+ RAM available
- Close other applications
- Reduce batch size in config
- Or keep captioning disabled

## Quick Commands

### Test captioning service (simple)
```bash
cd backend
python test_captioning_simple.py
```

### Test MinIO integration (requires video)
```bash
cd backend
python test_minio_captioning.py
```

### Check MongoDB videos
```bash
# Using MongoDB shell
mongosh
use DetectifAI_db
db.videos.find().pretty()
```

### Check MinIO keyframes
```bash
# Using MinIO client (mc)
mc ls minio/detectifai-keyframes/
```

## What Each Test Verifies

### test_captioning_simple.py
- ✅ Captioning service initialization
- ✅ BLIP model loading
- ✅ Caption generation
- ✅ MongoDB storage
- ✅ FAISS embeddings
- ❌ MinIO download (not tested)

### test_minio_captioning.py
- ✅ Captioning service initialization
- ✅ MongoDB video retrieval
- ✅ MinIO keyframe download
- ✅ Caption generation
- ✅ MongoDB storage
- ✅ FAISS embeddings
- ✅ Temporary file cleanup

## Recommended Testing Flow

1. **Start with simple test** to verify captioning works:
   ```bash
   python test_captioning_simple.py
   ```

2. **If successful**, upload a video through the frontend

3. **Then test MinIO integration**:
   ```bash
   python test_minio_captioning.py
   ```

4. **If both work**, enable in production:
   - Edit `app.py`: `config.enable_video_captioning = True`
   - Restart backend
   - Upload videos normally

## Expected Results

### Simple Test Success
```
✅ Video captioning integrator initialized
✅ Using test image: download.jpeg
✅ Successfully generated 1 caption(s)!
Caption: a person standing in front of a building
```

### MinIO Test Success
```
✅ Using video: video_20260210_122854_56ccc5c5
📊 Found keyframe_info in meta_data: 12 keyframes
✅ Created 5 mock keyframes
🔍 Attempting to download from MinIO
✅ Downloaded keyframe to temp directory
✅ Successfully generated 5 captions!
🧹 Cleaned up temporary directory
```

## Need Help?

If tests fail, check:
1. MongoDB connection (backend/.env)
2. MinIO connection (backend/.env)
3. Available RAM (8GB+ recommended)
4. Python dependencies (requirements.txt)
5. Log files for detailed errors

---

**Start with the simple test!** It's the fastest way to verify everything works.
