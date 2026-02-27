"""Check video records and extract/upload missing keyframes to MinIO."""
import os
import sys
from pymongo import MongoClient
from minio import Minio

MONGO_URI = "mongodb+srv://detectifai_user:DetectifAI123@cluster0.6f9uj.mongodb.net/detectifai?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client.detectifai

minio_client = Minio('127.0.0.1:9000', access_key='admin', secret_key='adminpassword', secure=False)

# 1. Find all video_ids that have captions
caption_vids = db.video_captions.distinct("video_id")
print(f"Videos with captions: {len(caption_vids)}")

# 2. Check which have keyframes in MinIO
for vid in caption_vids:
    if vid.startswith("test_"):
        continue
    objs = list(minio_client.list_objects("detectifai-keyframes", prefix=f"{vid}/", recursive=True))
    print(f"  {vid}: {len(objs)} keyframes in MinIO")
    
    # Get the frame_ids from captions
    frames = db.video_captions.distinct("frame_id", {"video_id": vid})
    print(f"    Caption frame_ids: {frames}")

print()

# 3. Check video_files and video_file collections
for coll_name in ['video_files', 'video_file']:
    coll = db[coll_name]
    count = coll.count_documents({})
    print(f"\n{coll_name}: {count} docs total")
    sample = coll.find_one()
    if sample:
        print(f"  Sample keys: {list(sample.keys())}")
        vid_key = sample.get('video_id', sample.get('_id'))
        print(f"  Sample id: {vid_key}")

# 4. Check which videos have local files
print("\n=== Local file check ===")
base = r"d:\FAST\Final Year Project\sem1_finalized_malaika\sem1"
for vid in caption_vids:
    if vid.startswith("test_"):
        continue
    locations = []
    for subdir in ["uploads", "video_processing_outputs"]:
        p = os.path.join(base, subdir, vid)
        if os.path.isdir(p):
            files = []
            for root, dirs, fnames in os.walk(p):
                for f in fnames:
                    files.append(os.path.join(root, f))
            locations.append((subdir, files))
    
    if locations:
        for subdir, files in locations:
            print(f"  {vid} in {subdir}: {len(files)} files")
            for f in files[:5]:
                print(f"    {f}")
    else:
        # Check MinIO for original video
        vid_objs = list(minio_client.list_objects("detectifai-videos", prefix=vid, recursive=True))
        if vid_objs:
            print(f"  {vid}: in MinIO detectifai-videos ({len(vid_objs)} objects)")
            for o in vid_objs[:3]:
                print(f"    {o.object_name} ({o.size} bytes)")
        else:
            print(f"  {vid}: NO local files, NO MinIO video")
