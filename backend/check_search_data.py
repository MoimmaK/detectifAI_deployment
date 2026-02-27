"""Quick check of MongoDB collections for search pipeline."""
from pymongo import MongoClient
client = MongoClient("mongodb+srv://detectifai_user:DetectifAI123@cluster0.6f9uj.mongodb.net/detectifai")
db = client.detectifai

# Check keyframes collection
kf_count = db.keyframes.count_documents({})
print(f"keyframes count: {kf_count}")
if kf_count > 0:
    sample = db.keyframes.find_one()
    print(f"keyframes keys: {list(sample.keys())}")
    for k in db.keyframes.find({}, {"frame_id":1,"minio_bucket":1,"minio_object_name":1,"video_id":1}).limit(3):
        fid = str(k.get("frame_id","?"))[:20]
        print(f"  frame_id={fid} bucket={k.get('minio_bucket')} obj={k.get('minio_object_name')}")

# Check frames collection
fr_count = db.frames.count_documents({})
print(f"\nframes count: {fr_count}")
if fr_count > 0:
    sample = db.frames.find_one()
    print(f"frames keys: {list(sample.keys())}")
    for f in db.frames.find({}, {"frame_id":1,"minio_bucket":1,"minio_object_name":1}).limit(3):
        fid = str(f.get("frame_id","?"))[:20]
        print(f"  frame_id={fid} bucket={f.get('minio_bucket')} obj={f.get('minio_object_name')}")

# Check what frame_ids look like in video_captions
print("\nvideo_captions samples:")
for vc in db.video_captions.find({}, {"frame_id":1,"video_id":1,"sanitized_caption":1}).limit(5):
    print(f"  frame_id={vc.get('frame_id')} video_id={vc.get('video_id')} caption={str(vc.get('sanitized_caption',''))[:50]}")

# Check all collections
print(f"\nAll collections: {db.list_collection_names()}")

# Check MinIO-related data
print("\n--- Checking for MinIO image references ---")
# event_clips might have image references
ec_count = db.event_clips.count_documents({})
print(f"event_clips count: {ec_count}")
if ec_count > 0:
    sample = db.event_clips.find_one()
    print(f"event_clips keys: {list(sample.keys())}")

# Check event_captions
ecc = db.event_captions.count_documents({})
print(f"event_captions count: {ecc}")
if ecc > 0:
    sample = db.event_captions.find_one()
    print(f"event_captions keys: {list(sample.keys())}")

client.close()
