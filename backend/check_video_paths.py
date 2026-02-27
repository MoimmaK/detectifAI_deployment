from pymongo import MongoClient
import os
from dotenv import load_dotenv
import json

load_dotenv()
client = MongoClient(os.getenv('MONGO_URI'))
db = client.get_default_database()

video_id = 'video_20251120_225446_63d7aa47'
print(f'Checking video: {video_id}')

# Check video_files collection
video = db.video_files.find_one({'video_id': video_id})
if video:
    print('\n=== video_files collection ===')
    print(json.dumps({k: str(v)[:200] for k, v in video.items()}, indent=2))
else:
    print('Not in video_files')

# Check video_file collection (singular)
video2 = db.video_file.find_one({'video_id': video_id})
if video2:
    print('\n=== video_file collection (singular) ===')
    print(json.dumps({k: str(v)[:200] for k, v in video2.items()}, indent=2))
else:
    print('Not in video_file')

# Check what files exist locally
import glob
video_folder = os.path.join('..', 'uploads', video_id)
if os.path.exists(video_folder):
    print(f'\n=== Local files in {video_folder} ===')
    files = glob.glob(os.path.join(video_folder, '**/*'), recursive=True)
    for f in files:
        if os.path.isfile(f):
            size = os.path.getsize(f)
            print(f'{os.path.basename(f)}: {size:,} bytes')
