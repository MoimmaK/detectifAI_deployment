from pymongo import MongoClient
from bson.objectid import ObjectId
import os
from dotenv import load_dotenv
import json

load_dotenv()
client = MongoClient(os.getenv('MONGO_URI'))
db = client.get_default_database()

event_id = '691f5635981b041b294e90ba'
print(f'Event ID: {event_id}, Length: {len(event_id)}')

print('\n=== Checking event (singular) collection ===')
try:
    evt = db.event.find_one({'_id': ObjectId(event_id)})
    print('By _id:', evt is not None)
    if evt:
        print(json.dumps({k: str(v)[:100] for k, v in evt.items()}, indent=2))
except Exception as e:
    print(f'Error with _id: {e}')

evt2 = db.event.find_one({'event_id': event_id})
print('\nBy event_id field:', evt2 is not None)
if evt2:
    print(json.dumps({k: str(v)[:100] for k, v in evt2.items()}, indent=2))

print('\n=== Sample event from collection ===')
sample = db.event.find_one({})
if sample:
    print(json.dumps({k: str(v)[:100] for k, v in sample.items()}, indent=2))
