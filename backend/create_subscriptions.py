"""Create Pro subscriptions for all users (dev utility)."""
from pymongo import MongoClient
from datetime import datetime, timedelta
import uuid

client = MongoClient("mongodb+srv://detectifai_user:DetectifAI123@cluster0.6f9uj.mongodb.net/detectifai")
db = client.detectifai

users = list(db.users.find({}, {"user_id": 1, "email": 1}))
for u in users:
    uid = u["user_id"]
    existing = db.subscriptions.find_one({"user_id": uid})
    if existing:
        print(f"Subscription already exists for {uid}")
        continue
    sub = {
        "subscription_id": str(uuid.uuid4()),
        "user_id": uid,
        "plan_id": "detectifai_pro",
        "status": "active",
        "plan_details": {
            "name": "DetectifAI Pro",
            "features": [
                "single_video", "object_detection", "face_recognition",
                "event_history_7day", "dashboard", "basic_reports", "video_clips",
                "behavior_analysis", "nlp_search", "person_tracking",
                "image_search", "custom_reports", "priority_queue", "event_history_30day",
            ],
            "limits": {
                "video_processing": 999999,
                "nlp_searches": 200,
                "image_searches": 100,
                "concurrent_streams": 1,
                "history_retention_days": 30,
            },
        },
        "current_period_start": datetime.utcnow(),
        "current_period_end": datetime.utcnow() + timedelta(days=365),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    db.subscriptions.insert_one(sub)
    email = u.get("email", "unknown")
    print(f"Created Pro subscription for {email} ({uid})")

print("Done — all users now have Pro subscriptions")
