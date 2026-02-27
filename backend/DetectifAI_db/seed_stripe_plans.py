"""
Seed Stripe-Integrated Subscription Plans

This script populates the subscription_plans collection with accurate
DetectifAI Basic and Pro plans connected to Stripe.
"""

from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv
from uuid import uuid4

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client.get_default_database()
subscription_plans = db.subscription_plans

print("🌱 Seeding Stripe-integrated subscription plans...")

# DetectifAI Basic Plan
basic_plan = {
    "plan_id": "detectifai_basic",
    "plan_name": "DetectifAI Basic",
    "description": "Essential AI-powered security monitoring for single installations",
    "price": 19.00,
    "features": [
        "single_video",
        "object_detection",
        "face_recognition",
        "event_history_7day",
        "dashboard",
        "basic_reports",
        "video_clips"
    ],
    "limits": {
        "video_processing": 10,  # Videos per month
        "history_retention_days": 7,
        "nlp_searches": 0,  # Not available in Basic
        "image_searches": 0,  # Not available in Basic
        "concurrent_streams": 1
    },
    "is_active": True,
    "stripe_product_id": "prod_TqIuL76gNG4hxu",
    "stripe_price_ids": {
        "monthly": "price_1SscIsBC7V4mGo7rR4T0YZIc",
        "yearly": "price_1SscMQBC7V4mGo7rigJ4bFFE"
    },
    "billing_periods": ["monthly", "yearly"],
    "created_at": datetime.utcnow(),
    "updated_at": datetime.utcnow()
}

# DetectifAI Pro Plan
pro_plan = {
    "plan_id": "detectifai_pro",
    "plan_name": "DetectifAI Pro",
    "description": "Advanced security intelligence with extended capabilities",
    "price": 49.00,
    "features": [
        "single_video",
        "object_detection",
        "face_recognition",
        "event_history_30day",
        "dashboard",
        "basic_reports",
        "video_clips",
        "behavior_analysis",
        "person_tracking",
        "nlp_search",
        "image_search",
        "custom_reports",
        "priority_queue"
    ],
    "limits": {
        "video_processing": 999999,  # Unlimited videos per month for Pro
        "history_retention_days": 30,
        "nlp_searches": 200,  # NLP searches per month
        "image_searches": 100,  # Image searches per month
        "concurrent_streams": 1
    },
    "is_active": True,
    "stripe_product_id": "prod_TqIyhR08zDDa2B",
    "stripe_price_ids": {
        "monthly": "price_1SscMwBC7V4mGo7rmmRPTTOz",
        "yearly": "price_1SscNXBC7V4mGo7rdGgYAYRs"
    },
    "billing_periods": ["monthly", "yearly"],
    "created_at": datetime.utcnow(),
    "updated_at": datetime.utcnow()
}

# Upsert plans
for plan in [basic_plan, pro_plan]:
    result = subscription_plans.update_one(
        {"plan_id": plan["plan_id"]},
        {"$set": plan},
        upsert=True
    )
    if result.upserted_id:
        print(f"✅ Created plan: {plan['plan_name']}")
    else:
        print(f"✅ Updated plan: {plan['plan_name']}")

# Display summary
print("\n" + "="*60)
print("📊 SUBSCRIPTION PLANS")
print("="*60)

all_plans = list(subscription_plans.find({"is_active": True}))
for plan in all_plans:
    print(f"\n{plan['plan_name']} - ${plan['price']}/month")
    print(f"  Description: {plan['description']}")
    
    # Only print if exists (for compatibility with old plans)
    if 'stripe_product_id' in plan:
        print(f"  Stripe Product: {plan['stripe_product_id']}")
    
    if 'stripe_price_ids' in plan:
        monthly_price = plan['stripe_price_ids'].get('monthly', 'N/A')
        yearly_price = plan['stripe_price_ids'].get('yearly', 'N/A')
        print(f"  Monthly Price ID: {monthly_price}")
        print(f"  Yearly Price ID: {yearly_price}")
    
    if 'features' in plan:
        features = plan['features']
        if isinstance(features, list):
            print(f"  Features: {', '.join(features)}")
        else:
            print(f"  Features: {features}")
    
    if 'limits' in plan:
        print(f"  Limits:")
        for limit_name, limit_value in plan['limits'].items():
            print(f"    - {limit_name}: {limit_value}")

print("\n✅ Subscription plans seeded successfully!")

client.close()
