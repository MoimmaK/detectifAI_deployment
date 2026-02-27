"""
Database Migration Script: Add Stripe Integration to Subscription Plans

This script updates existing subscription_plans and prepares the database
for Stripe payment integration.

Run this script ONCE after updating the database schema.
"""

from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv
from uuid import uuid4

load_dotenv()

# Connect to MongoDB
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client.get_default_database()

subscription_plans = db.subscription_plans
user_subscriptions = db.user_subscriptions

print("🔄 Starting Stripe integration migration...")

# ========================================
# Step 1: Update existing subscription plans with Stripe data
# ========================================

print("\n📋 Step 1: Updating subscription plans with Stripe data...")

# DetectifAI Basic Plan
basic_plan = subscription_plans.find_one({"plan_name": "Basic"})
if basic_plan:
    subscription_plans.update_one(
        {"_id": basic_plan["_id"]},
        {
            "$set": {
                "stripe_product_id": "prod_TqIuL76gNG4hxu",
                "stripe_price_ids": {
                    "monthly": "price_1SscIsBC7V4mGo7rR4T0YZIc",
                    "yearly": "price_1SscMQBC7V4mGo7rigJ4bFFE"
                },
                "billing_periods": ["monthly", "yearly"],
                "price": 19.00,
                "description": "Essential AI-powered security monitoring",
                "features": "single_video,object_detection,face_recognition,7day_history,dashboard,basic_reports",
                "updated_at": datetime.utcnow()
            }
        }
    )
    print("✅ Updated Basic plan with Stripe integration")
else:
    # Create Basic plan if it doesn't exist
    basic_plan_data = {
        "plan_id": str(uuid4()),
        "plan_name": "Basic",
        "description": "Essential AI-powered security monitoring",
        "price": 19.00,
        "features": "single_video,object_detection,face_recognition,7day_history,dashboard,basic_reports",
        "storage_limit": 50,
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
    subscription_plans.insert_one(basic_plan_data)
    print("✅ Created Basic plan with Stripe integration")

# DetectifAI Pro Plan
pro_plan = subscription_plans.find_one({"plan_name": "Pro"})
if pro_plan:
    subscription_plans.update_one(
        {"_id": pro_plan["_id"]},
        {
            "$set": {
                "stripe_product_id": "prod_TqIyhR08zDDa2B",
                "stripe_price_ids": {
                    "monthly": "price_1SscMwBC7V4mGo7rmmRPTTOz",
                    "yearly": "price_1SscNXBC7V4mGo7rdGgYAYRs"
                },
                "billing_periods": ["monthly", "yearly"],
                "price": 49.00,
                "description": "Advanced security intelligence with extended capabilities",
                "features": "everything_basic,30day_history,behavior_analysis,person_tracking,nlp_search,image_search,custom_reports,priority_queue",
                "updated_at": datetime.utcnow()
            }
        }
    )
    print("✅ Updated Pro plan with Stripe integration")
else:
    # Create Pro plan if it doesn't exist
    pro_plan_data = {
        "plan_id": str(uuid4()),
        "plan_name": "Pro",
        "description": "Advanced security intelligence with extended capabilities",
        "price": 49.00,
        "features": "everything_basic,30day_history,behavior_analysis,person_tracking,nlp_search,image_search,custom_reports,priority_queue",
        "storage_limit": 200,
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
    subscription_plans.insert_one(pro_plan_data)
    print("✅ Created Pro plan with Stripe integration")

# Remove Enterprise plan if it exists (not part of current offering)
enterprise_plan = subscription_plans.find_one({"plan_name": "Enterprise"})
if enterprise_plan:
    subscription_plans.update_one(
        {"_id": enterprise_plan["_id"]},
        {"$set": {"is_active": False, "updated_at": datetime.utcnow()}}
    )
    print("✅ Deactivated Enterprise plan (not in current offering)")

# ========================================
# Step 2: Add Stripe fields to existing user subscriptions
# ========================================

print("\n📋 Step 2: Adding Stripe fields to existing user subscriptions...")

existing_subscriptions = user_subscriptions.find({})
updated_count = 0

for sub in existing_subscriptions:
    # Check if Stripe fields already exist
    if "stripe_customer_id" not in sub:
        user_subscriptions.update_one(
            {"_id": sub["_id"]},
            {
                "$set": {
                    "stripe_customer_id": None,
                    "stripe_subscription_id": None,
                    "billing_period": "monthly",
                    "status": "active",
                    "current_period_start": sub.get("start_date"),
                    "current_period_end": sub.get("end_date"),
                    "cancel_at_period_end": False,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        updated_count += 1

if updated_count > 0:
    print(f"✅ Updated {updated_count} existing subscriptions with Stripe fields")
else:
    print("✅ No existing subscriptions to update")

# ========================================
# Step 3: Verify collections exist
# ========================================

print("\n📋 Step 3: Verifying new collections...")

collections_to_check = [
    "subscription_events",
    "payment_history",
    "subscription_usage"
]

for collection_name in collections_to_check:
    if collection_name in db.list_collection_names():
        count = db[collection_name].count_documents({})
        print(f"✅ Collection '{collection_name}' exists (documents: {count})")
    else:
        print(f"⚠️  Collection '{collection_name}' not found - run database_setup.py first")

# ========================================
# Step 4: Display summary
# ========================================

print("\n" + "="*60)
print("📊 MIGRATION SUMMARY")
print("="*60)

all_plans = list(subscription_plans.find({"is_active": True}))
print(f"\n✅ Active Subscription Plans: {len(all_plans)}")
for plan in all_plans:
    print(f"   • {plan['plan_name']}: ${plan['price']}/month")
    print(f"     Stripe Product: {plan.get('stripe_product_id', 'NOT SET')}")
    print(f"     Billing: {', '.join(plan.get('billing_periods', []))}")

all_subs = user_subscriptions.count_documents({})
print(f"\n✅ Total User Subscriptions: {all_subs}")

print("\n" + "="*60)
print("✅ Migration completed successfully!")
print("="*60)
print("\nNext steps:")
print("1. Test Stripe integration endpoints")
print("2. Create webhook endpoint for Stripe events")
print("3. Test checkout flow with test cards")
print("4. Update frontend pricing components")

client.close()
