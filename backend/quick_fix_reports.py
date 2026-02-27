"""
Quick Fix Script for Report Generation
Run this to ensure report generation works
"""

import sys
import os

# Ensure we're in the backend directory
backend_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(backend_dir)
sys.path.insert(0, backend_dir)

print("🔧 REPORT GENERATION QUICK FIX")
print("="*60)

# Step 1: Verify imports
print("\n[1/4] Checking imports...")
try:
    from report_generation import ReportGenerator, ReportConfig
    from report_generation.data_collector import DataCollector
    from report_generation.report_builder import ReportGenerator as RG
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease ensure report_generation module is in backend/")
    sys.exit(1)

# Step 2: Check database
print("\n[2/4] Checking database connection...")
try:
    from pymongo import MongoClient
    from dotenv import load_dotenv
    
    load_dotenv()
    mongo_uri = os.getenv('MONGO_URI')
    
    if not mongo_uri:
        print("❌ MONGO_URI not in environment")
        print("Add to backend/.env file")
        sys.exit(1)
    
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    db = client.get_default_database()
    db.command('ping')
    print(f"✅ Connected to: {db.name}")
    
    # Check for videos
    video_count = db.video_file.count_documents({})
    print(f"✅ Found {video_count} videos")
    
    if video_count == 0:
        print("\n⚠️  WARNING: No videos in database")
        print("Upload a video before generating reports")
        sys.exit(0)
    
    # Get sample video
    sample = db.video_file.find_one()
    video_id = sample.get('video_id')
    print(f"✅ Sample video: {video_id}")
    
except Exception as e:
    print(f"❌ Database error: {e}")
    sys.exit(1)

# Step 3: Test initialization
print("\n[3/4] Testing initialization...")
try:
    config = ReportConfig()
    generator = ReportGenerator(config)
    
    if not generator.initialize():
        print("❌ Initialization failed")
        sys.exit(1)
    
    print("✅ Generator initialized")
    
except Exception as e:
    print(f"❌ Init error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Quick generation test
print("\n[4/4] Testing report generation...")
print(f"Generating for: {video_id}")
print("(This takes 2-5 minutes...)")

try:
    report = generator.generate_report(video_id=video_id)
    print(f"✅ Report generated: {report.report_id}")
    
    # Export HTML
    output_dir = os.path.join('video_processing_outputs', video_id, 'reports')
    os.makedirs(output_dir, exist_ok=True)
    
    html_path = os.path.join(output_dir, 'test_report.html')
    generator.export_html(report, output_path=html_path)
    print(f"✅ HTML saved: {html_path}")
    
    print("\n" + "="*60)
    print("✅ SUCCESS - REPORT GENERATION WORKING!")
    print("="*60)
    print("\nYour backend is ready.")
    print("Make sure to:")
    print("1. Restart Flask backend")
    print("2. Restart Next.js frontend")
    print("3. Try from UI")
    
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    print("\nCheck the error above and:")
    print("1. Ensure all data_collector.py fixes are applied")
    print("2. Ensure all report_builder.py fixes are applied")
    print("3. Check MongoDB has valid data")
    sys.exit(1)
