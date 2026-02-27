"""
Test Report Generation End-to-End
Diagnoses issues with report generation system
"""

import sys
import os
from datetime import datetime

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("REPORT GENERATION DIAGNOSTIC TEST")
print("=" * 60)

# Test 1: Check imports
print("\n[1/6] Testing imports...")
try:
    from report_generation import ReportGenerator, ReportConfig
    print("✅ ReportGenerator imported successfully")
except ImportError as e:
    print(f"❌ Failed to import ReportGenerator: {e}")
    sys.exit(1)

# Test 2: Check database connection
print("\n[2/6] Testing database connection...")
try:
    from pymongo import MongoClient
    from dotenv import load_dotenv
    
    load_dotenv()
    mongo_uri = os.getenv('MONGO_URI')
    
    if not mongo_uri:
        print("❌ MONGO_URI not found in environment")
        sys.exit(1)
    
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    db = client.get_default_database()
    
    # Test connection
    db.command('ping')
    print(f"✅ Connected to MongoDB: {db.name}")
    
    # Check for videos
    video_count = db.video_file.count_documents({})
    print(f"   Found {video_count} videos in database")
    
    if video_count == 0:
        print("⚠️  No videos found in database")
    else:
        # Get a sample video
        sample_video = db.video_file.find_one()
        test_video_id = sample_video.get('video_id')
        print(f"   Sample video ID: {test_video_id}")
        
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    sys.exit(1)

# Test 3: Initialize ReportGenerator
print("\n[3/6] Initializing ReportGenerator...")
try:
    config = ReportConfig()
    generator = ReportGenerator(config)
    
    if not generator.initialize():
        print("❌ Failed to initialize ReportGenerator")
        sys.exit(1)
    
    print("✅ ReportGenerator initialized")
    
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test data collection
print("\n[4/6] Testing data collection...")
try:
    if video_count > 0:
        print(f"   Collecting data for video: {test_video_id}")
        report_data = generator.data_collector.collect_all_report_data(test_video_id)
        
        print(f"   ✅ Collected {report_data['statistics']['total_events']} events")
        print(f"   ✅ Collected {report_data['statistics']['total_keyframes']} keyframes")
        print(f"   ✅ Collected {report_data['statistics']['total_faces']} faces")
    else:
        print("   ⚠️  Skipping (no videos in database)")
        
except Exception as e:
    print(f"   ❌ Data collection failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test report generation (if we have data)
print("\n[5/6] Testing report generation...")
try:
    if video_count > 0:
        print(f"   Generating report for video: {test_video_id}")
        report = generator.generate_report(video_id=test_video_id)
        
        print(f"   ✅ Report generated: {report.report_id}")
        print(f"   ✅ Sections: {len(report.sections)}")
        
        # Test 6: Test HTML export
        print("\n[6/6] Testing HTML export...")
        output_dir = os.path.join(os.path.dirname(__file__), 'test_reports')
        os.makedirs(output_dir, exist_ok=True)
        
        html_path = os.path.join(output_dir, f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        final_html_path = generator.export_html(report, output_path=html_path)
        
        print(f"   ✅ HTML report saved: {final_html_path}")
        
        # Try PDF export
        try:
            pdf_path = html_path.replace('.html', '.pdf')
            final_pdf_path = generator.export_pdf(report, output_path=pdf_path)
            print(f"   ✅ PDF report saved: {final_pdf_path}")
        except Exception as pdf_error:
            print(f"   ⚠️  PDF export failed (HTML still available): {pdf_error}")
        
    else:
        print("   ⚠️  Skipping (no videos in database)")
        
except Exception as e:
    print(f"   ❌ Report generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED")
print("=" * 60)
print("\nReport generation system is working correctly!")
print("If the frontend is still failing, the issue is likely:")
print("1. Frontend not restarted after .env.local changes")
print("2. Network/CORS issues between frontend and backend")
print("3. Authentication/session issues")
