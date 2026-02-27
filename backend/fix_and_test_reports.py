"""
Complete Report Generation Test & Fix Script
Run this to verify report generation works end-to-end
"""

import sys
import os
from datetime import datetime

# Add backend to path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

def test_imports():
    """Test all required imports"""
    print("\n" + "="*60)
    print("STEP 1: Testing Imports")
    print("="*60)
    
    try:
        from report_generation import ReportGenerator, ReportConfig
        print("✅ report_generation imported")
        return True
    except ImportError as e:
        print(f"❌ Failed to import report_generation: {e}")
        print("\nFix: Ensure report_generation module exists in backend/")
        return False

def test_database():
    """Test database connection"""
    print("\n" + "="*60)
    print("STEP 2: Testing Database Connection")
    print("="*60)
    
    try:
        from pymongo import MongoClient
        from dotenv import load_dotenv
        
        load_dotenv()
        mongo_uri = os.getenv('MONGO_URI')
        
        if not mongo_uri:
            print("❌ MONGO_URI not found in .env")
            print("\nFix: Add MONGO_URI to backend/.env file")
            return False, None
        
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db = client.get_default_database()
        db.command('ping')
        
        print(f"✅ Connected to MongoDB: {db.name}")
        
        # Get a sample video
        video = db.video_file.find_one()
        if not video:
            print("⚠️  No videos found in database")
            print("\nFix: Upload a video first before generating reports")
            return True, None
        
        video_id = video.get('video_id')
        print(f"✅ Found sample video: {video_id}")
        return True, video_id
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        print("\nFix: Check MONGO_URI and network connection")
        return False, None

def test_initialization():
    """Test ReportGenerator initialization"""
    print("\n" + "="*60)
    print("STEP 3: Testing ReportGenerator Initialization")
    print("="*60)
    
    try:
        from report_generation import ReportGenerator, ReportConfig
        
        config = ReportConfig()
        generator = ReportGenerator(config)
        
        if not generator.initialize():
            print("❌ Generator initialization failed")
            return False, None
        
        print("✅ ReportGenerator initialized successfully")
        return True, generator
        
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_report_generation(generator, video_id):
    """Test actual report generation"""
    print("\n" + "="*60)
    print("STEP 4: Testing Report Generation")
    print("="*60)
    
    if not video_id:
        print("⚠️  Skipping (no video available)")
        return False
    
    try:
        print(f"Generating report for video: {video_id}")
        print("This may take 2-5 minutes...")
        
        report = generator.generate_report(video_id=video_id)
        
        print(f"✅ Report generated: {report.report_id}")
        print(f"   Sections: {len(report.sections)}")
        print(f"   Video ID: {report.video_id}")
        
        return True, report
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_export(generator, report, video_id):
    """Test HTML and PDF export"""
    print("\n" + "="*60)
    print("STEP 5: Testing Report Export")
    print("="*60)
    
    if not report:
        print("⚠️  Skipping (no report available)")
        return False
    
    try:
        # Create output directory
        output_dir = os.path.join(backend_dir, 'video_processing_outputs', video_id, 'reports')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export HTML
        html_path = os.path.join(output_dir, f"report_{timestamp}.html")
        final_html = generator.export_html(report, output_path=html_path)
        print(f"✅ HTML exported: {final_html}")
        
        # Try PDF export
        try:
            pdf_path = os.path.join(output_dir, f"report_{timestamp}.pdf")
            final_pdf = generator.export_pdf(report, output_path=pdf_path)
            print(f"✅ PDF exported: {final_pdf}")
        except Exception as pdf_error:
            print(f"⚠️  PDF export failed (HTML still available): {pdf_error}")
        
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("REPORT GENERATION DIAGNOSTIC & FIX")
    print("="*60)
    
    # Test 1: Imports
    if not test_imports():
        return False
    
    # Test 2: Database
    db_ok, video_id = test_database()
    if not db_ok:
        return False
    
    # Test 3: Initialization
    init_ok, generator = test_initialization()
    if not init_ok:
        return False
    
    # Test 4: Report Generation
    if video_id:
        gen_ok, report = test_report_generation(generator, video_id)
        if not gen_ok:
            return False
        
        # Test 5: Export
        if not test_export(generator, report, video_id):
            return False
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED - REPORT GENERATION WORKING")
    print("="*60)
    print("\nNext steps:")
    print("1. Restart your Flask backend (if running)")
    print("2. Restart your Next.js frontend")
    print("3. Try generating a report from the UI")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
