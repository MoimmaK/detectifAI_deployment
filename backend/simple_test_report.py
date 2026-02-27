"""
Simple direct test of report generation
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Testing report generation...")

try:
    from report_generation import ReportGenerator, ReportConfig
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize
    config = ReportConfig()
    generator = ReportGenerator(config)
    
    print("Initializing...")
    if not generator.initialize():
        print("ERROR: Failed to initialize")
        sys.exit(1)
    
    print("SUCCESS: Generator initialized")
    
    # Try to generate a report for a test video
    # Use a video ID from your database
    test_video_id = "video_20260130_001635_39e22815"  # Replace with actual video ID
    
    print(f"Generating report for {test_video_id}...")
    report = generator.generate_report(video_id=test_video_id)
    
    print(f"SUCCESS: Report generated - {report.report_id}")
    print(f"Sections: {len(report.sections)}")
    
    # Export HTML
    output_dir = "test_reports"
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, "test.html")
    
    final_path = generator.export_html(report, output_path=html_path)
    print(f"SUCCESS: HTML saved to {final_path}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
