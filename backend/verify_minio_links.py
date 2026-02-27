import sys
import os
import logging
from datetime import datetime
import traceback

# Redirect output to file
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'verify_log.txt')
log_file = open(log_path, 'w', encoding='utf-8')
sys.stdout = log_file
sys.stderr = log_file

# Setup logging to file as well
logging.basicConfig(stream=log_file, level=logging.INFO)

# Add sem1/backend to path so we can import report_generation
# Assumed location: sem1/backend/verify_minio_links.py
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Add sem1 to path (parent of backend)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

print(f"Path: {sys.path}")
sys.stdout.flush()

# Import as part of backend package to support relative imports
from backend.report_generation import ReportGenerator
from backend.report_generation.config import ReportConfig

def test():
    print("Initializing Generator (skipping LLM)...")
    config = ReportConfig()
    gen = ReportGenerator(config)
    
    # Manually initialize data collector
    gen.data_collector = DataCollector(config)
    gen._initialized = True # spoof initialization to skip LLM loading
    success = True
    if not success:
        print("Failed to initialize generator")
        return
    
    if gen.data_collector.db is None:
        print("DB not connected")
        return

    # Find a video
    # Try video_file or video_metadata
    video = gen.data_collector.db.video_file.find_one()
    if not video:
        video = gen.data_collector.db.video_metadata.find_one()
        
    if not video:
        print("No video found in DB")
        return
    
    vid = video['video_id']
    print(f"Processing video {vid}")
    
    # Collect Data
    data = gen.data_collector.collect_all_report_data(vid)
    
    print(f"Metadata Video URL: {data['metadata'].get('video_url', 'NOT FOUND')}")
    
    if data['keyframes']:
        print(f"Keyframe 0 URL: {data['keyframes'][0].get('image_url', 'NOT FOUND')}")
        print(f"Keyframe 0 Path: {data['keyframes'][0].get('image_path', 'NOT FOUND')}")
    else:
        print("No keyframes found")
        
    if data['faces']:
         print(f"Face 0 URL: {data['faces'][0].get('crop_url', 'NOT FOUND')}")

    # Generate Report
    report = gen.generate_report(vid)
    
    # Check sections for links
    for section in report.sections:
        if section.name == 'header':
            if 'Download/View Video' in section.content:
                print("✅ Video Link found in header")
            else:
                print("❌ Video Link NOT found in header")
        if section.name == 'evidence':
            if section.images:
                img = section.images[0]
                if img.get('url'):
                    print(f"✅ Evidence image has URL: {img.get('url')[:50]}...")
                else:
                    print("❌ Evidence image missing URL")

    # Export
    output_dir = os.path.join(current_dir, 'test_output')
    os.makedirs(output_dir, exist_ok=True)
    html_path = gen.export_html(report, output_path=os.path.join(output_dir, f"report_{vid}.html"))
    print(f"Report generated at {html_path}")

if __name__ == "__main__":
    test()
