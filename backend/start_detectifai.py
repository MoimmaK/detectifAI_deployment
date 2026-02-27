"""
DetectifAI API Startup Script

Quick script to launch the DetectifAI API server with proper environment setup
and preliminary checks for the surveillance system.
"""

import sys
import os
import subprocess
import time
import logging

def check_python_environment():
    """Check if required Python packages are available"""
    print("🐍 Checking Python environment...")
    
    required_packages = [
        'flask', 'flask_cors', 'opencv-python', 'numpy', 
        'ultralytics', 'pillow', 'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'flask_cors':
                from flask_cors import CORS
            elif package == 'ultralytics':
                from ultralytics import YOLO
            elif package == 'pillow':
                from PIL import Image
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("💡 Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_model_files():
    """Check if YOLO model files are available"""
    print("\n🤖 Checking AI model files...")
    
    model_files = [
        'models/fire_YOLO11.pt',
        'models/weapon_YOLO11.pt'
    ]
    
    missing_models = []
    found_models = []
    
    for model_file in model_files:
        if os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f"  ✅ {model_file} ({size_mb:.1f} MB)")
            found_models.append(model_file)
        else:
            missing_models.append(model_file)
            print(f"  ❌ {model_file}")
    
    if missing_models:
        print(f"\n⚠️ Missing model files: {', '.join(missing_models)}")
        if found_models:
            print(f"✅ Found {len(found_models)} model(s): {', '.join([os.path.basename(f) for f in found_models])}")
            print("💡 DetectifAI will work with available models")
        else:
            print("💡 DetectifAI will work with reduced functionality")
        return len(found_models) > 0
    
    return True

def check_test_videos():
    """Check if test videos are available"""
    print("\n🎬 Checking test videos...")
    
    test_videos = ['rob.mp4', 'fire.avi']
    available_videos = []
    
    for video in test_videos:
        if os.path.exists(video):
            size_mb = os.path.getsize(video) / (1024 * 1024)
            print(f"  ✅ {video} ({size_mb:.1f} MB)")
            available_videos.append(video)
        else:
            print(f"  ❌ {video}")
    
    print(f"\n📊 {len(available_videos)}/{len(test_videos)} test videos available")
    return available_videos

def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directories...")
    
    directories = [
        'uploads',
        'video_processing_outputs',
        'logs',
        'core',
        'docs',
        'models'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"  ✅ Created {directory}/")
        else:
            print(f"  ✅ {directory}/ exists")

def start_detectifai_api():
    """Start the DetectifAI API server"""
    print("\n🚀 Starting DetectifAI API server...")
    print("=" * 50)
    
    try:
        # Change to backend directory if needed
        if not os.path.exists('app.py'):
            print("❌ app.py not found in current directory")
            print("💡 Make sure you're in the backend directory")
            return False
        
        # Start the API server
        print("🌐 API will be available at: http://localhost:5000")
        print("📋 API endpoints:")
        print("  • Health: GET /api/health")
        print("  • Upload: POST /api/upload")
        print("  • Status: GET /api/status/<video_id>")
        print("  • Results: GET /api/results/<video_id>")
        print("  • Demo: GET /api/detectifai/demo")
        print("  • DetectifAI Events: GET /api/detectifai/events/<video_id>")
        print("  • Keyframes: GET /api/keyframes/<video_id>")
        print("")
        print("🔧 To test the API, run: python test_detectifai_integration.py")
        print("🌐 For frontend integration, ensure CORS is enabled")
        print("")
        print("Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Run the API server
        subprocess.run([sys.executable, 'app.py'])
        
    except KeyboardInterrupt:
        print("\n\n🛑 DetectifAI API server stopped")
        return True
    except Exception as e:
        print(f"\n❌ Error starting API server: {e}")
        return False

def main():
    """Main startup function"""
    print("🔧 DetectifAI API Startup")
    print("========================")
    
    # System checks
    env_ok = check_python_environment()
    models_ok = check_model_files()
    videos = check_test_videos()
    
    # Setup
    setup_directories()
    
    # Summary
    print("\n📋 System Status Summary:")
    print(f"  🐍 Python Environment: {'✅' if env_ok else '⚠️'}")
    print(f"  🤖 AI Models: {'✅' if models_ok else '⚠️'}")
    print(f"  🎬 Test Videos: {len(videos)} available")
    
    if not env_ok:
        print("\n❌ Cannot start API - missing required Python packages")
        return False
    
    print(f"\n🎯 DetectifAI System Ready")
    
    if videos:
        print(f"💡 Demo videos available: {', '.join(videos)}")
    
    # Ask user if they want to continue
    try:
        response = input("\n🚀 Start DetectifAI API server? (y/n): ").lower().strip()
        if response in ['y', 'yes', '']:
            return start_detectifai_api()
        else:
            print("👋 Startup cancelled")
            return True
    except KeyboardInterrupt:
        print("\n👋 Startup cancelled")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)