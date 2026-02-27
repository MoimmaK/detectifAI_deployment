"""
Working video test with dependency handling
"""

import os
import sys
import subprocess
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


def install_dependencies():
    """Install required dependencies"""
    print("Installing required dependencies...")
    
    packages = [
        "numpy<2",  # Fix NumPy compatibility
        "opencv-python",
        "Pillow",
        "torch",
        "transformers", 
        "sentence-transformers"
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package, "--quiet"
            ])
            print(f"✓ {package} installed")
        except subprocess.CalledProcessError:
            print(f"⚠️  Failed to install {package}")
    
    print("Dependencies installation completed.")


def test_video_processing():
    """Test video processing with actual models"""
    print("\n" + "="*60)
    print("TESTING VIDEO CAPTIONING WITH REAL MODELS")
    print("="*60)
    
    try:
        # Import after dependencies are installed
        from datetime import datetime
        from PIL import Image
        import cv2
        import numpy as np
        
        from models import Frame
        from config import CaptioningConfig
        from captioning_service import CaptioningService
        
        # Find test video
        video_files = [
            "../backend/fight_0002.mp4",
            "../backend/fire.mp4", 
            "../backend/rob.mp4"
        ]
        
        test_video = None
        for video in video_files:
            if os.path.exists(video):
                test_video = video
                break
        
        if not test_video:
            print("No test video found!")
            return
        
        print(f"Using video: {test_video}")
        
        # Extract a few frames
        print("Extracting frames...")
        cap = cv2.VideoCapture(test_video)
        frames = []
        
        for i in range(3):  # Extract 3 frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * 30)  # Every 30 frames
            ret, cv_frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB and PIL
            rgb_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            pil_image = pil_image.resize((224, 224))  # Standard size
            
            frame = Frame(
                frame_id=f"test_frame_{i:03d}",
                timestamp=datetime.now(),
                video_id=Path(test_video).stem,
                image=pil_image
            )
            frames.append(frame)
            print(f"✓ Extracted frame {i+1}")
        
        cap.release()
        
        if not frames:
            print("No frames extracted!")
            return
        
        # Configure service for CPU (safer for testing)
        config = CaptioningConfig(
            vision_model_name="Salesforce/blip-image-captioning-base",
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            vision_device="cpu",
            embedding_device="cpu",
            vision_batch_size=1,  # Process one at a time
            enable_async_processing=False,
            log_rejected_captions=True
        )
        
        print("\nInitializing captioning service...")
        print("(This may take a while to download models on first run)")
        
        service = CaptioningService(config)
        print("✓ Service initialized")
        
        # Process frames
        print(f"\nProcessing {len(frames)} frames...")
        result = service.process_frames(frames)
        
        # Show results
        print(f"\n{'='*50}")
        print("RESULTS")
        print(f"{'='*50}")
        print(f"Success: {result.success}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Records created: {len(result.caption_records)}")
        print(f"Errors: {len(result.errors)}")
        
        if result.errors:
            print("\nErrors:")
            for error in result.errors:
                print(f"  - {error}")
        
        if result.caption_records:
            print(f"\nGenerated Captions:")
            for i, record in enumerate(result.caption_records, 1):
                print(f"\n{i}. Frame: {record.frame_id}")
                print(f"   Raw: {record.raw_caption}")
                print(f"   Safe: {record.sanitized_caption}")
                print(f"   Embedding: {record.embedding.shape}")
        
        # Test search
        print(f"\n{'='*50}")
        print("TESTING SEARCH")
        print(f"{'='*50}")
        
        search_queries = ["person", "activity", "movement"]
        for query in search_queries:
            results = service.search_captions(query, top_k=2)
            print(f"\nSearch '{query}': {len(results)} results")
            for result_item in results:
                similarity = result_item.get('similarity', 0)
                print(f"  - {result_item['sanitized_caption']} ({similarity:.3f})")
        
        # Statistics
        stats = service.get_statistics()
        print(f"\n{'='*50}")
        print("STATISTICS")
        print(f"{'='*50}")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        service.close()
        
        print(f"\n{'='*60}")
        print("🎉 VIDEO TEST COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error during video test: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function"""
    print("Video Captioning Module - Full Test")
    print("This will install dependencies and test with real video")
    
    response = input("\nProceed with installation and testing? (y/n): ")
    if response.lower() != 'y':
        print("Test cancelled.")
        return
    
    # Install dependencies
    install_dependencies()
    
    # Test video processing
    test_video_processing()


if __name__ == "__main__":
    main()