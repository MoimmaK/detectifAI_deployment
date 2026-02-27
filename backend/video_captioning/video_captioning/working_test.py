"""
Working test for video captioning
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from PIL import Image
import cv2

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules
from models import Frame
from config import CaptioningConfig
from captioning_service import CaptioningService


def main():
    """Test video captioning with a real video"""
    print("="*60)
    print("VIDEO CAPTIONING TEST")
    print("="*60)
    
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
            print(f"Found video: {video}")
            break
    
    if not test_video:
        print("No test video found!")
        return
    
    try:
        # Extract 2 frames from video
        print("\nExtracting frames...")
        cap = cv2.VideoCapture(test_video)
        frames = []
        
        for i in range(2):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * 100)  # Every 100 frames
            ret, cv_frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB and PIL
            rgb_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            pil_image = pil_image.resize((224, 224))
            
            frame = Frame(
                frame_id=f"frame_{i:03d}",
                timestamp=datetime.now(),
                video_id=Path(test_video).stem,
                image=pil_image
            )
            frames.append(frame)
            print(f"✓ Frame {i+1} extracted")
        
        cap.release()
        
        if not frames:
            print("No frames extracted!")
            return
        
        # Configure service
        print("\nConfiguring captioning service...")
        config = CaptioningConfig(
            vision_model_name="Salesforce/blip-image-captioning-base",
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            vision_device="cpu",
            embedding_device="cpu",
            vision_batch_size=1,
            enable_async_processing=False
        )
        
        # Initialize service
        print("Initializing service (downloading models if needed)...")
        service = CaptioningService(config)
        print("✓ Service ready")
        
        # Process frames
        print(f"\nProcessing {len(frames)} frames...")
        result = service.process_frames(frames)
        
        # Show results
        print("\n" + "="*50)
        print("RESULTS")
        print("="*50)
        print(f"Success: {result.success}")
        print(f"Time: {result.processing_time:.2f}s")
        print(f"Records: {len(result.caption_records)}")
        print(f"Errors: {len(result.errors)}")
        
        if result.errors:
            print("\nErrors:")
            for error in result.errors:
                print(f"  - {error}")
        
        if result.caption_records:
            print("\nCaptions Generated:")
            for i, record in enumerate(result.caption_records, 1):
                print(f"\n{i}. Frame: {record.frame_id}")
                print(f"   Raw: {record.raw_caption}")
                print(f"   Safe: {record.sanitized_caption}")
        
        # Test search
        print("\n" + "="*50)
        print("SEARCH TEST")
        print("="*50)
        
        queries = ["person", "activity", "scene"]
        for query in queries:
            results = service.search_captions(query, top_k=2)
            print(f"\nSearch '{query}': {len(results)} results")
            for res in results:
                sim = res.get('similarity', 0)
                print(f"  - {res['sanitized_caption']} ({sim:.3f})")
        
        service.close()
        
        print("\n" + "="*60)
        print("🎉 TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()