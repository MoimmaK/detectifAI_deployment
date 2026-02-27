"""
Integration example showing how to use the video captioning module
with an existing surveillance system
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from PIL import Image
import cv2
import numpy as np

# Add the parent directory to path to import from backend
sys.path.append(str(Path(__file__).parent.parent))

from video_captioning import CaptioningService, Frame, CaptioningConfig


class SurveillanceIntegration:
    """Example integration with surveillance system"""
    
    def __init__(self):
        # Configure captioning service
        self.config = CaptioningConfig(
            vision_model_name="Salesforce/blip-image-captioning-base",
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            db_connection_string="surveillance_captions.db",
            vector_db_path="./surveillance_vectors",
            enable_async_processing=True,
            log_rejected_captions=True,
            vision_batch_size=8  # Process more frames at once
        )
        
        self.captioning_service = CaptioningService(self.config)
        print("Captioning service initialized")
    
    def extract_frames_from_video(self, video_path: str, 
                                 frame_interval: int = 30) -> list:
        """Extract frames from video file at specified intervals"""
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                return frames
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, cv_frame = cap.read()
                if not ret:
                    break
                
                # Extract frame at intervals
                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    
                    # Create Frame object
                    timestamp = datetime.now()
                    timestamp = timestamp.replace(
                        microsecond=int((frame_count / fps) * 1000000) % 1000000
                    )
                    
                    frame = Frame(
                        frame_id=f"frame_{frame_count:06d}",
                        timestamp=timestamp,
                        video_id=Path(video_path).stem,
                        image=pil_image
                    )
                    
                    frames.append(frame)
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            print(f"Extracted {extracted_count} frames from {video_path}")
            
        except Exception as e:
            print(f"Error extracting frames: {e}")
        
        return frames
    
    def process_video_file(self, video_path: str):
        """Process a complete video file"""
        print(f"\nProcessing video: {video_path}")
        
        # Extract frames
        frames = self.extract_frames_from_video(video_path, frame_interval=60)
        
        if not frames:
            print("No frames extracted")
            return
        
        # Process frames in batches
        batch_size = 10
        all_records = []
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(frames)-1)//batch_size + 1}")
            
            result = self.captioning_service.process_frames(batch)
            
            if result.success:
                all_records.extend(result.caption_records)
                print(f"  Processed {len(result.caption_records)} frames")
            else:
                print(f"  Batch failed with {len(result.errors)} errors")
                for error in result.errors:
                    print(f"    - {error}")
        
        print(f"Total processed: {len(all_records)} caption records")
        
        # Show sample results
        if all_records:
            print("\nSample captions:")
            for i, record in enumerate(all_records[:3]):
                print(f"  Frame {record.frame_id}:")
                print(f"    Raw: {record.raw_caption}")
                print(f"    Safe: {record.sanitized_caption}")
                print()
    
    def search_events(self, query: str, top_k: int = 5):
        """Search for events using natural language"""
        print(f"\nSearching for: '{query}'")
        
        results = self.captioning_service.search_captions(query, top_k=top_k)
        
        if results:
            print(f"Found {len(results)} similar events:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. Video: {result['video_id']}")
                print(f"     Frame: {result['frame_id']}")
                print(f"     Caption: {result['sanitized_caption']}")
                print(f"     Similarity: {result.get('similarity', 0):.3f}")
                print(f"     Time: {result['timestamp']}")
                print()
        else:
            print("No similar events found")
    
    def get_video_summary(self, video_id: str):
        """Get summary of all captions for a video"""
        print(f"\nVideo summary for: {video_id}")
        
        captions = self.captioning_service.get_video_captions(video_id)
        
        if captions:
            print(f"Total frames: {len(captions)}")
            print("Timeline:")
            for caption in captions[:10]:  # Show first 10
                print(f"  {caption['timestamp']}: {caption['sanitized_caption']}")
            
            if len(captions) > 10:
                print(f"  ... and {len(captions) - 10} more frames")
        else:
            print("No captions found for this video")
    
    def show_statistics(self):
        """Display system statistics"""
        stats = self.captioning_service.get_statistics()
        
        print("\n=== System Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Show rejected captions if any
        rejected = self.captioning_service.get_rejected_captions()
        if rejected:
            print(f"\nRejected captions: {len(rejected)}")
            for rejection in rejected[:3]:  # Show first 3
                print(f"  Raw: {rejection['raw']}")
                print(f"  Reason: {rejection['reason']}")
    
    def close(self):
        """Cleanup resources"""
        self.captioning_service.close()
        print("Integration closed")


def main():
    """Main demonstration function"""
    integration = SurveillanceIntegration()
    
    try:
        # Example 1: Process a video file (if available)
        video_files = [
            "../backend/fight_0002.mp4",
            "../backend/fire.mp4",
            "../backend/rob.mp4"
        ]
        
        processed_any = False
        for video_file in video_files:
            if os.path.exists(video_file):
                integration.process_video_file(video_file)
                processed_any = True
                break
        
        if not processed_any:
            print("No video files found, creating sample data...")
            # Create sample frames for demonstration
            sample_frames = []
            for i in range(5):
                # Create test images with different colors
                image = Image.new('RGB', (640, 480), 
                                color=(50 + i*40, 100 + i*30, 150 + i*20))
                frame = Frame(
                    frame_id=f"demo_frame_{i:03d}",
                    timestamp=datetime.now(),
                    video_id="demo_video",
                    image=image
                )
                sample_frames.append(frame)
            
            result = integration.captioning_service.process_frames(sample_frames)
            print(f"Processed {len(result.caption_records)} demo frames")
        
        # Example 2: Search functionality
        integration.search_events("person walking")
        integration.search_events("movement in scene")
        
        # Example 3: Video summary
        integration.get_video_summary("demo_video")
        
        # Example 4: Statistics
        integration.show_statistics()
        
    except Exception as e:
        print(f"Error in main: {e}")
    
    finally:
        integration.close()


if __name__ == "__main__":
    main()