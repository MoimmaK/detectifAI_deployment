# Video Captioning Module

A comprehensive vision-language captioning system for surveillance applications that generates neutral, policy-safe captions from video frames and stores them with semantic embeddings for retrieval.

## Features

- **Vision-Language Captioning**: Uses BLIP/similar models to generate descriptive captions
- **LLM-based Sanitization**: Ensures captions are neutral and policy-compliant
- **Semantic Embeddings**: Sentence-BERT embeddings for semantic search
- **Dual Storage**: Relational database for metadata, vector database for embeddings
- **Async Processing**: Non-blocking operations for high throughput
- **Safety-First**: Built-in content filtering and audit logging

## Architecture

```
Frame Input → Vision Model → LLM Sanitizer → Embedding Generator → Storage
     ↓              ↓             ↓               ↓              ↓
  PIL Images    Raw Captions  Safe Captions   Embeddings    DB + Vector Store
```

## Quick Start

```python
from video_captioning import CaptioningService, Frame, CaptioningConfig
from PIL import Image
from datetime import datetime

# Configure the service
config = CaptioningConfig(
    vision_model_name="Salesforce/blip-image-captioning-base",
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize service
service = CaptioningService(config)

# Create frame objects
frame = Frame(
    frame_id="frame_001",
    timestamp=datetime.now(),
    video_id="video_001",
    image=Image.open("frame.jpg")
)

# Process frames
result = service.process_frames([frame])

# Search captions
results = service.search_captions("person walking", top_k=5)
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

The `CaptioningConfig` class provides comprehensive configuration options:

```python
config = CaptioningConfig(
    # Vision model settings
    vision_model_name="Salesforce/blip-image-captioning-base",
    vision_device="cpu",  # or "cuda"
    vision_batch_size=4,
    
    # Embedding settings
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    embedding_normalize=True,
    
    # Processing settings
    enable_async_processing=True,
    max_concurrent_requests=10,
    log_rejected_captions=True
)
```

## Safety Features

The module implements strict safety measures:

- **Content Filtering**: Removes references to gender, race, age, appearance
- **Neutral Language**: Focuses only on observable actions and objects
- **Audit Logging**: Tracks all rejected/modified captions
- **Policy Compliance**: Built-in safety prompt templates

## API Reference

### CaptioningService

Main service class that orchestrates the entire pipeline.

#### Methods

- `process_frames(frames: List[Frame]) -> ProcessingResult`
- `process_frames_async(frames: List[Frame]) -> ProcessingResult`
- `search_captions(query: str, top_k: int = 5) -> List[dict]`
- `get_video_captions(video_id: str) -> List[dict]`
- `get_statistics() -> dict`

### Frame

Input data structure for video frames.

```python
@dataclass
class Frame:
    frame_id: str
    timestamp: datetime
    video_id: str
    image: Image.Image
```

### CaptionRecord

Output data structure for processed captions.

```python
@dataclass
class CaptionRecord:
    caption_id: str
    video_id: str
    frame_id: str
    timestamp: datetime
    raw_caption: str
    sanitized_caption: str
    embedding: np.ndarray
    created_at: datetime
```

## Storage

The module uses a dual storage approach:

1. **Relational Database** (SQLite): Stores caption metadata
2. **Vector Database** (File-based): Stores embeddings for similarity search

### Database Schema

```sql
-- Captions table
CREATE TABLE captions (
    caption_id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    frame_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    raw_caption TEXT NOT NULL,
    sanitized_caption TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Audit table
CREATE TABLE caption_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_caption TEXT NOT NULL,
    sanitized_caption TEXT,
    rejection_reason TEXT,
    created_at TEXT NOT NULL
);
```

## Performance

- **Batch Processing**: Optimized for multiple frames
- **Async Support**: Non-blocking operations
- **Memory Efficient**: Streaming processing for large datasets
- **GPU Acceleration**: CUDA support for models

## Examples

See `example_usage.py` for comprehensive usage examples including:
- Basic frame processing
- Async processing
- Search functionality
- Configuration options

## Integration

This module is designed to integrate with larger surveillance systems:

```python
# In your surveillance pipeline
from video_captioning import CaptioningService

# Initialize once
captioning_service = CaptioningService(config)

# Process frames from video stream
def process_video_segment(frames):
    result = captioning_service.process_frames(frames)
    return result.caption_records

# Search historical data
def search_events(query):
    return captioning_service.search_captions(query)
```

## License

This module is designed for surveillance and security applications with built-in privacy and safety measures.