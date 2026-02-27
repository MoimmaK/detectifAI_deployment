# Data Flow Diagram - Video Captioning Module

## High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Video Frames  │───▶│  Captioning      │───▶│   Storage &     │
│   (PIL Images)  │    │   Pipeline       │    │   Retrieval     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Detailed Pipeline Flow

```
┌─────────────────┐
│  Input Frames   │
│  - frame_id     │
│  - timestamp    │
│  - video_id     │
│  - PIL Image    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Vision Captioner│
│ (BLIP Model)    │
│ - Batch process │
│ - GPU support   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Raw Captions    │
│ "A man walking  │
│  in blue shirt" │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Caption         │
│ Sanitizer (LLM) │
│ - Safety prompt │
│ - Rule-based    │
│ - Audit logging │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Safe Captions   │
│ "Person walking │
│  in outdoor     │
│  environment"   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Embedding       │
│ Generator       │
│ (Sentence-BERT) │
│ - Normalized    │
│ - Deterministic │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Caption Records │
│ - Metadata      │
│ - Embeddings    │
│ - Timestamps    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐    ┌─────────────────┐
│ Relational DB   │    │  Vector Store   │
│ - SQLite        │    │ - Embeddings    │
│ - Metadata      │    │ - Similarity    │
│ - Audit logs    │    │ - Search index  │
└─────────────────┘    └─────────────────┘
```

## Component Interactions

### 1. Frame Input Processing
```
Frame Object
├── frame_id: str
├── timestamp: datetime
├── video_id: str
└── image: PIL.Image
    └── Validation
        ├── Format check
        ├── Size validation
        └── Error handling
```

### 2. Vision Model Processing
```
Vision Captioner
├── Model Loading
│   ├── BLIP Processor
│   ├── BLIP Model
│   └── Device allocation
├── Batch Processing
│   ├── Image preprocessing
│   ├── Tensor conversion
│   ├── Model inference
│   └── Caption decoding
└── Output: Raw captions
```

### 3. Caption Sanitization
```
Caption Sanitizer
├── LLM Processing
│   ├── Safety prompt template
│   ├── Token generation
│   └── Response parsing
├── Rule-based Fallback
│   ├── Sensitive term filtering
│   ├── Generic replacements
│   └── Content validation
├── Safety Validation
│   ├── Prohibited term check
│   ├── Policy compliance
│   └── Audit logging
└── Output: Safe captions
```

### 4. Embedding Generation
```
Embedding Generator
├── Model Loading
│   ├── Sentence-BERT
│   └── Device allocation
├── Text Processing
│   ├── Tokenization
│   ├── Encoding
│   └── Normalization
└── Output: Vector embeddings
```

### 5. Storage Layer
```
Storage System
├── Relational Database (SQLite)
│   ├── Captions table
│   │   ├── caption_id (PK)
│   │   ├── video_id
│   │   ├── frame_id
│   │   ├── timestamp
│   │   ├── raw_caption
│   │   ├── sanitized_caption
│   │   └── created_at
│   └── Audit table
│       ├── id (PK)
│       ├── raw_caption
│       ├── sanitized_caption
│       ├── rejection_reason
│       └── created_at
└── Vector Database (File-based)
    ├── embeddings.pkl
    │   └── List[np.ndarray]
    └── metadata.json
        └── List[dict] (caption_id, video_id, etc.)
```

## Data Transformations

### Input → Raw Caption
```
PIL Image (224x224x3) → Tensor → BLIP Model → "A person walking down a street"
```

### Raw Caption → Safe Caption
```
"A man in blue shirt walking" → LLM/Rules → "Person walking in outdoor area"
```

### Safe Caption → Embedding
```
"Person walking in outdoor area" → Sentence-BERT → [0.1, -0.3, 0.7, ...] (384-dim)
```

## Async Processing Flow

```
┌─────────────────┐
│  Async Request  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Thread Pool     │
│ Executor        │
└─────────┬───────┘
          │
    ┌─────┴─────┐
    ▼           ▼
┌─────────┐ ┌─────────┐
│ Vision  │ │ Other   │
│ Task    │ │ Tasks   │
└─────────┘ └─────────┘
    │           │
    └─────┬─────┘
          ▼
┌─────────────────┐
│ Await Results   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Continue        │
│ Pipeline        │
└─────────────────┘
```

## Error Handling Flow

```
┌─────────────────┐
│ Processing Step │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Try Operation   │
└─────────┬───────┘
          │
    ┌─────┴─────┐
    ▼           ▼
┌─────────┐ ┌─────────┐
│Success  │ │ Error   │
└─────────┘ └─────┬───┘
    │             │
    │             ▼
    │     ┌─────────────────┐
    │     │ Log Error       │
    │     │ Add to errors[] │
    │     │ Use fallback    │
    │     └─────────┬───────┘
    │               │
    └───────┬───────┘
            ▼
┌─────────────────┐
│ Continue or     │
│ Return Result   │
└─────────────────┘
```

## Search and Retrieval Flow

```
┌─────────────────┐
│ Search Query    │
│ "person walking"│
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Generate Query  │
│ Embedding       │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Vector Search   │
│ - Cosine sim    │
│ - Top-K results │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Fetch Metadata  │
│ from Relational │
│ Database        │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Return Results  │
│ with Similarity │
│ Scores          │
└─────────────────┘
```

This data flow ensures efficient, safe, and scalable processing of video frames into searchable, policy-compliant captions.