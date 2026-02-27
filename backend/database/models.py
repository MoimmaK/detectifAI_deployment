"""
Data Models for DetectifAI Database Integration

This module defines data models that map EXACTLY to the MongoDB collections
defined in DetectifAI_db/database_setup.py schema.

CRITICAL RULES:
1. Only use fields defined in the MongoDB schema validators
2. Extra fields must go in meta_data for video_file or use related collections
3. Always convert numpy types before MongoDB operations
4. Timestamps in events must be milliseconds (int/long), not seconds (float)
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from bson import ObjectId
from dataclasses import dataclass, asdict
import json
import numpy as np

# ========================================
# Schema-Compliant Data Models
# ========================================

@dataclass
class VideoFileModel:
    """Maps EXACTLY to video_file collection schema in MongoDB Atlas"""
    # Required fields (from schema)
    video_id: str
    user_id: str
    file_path: str  # MinIO path or local path
    
    # Optional fields (from schema)
    minio_object_key: Optional[str] = None
    minio_bucket: Optional[str] = None
    codec: Optional[str] = None
    fps: Optional[float] = 30.0  # bsonType: double - must be float
    upload_date: Optional[datetime] = None
    duration_secs: Optional[int] = None  # bsonType: int - must be INTEGER not float
    file_size_bytes: Optional[int] = None  # bsonType: long
    meta_data: Optional[Dict] = None  # Store ALL extra fields here (processing_status, resolution, etc.)
    
    _id: Optional[ObjectId] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for MongoDB insertion with proper type conversion"""
        data = asdict(self)
        
        # Set defaults
        if data.get('upload_date') is None:
            data['upload_date'] = datetime.utcnow()
        if data.get('fps') is None:
            data['fps'] = 30.0
        
        # Ensure duration is integer (MongoDB schema requires int)
        if data.get('duration_secs') is not None:
            data['duration_secs'] = int(data['duration_secs'])
        
        # Ensure file_size is integer (MongoDB schema requires long)
        if data.get('file_size_bytes') is not None:
            data['file_size_bytes'] = int(data['file_size_bytes'])
        
        # Ensure fps is float (MongoDB schema requires double)
        if data.get('fps') is not None:
            data['fps'] = float(data['fps'])
        
        return data

@dataclass
class EventModel:
    """Maps EXACTLY to event collection schema in MongoDB Atlas"""
    # Required fields (from schema)
    event_id: str
    video_id: str
    start_timestamp_ms: int  # bsonType: long - MUST be milliseconds as INTEGER
    end_timestamp_ms: int    # bsonType: long - MUST be milliseconds as INTEGER
    
    # Optional fields (from schema)
    event_type: Optional[str] = None  # 'object_detection', 'motion', 'fire', 'weapon', etc.
    confidence_score: Optional[float] = None  # bsonType: double (NOT 'confidence')
    is_verified: bool = False
    is_false_positive: bool = False
    verified_at: Optional[datetime] = None
    verified_by: Optional[str] = None
    visual_embedding: Optional[List[float]] = None  # For future FAISS integration
    bounding_boxes: Optional[Dict] = None  # Store detection bboxes here as object
    
    _id: Optional[ObjectId] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for MongoDB insertion with proper type conversion"""
        data = asdict(self)
        
        # Ensure timestamps are integers (milliseconds) - CRITICAL for MongoDB long type
        data['start_timestamp_ms'] = int(data['start_timestamp_ms'])
        data['end_timestamp_ms'] = int(data['end_timestamp_ms'])
        
        # Ensure confidence_score is float
        if data.get('confidence_score') is not None:
            data['confidence_score'] = float(data['confidence_score'])
        
        # Set default empty arrays/objects for schema compliance
        if data.get('visual_embedding') is None:
            data['visual_embedding'] = []
        if data.get('bounding_boxes') is None:
            data['bounding_boxes'] = {}
        
        return data

@dataclass
class EventDescriptionModel:
    """Maps EXACTLY to event_description collection schema"""
    # Required fields
    description_id: str
    event_id: str
    text_embedding: List[float]  # Required (empty array if not generated yet)
    
    # Optional fields
    caption: Optional[str] = None
    confidence: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    _id: Optional[ObjectId] = None
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        if data.get('created_at') is None:
            data['created_at'] = datetime.utcnow()
        if data.get('updated_at') is None:
            data['updated_at'] = datetime.utcnow()
        # Ensure text_embedding is always a list
        if data.get('text_embedding') is None:
            data['text_embedding'] = []
        return data

@dataclass
class EventCaptionModel:
    """Maps EXACTLY to event_caption collection schema"""
    # Required fields
    description_id: str
    description: str
    _id: Optional[ObjectId] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class EventClipModel:
    """Maps EXACTLY to event_clip collection schema"""
    # Required fields
    clip_id: str
    event_id: str
    clip_path: str
    
    # Optional fields
    thumbnail_path: Optional[str] = None
    minio_object_key: Optional[str] = None
    minio_bucket: Optional[str] = None
    duration_ms: Optional[int] = None  # bsonType: long
    extracted_at: Optional[datetime] = None
    file_size_bytes: Optional[int] = None  # bsonType: long
    _id: Optional[ObjectId] = None
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        if data.get('extracted_at') is None:
            data['extracted_at'] = datetime.utcnow()
        # Ensure integer types
        if data.get('duration_ms') is not None:
            data['duration_ms'] = int(data['duration_ms'])
        if data.get('file_size_bytes') is not None:
            data['file_size_bytes'] = int(data['file_size_bytes'])
        return data

@dataclass
class DetectedFaceModel:
    """Maps EXACTLY to detected_faces collection schema"""
    # Required fields
    face_id: str
    event_id: str
    detected_at: datetime
    
    # Optional fields
    confidence_score: Optional[float] = None
    face_embedding: Optional[List[float]] = None
    minio_object_key: Optional[str] = None
    minio_bucket: Optional[str] = None
    face_image_path: Optional[str] = None
    bounding_boxes: Optional[Dict] = None
    _id: Optional[ObjectId] = None
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        if data.get('face_embedding') is None:
            data['face_embedding'] = []
        return data

@dataclass
class FaceMatchModel:
    """Maps EXACTLY to face_matches collection schema"""
    # Required fields
    match_id: str
    face_id_1: str
    face_id_2: str
    similarity_score: float
    
    # Optional fields
    matched_at: Optional[datetime] = None
    _id: Optional[ObjectId] = None
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        if data.get('matched_at') is None:
            data['matched_at'] = datetime.utcnow()
        return data

# ========================================
# Helper Functions for Type Safety
# ========================================

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for MongoDB compatibility.
    
    MongoDB cannot serialize numpy types directly, causing BSON errors.
    This function ensures all numpy integers become int, numpy floats become float, etc.
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def seconds_to_milliseconds(seconds: float) -> int:
    """Convert seconds (float) to milliseconds (int) for MongoDB long type"""
    return int(seconds * 1000)

def milliseconds_to_seconds(milliseconds: int) -> float:
    """Convert milliseconds (int) to seconds (float) for display"""
    return float(milliseconds) / 1000.0

def prepare_for_mongodb(data: Dict) -> Dict:
    """
    Prepare data dictionary for MongoDB insertion.
    - Remove None ObjectId fields
    - Convert numpy types to Python natives
    """
    # First convert numpy types
    data = convert_numpy_types(data)
    
    # Remove None ObjectId fields
    cleaned_data = {}
    for key, value in data.items():
        if key == '_id' and value is None:
            continue
        cleaned_data[key] = value
    return cleaned_data

def convert_objectid_to_string(doc: Dict) -> Dict:
    """Convert ObjectId fields to strings for JSON serialization"""
    if isinstance(doc, dict):
        for key, value in doc.items():
            if isinstance(value, ObjectId):
                doc[key] = str(value)
            elif isinstance(value, list):
                doc[key] = [
                    convert_objectid_to_string(item) if isinstance(item, dict) 
                    else str(item) if isinstance(item, ObjectId) 
                    else item 
                    for item in value
                ]
            elif isinstance(value, dict):
                doc[key] = convert_objectid_to_string(value)
    return doc


# ========================================
# Subscription & Payment Models
# ========================================

@dataclass
class SubscriptionPlanModel:
    """Maps to subscription_plans collection with Stripe integration"""
    # Required fields
    plan_id: str
    plan_name: str
    price: float
    
    # Optional fields
    description: Optional[str] = None
    features: Optional[str] = None  # Comma-separated feature list
    storage_limit: Optional[int] = None
    is_active: bool = True
    stripe_product_id: Optional[str] = None
    stripe_price_ids: Optional[Dict[str, str]] = None  # {"monthly": "price_xxx", "yearly": "price_xxx"}
    billing_periods: Optional[List[str]] = None  # ["monthly", "yearly"]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    _id: Optional[ObjectId] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for MongoDB insertion"""
        data = asdict(self)
        if data.get('created_at') is None:
            data['created_at'] = datetime.utcnow()
        if data.get('updated_at') is None:
            data['updated_at'] = datetime.utcnow()
        if data.get('stripe_price_ids') is None:
            data['stripe_price_ids'] = {}
        if data.get('billing_periods') is None:
            data['billing_periods'] = []
        return data


@dataclass
class UserSubscriptionModel:
    """Maps to user_subscriptions collection with Stripe integration"""
    # Required fields
    subscription_id: str
    user_id: str
    plan_id: str
    
    # Optional fields
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    billing_period: Optional[str] = None  # "monthly" or "yearly"
    status: Optional[str] = "active"  # 'active', 'canceled', 'past_due', 'trialing'
    current_period_start: Optional[datetime] = None
    current_period_end: Optional[datetime] = None
    cancel_at_period_end: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    _id: Optional[ObjectId] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for MongoDB insertion"""
        data = asdict(self)
        if data.get('start_date') is None:
            data['start_date'] = datetime.utcnow()
        if data.get('created_at') is None:
            data['created_at'] = datetime.utcnow()
        if data.get('updated_at') is None:
            data['updated_at'] = datetime.utcnow()
        return data


@dataclass
class SubscriptionEventModel:
    """Maps to subscription_events collection for audit trail"""
    # Required fields
    event_id: str
    subscription_id: str
    event_type: str  # 'created', 'updated', 'canceled', 'payment_succeeded', etc.
    
    # Optional fields
    stripe_event_id: Optional[str] = None
    event_data: Optional[Dict] = None
    created_at: Optional[datetime] = None
    _id: Optional[ObjectId] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for MongoDB insertion"""
        data = asdict(self)
        if data.get('created_at') is None:
            data['created_at'] = datetime.utcnow()
        if data.get('event_data') is None:
            data['event_data'] = {}
        return data


@dataclass
class PaymentHistoryModel:
    """Maps to payment_history collection for transaction records"""
    # Required fields
    payment_id: str
    user_id: str
    amount: float
    
    # Optional fields
    stripe_payment_intent_id: Optional[str] = None
    currency: str = "USD"
    status: Optional[str] = None  # 'succeeded', 'pending', 'failed'
    payment_method: Optional[str] = None
    created_at: Optional[datetime] = None
    _id: Optional[ObjectId] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for MongoDB insertion"""
        data = asdict(self)
        if data.get('created_at') is None:
            data['created_at'] = datetime.utcnow()
        # Ensure amount is float
        data['amount'] = float(data['amount'])
        return data


@dataclass
class SubscriptionUsageModel:
    """Maps to subscription_usage collection for analytics and limits"""
    # Required fields
    usage_id: str
    user_id: str
    usage_type: str  # 'video_processed', 'storage_used', 'searches_performed'
    
    # Optional fields
    usage_value: Optional[float] = None
    usage_date: Optional[datetime] = None
    created_at: Optional[datetime] = None
    _id: Optional[ObjectId] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for MongoDB insertion"""
        data = asdict(self)
        if data.get('usage_date') is None:
            data['usage_date'] = datetime.utcnow()
        if data.get('created_at') is None:
            data['created_at'] = datetime.utcnow()
        if data.get('usage_value') is not None:
            data['usage_value'] = float(data['usage_value'])
        return data

