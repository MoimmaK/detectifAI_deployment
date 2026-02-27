"""
MongoDB-based storage layer for captions and embeddings
Replaces SQLite with MongoDB Atlas integration
"""

import logging
import json
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
import numpy as np
import pickle
from pathlib import Path
from pymongo import MongoClient
from pymongo.collection import Collection

try:
    from .models import CaptionRecord
    from .config import CaptioningConfig
except ImportError:
    from models import CaptionRecord
    from config import CaptioningConfig


class MongoDBCaptionStorage:
    """Handles storage of captions and embeddings using MongoDB and FAISS"""
    
    def __init__(self, config: CaptioningConfig, db_manager=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_manager = db_manager
        
        # Initialize databases
        self._init_mongodb()
        self._init_vector_db()
    
    def _init_mongodb(self):
        """Initialize MongoDB connection for caption metadata"""
        try:
            if self.db_manager:
                # Use existing database manager
                self.db = self.db_manager.db
                self.logger.info("Using existing MongoDB connection from db_manager")
            else:
                # Create new connection
                mongo_uri = self.config.db_connection_string or os.getenv(
                    'MONGO_URI',
                    'mongodb+srv://detectifai_user:DetectifAI123@cluster0.6f9uj.mongodb.net/detectifai?retryWrites=true&w=majority&appName=Cluster0'
                )
                client = MongoClient(mongo_uri)
                self.db = client['detectifai']
                self.logger.info("Created new MongoDB connection")
            
            # Get or create captions collection
            self.captions_collection = self.db['video_captions']
            
            # Create indexes for efficient querying
            self.captions_collection.create_index("caption_id", unique=True)
            self.captions_collection.create_index("video_id")
            self.captions_collection.create_index("frame_id")
            self.captions_collection.create_index("timestamp")
            
            # Create audit collection for rejected captions
            self.audit_collection = self.db['caption_audit']
            self.audit_collection.create_index("created_at")
            
            self.logger.info("✅ MongoDB caption storage initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize MongoDB: {e}")
            raise
    
    def _init_vector_db(self):
        """Initialize FAISS vector database for embeddings"""
        try:
            # Create vector store directory
            vector_path = Path(self.config.vector_db_path or "./video_captioning_store")
            vector_path.mkdir(exist_ok=True, parents=True)
            
            self.vector_db_path = vector_path
            self.embeddings_file = vector_path / "caption_embeddings.pkl"
            self.metadata_file = vector_path / "caption_metadata.json"
            
            # Load existing data if available
            self._load_vector_data()
            
            self.logger.info("✅ FAISS vector database initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize vector database: {e}")
            raise
    
    def _load_vector_data(self):
        """Load existing vector data from FAISS"""
        try:
            if self.embeddings_file.exists() and self.metadata_file.exists():
                # Load embeddings
                with open(self.embeddings_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                
                # Load metadata
                with open(self.metadata_file, 'r') as f:
                    self.vector_metadata = json.load(f)
                
                self.logger.info(f"📦 Loaded {len(self.embeddings)} existing embeddings from FAISS")
            else:
                self.embeddings = []
                self.vector_metadata = []
                self.logger.info("🆕 Initialized empty FAISS vector store")
                
        except Exception as e:
            self.logger.error(f"⚠️ Failed to load vector data: {e}")
            self.embeddings = []
            self.vector_metadata = []
    
    def _save_vector_data(self):
        """Save vector data to FAISS disk storage"""
        try:
            # Check if Python is shutting down
            import sys
            if sys.meta_path is None:
                return  # Python is shutting down, skip save
            
            # Ensure directory exists
            self.vector_db_path.mkdir(exist_ok=True, parents=True)
            
            # Save embeddings
            import builtins
            with builtins.open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
            
            # Save metadata
            with builtins.open(self.metadata_file, 'w') as f:
                json.dump(self.vector_metadata, f, indent=2)
            
            self.logger.debug(f"💾 Saved {len(self.embeddings)} embeddings to FAISS")
                
        except Exception as e:
            # Ignore shutdown errors
            if "sys.meta_path is None" not in str(e) and "Python is likely shutting down" not in str(e):
                self.logger.error(f"❌ Failed to save vector data: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
    
    def store_caption_record(self, record: CaptionRecord) -> bool:
        """Store a single caption record in MongoDB and FAISS"""
        try:
            # Prepare document for MongoDB
            caption_doc = {
                "caption_id": record.caption_id,
                "video_id": record.video_id,
                "frame_id": record.frame_id,
                "timestamp": record.timestamp.isoformat() if isinstance(record.timestamp, datetime) else str(record.timestamp),
                "raw_caption": record.raw_caption,
                "sanitized_caption": record.sanitized_caption,
                "created_at": record.created_at.isoformat() if isinstance(record.created_at, datetime) else datetime.now().isoformat()
            }
            
            # Store in MongoDB (upsert to avoid duplicates)
            self.captions_collection.update_one(
                {"caption_id": record.caption_id},
                {"$set": caption_doc},
                upsert=True
            )
            
            # Store embedding in FAISS vector database
            self.embeddings.append(record.embedding)
            self.vector_metadata.append({
                'caption_id': record.caption_id,
                'video_id': record.video_id,
                'frame_id': record.frame_id,
                'timestamp': caption_doc['timestamp']
            })
            
            # Save vector data to disk
            self._save_vector_data()
            
            self.logger.debug(f"✅ Stored caption: {record.caption_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store caption record: {e}")
            return False
    
    def store_caption_records_batch(self, records: List[CaptionRecord]) -> int:
        """Store multiple caption records in batch"""
        stored_count = 0
        
        try:
            # Prepare documents for MongoDB
            caption_docs = []
            embeddings_batch = []
            metadata_batch = []
            
            for record in records:
                caption_doc = {
                    "caption_id": record.caption_id,
                    "video_id": record.video_id,
                    "frame_id": record.frame_id,
                    "timestamp": record.timestamp.isoformat() if isinstance(record.timestamp, datetime) else str(record.timestamp),
                    "raw_caption": record.raw_caption,
                    "sanitized_caption": record.sanitized_caption,
                    "created_at": record.created_at.isoformat() if isinstance(record.created_at, datetime) else datetime.now().isoformat()
                }
                caption_docs.append(caption_doc)
                
                embeddings_batch.append(record.embedding)
                metadata_batch.append({
                    'caption_id': record.caption_id,
                    'video_id': record.video_id,
                    'frame_id': record.frame_id,
                    'timestamp': caption_doc['timestamp']
                })
            
            # Batch insert into MongoDB (using bulk write for upserts)
            from pymongo import UpdateOne
            operations = [
                UpdateOne(
                    {"caption_id": doc["caption_id"]},
                    {"$set": doc},
                    upsert=True
                )
                for doc in caption_docs
            ]
            
            result = self.captions_collection.bulk_write(operations)
            
            # Batch insert into FAISS vector database
            self.embeddings.extend(embeddings_batch)
            self.vector_metadata.extend(metadata_batch)
            
            # Save vector data to disk
            self._save_vector_data()
            
            stored_count = len(records)
            self.logger.info(f"✅ Stored {stored_count} caption records in MongoDB + FAISS")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store caption records batch: {e}")
        
        return stored_count
    
    def get_caption_by_id(self, caption_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve caption by ID from MongoDB"""
        try:
            doc = self.captions_collection.find_one({"caption_id": caption_id})
            if doc:
                # Remove MongoDB _id field
                doc.pop('_id', None)
                return doc
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get caption by ID: {e}")
            return None
    
    def get_captions_by_video(self, video_id: str) -> List[Dict[str, Any]]:
        """Retrieve all captions for a video from MongoDB"""
        try:
            cursor = self.captions_collection.find({"video_id": video_id}).sort("timestamp", 1)
            
            captions = []
            for doc in cursor:
                doc.pop('_id', None)
                captions.append(doc)
            
            return captions
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get captions by video: {e}")
            return []
    
    def search_similar_captions(self, query_embedding: np.ndarray, 
                              top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar captions using FAISS embeddings"""
        try:
            if not self.embeddings:
                self.logger.warning("No embeddings available for search")
                return []
            
            # Compute cosine similarities
            similarities = []
            for i, embedding in enumerate(self.embeddings):
                # Compute cosine similarity
                similarity = np.dot(query_embedding, embedding)
                similarities.append((i, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top results
            results = []
            for i, similarity in similarities[:top_k]:
                metadata = self.vector_metadata[i]
                caption_data = self.get_caption_by_id(metadata['caption_id'])
                if caption_data:
                    caption_data['similarity'] = float(similarity)
                    results.append(caption_data)
            
            self.logger.info(f"🔍 Found {len(results)} similar captions")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Failed to search similar captions: {e}")
            return []
    
    def log_rejected_caption(self, raw_caption: str, sanitized_caption: str, 
                           reason: str):
        """Log rejected caption for auditing in MongoDB"""
        try:
            audit_doc = {
                "raw_caption": raw_caption,
                "sanitized_caption": sanitized_caption,
                "rejection_reason": reason,
                "created_at": datetime.now().isoformat()
            }
            
            self.audit_collection.insert_one(audit_doc)
            self.logger.debug(f"📝 Logged rejected caption")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log rejected caption: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics from MongoDB and FAISS"""
        try:
            total_captions = self.captions_collection.count_documents({})
            
            unique_videos = len(self.captions_collection.distinct("video_id"))
            
            rejected_captions = self.audit_collection.count_documents({})
            
            return {
                'total_captions': total_captions,
                'unique_videos': unique_videos,
                'rejected_captions': rejected_captions,
                'vector_embeddings': len(self.embeddings)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get statistics: {e}")
            return {}
    
    def close(self):
        """Close database connections and save vector data"""
        try:
            # Check if Python is shutting down
            import sys
            if sys.meta_path is None:
                return  # Python is shutting down, skip cleanup
            
            self._save_vector_data()
            self.logger.info("💾 Caption storage closed and saved")
        except Exception as e:
            # Ignore errors during shutdown
            if "sys.meta_path is None" not in str(e):
                self.logger.error(f"❌ Failed to close storage: {e}")
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            # Check if Python is shutting down
            import sys
            if sys.meta_path is not None:
                self.close()
        except:
            # Silently ignore errors during shutdown
            pass
