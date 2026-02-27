"""
Storage layer for captions and embeddings
"""

import logging
import sqlite3
import json
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
import numpy as np
import pickle
from pathlib import Path

try:
    from .models import CaptionRecord
    from .config import CaptioningConfig
except ImportError:
    from models import CaptionRecord
    from config import CaptioningConfig


class CaptionStorage:
    """Handles storage of captions and embeddings"""
    
    def __init__(self, config: CaptioningConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize databases
        self._init_relational_db()
        self._init_vector_db()
    
    def _init_relational_db(self):
        """Initialize SQLite database for caption metadata"""
        try:
            # Use default path if not specified
            db_path = self.config.db_connection_string or "captions.db"
            
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # Enable dict-like access
            
            # Create captions table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS captions (
                    caption_id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    frame_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    raw_caption TEXT NOT NULL,
                    sanitized_caption TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create indexes separately
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_video_id ON captions(video_id)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_frame_id ON captions(frame_id)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON captions(timestamp)")
            
            # Create audit table for rejected captions
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS caption_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    raw_caption TEXT NOT NULL,
                    sanitized_caption TEXT,
                    rejection_reason TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            self.conn.commit()
            self.logger.info("Relational database initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize relational database: {e}")
            raise
    
    def _init_vector_db(self):
        """Initialize vector database for embeddings"""
        try:
            # Create vector store directory
            vector_path = Path(self.config.vector_db_path or "./vector_store")
            vector_path.mkdir(exist_ok=True)
            
            self.vector_db_path = vector_path
            self.embeddings_file = vector_path / "embeddings.pkl"
            self.metadata_file = vector_path / "metadata.json"
            
            # Load existing data if available
            self._load_vector_data()
            
            self.logger.info("Vector database initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector database: {e}")
            raise
    
    def _load_vector_data(self):
        """Load existing vector data"""
        try:
            if self.embeddings_file.exists() and self.metadata_file.exists():
                # Load embeddings
                with open(self.embeddings_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                
                # Load metadata
                with open(self.metadata_file, 'r') as f:
                    self.vector_metadata = json.load(f)
                
                self.logger.info(f"Loaded {len(self.embeddings)} existing embeddings")
            else:
                self.embeddings = []
                self.vector_metadata = []
                
        except Exception as e:
            self.logger.error(f"Failed to load vector data: {e}")
            self.embeddings = []
            self.vector_metadata = []
    
    def _save_vector_data(self):
        """Save vector data to disk"""
        try:
            # Save embeddings
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
            
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(self.vector_metadata, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save vector data: {e}")
    
    def store_caption_record(self, record: CaptionRecord) -> bool:
        """Store a single caption record"""
        try:
            # Store in relational database
            self.conn.execute("""
                INSERT OR REPLACE INTO captions 
                (caption_id, video_id, frame_id, timestamp, raw_caption, 
                 sanitized_caption, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                record.caption_id,
                record.video_id,
                record.frame_id,
                record.timestamp.isoformat(),
                record.raw_caption,
                record.sanitized_caption,
                record.created_at.isoformat()
            ))
            
            # Store in vector database
            self.embeddings.append(record.embedding)
            self.vector_metadata.append({
                'caption_id': record.caption_id,
                'video_id': record.video_id,
                'frame_id': record.frame_id,
                'timestamp': record.timestamp.isoformat()
            })
            
            self.conn.commit()
            self._save_vector_data()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store caption record: {e}")
            return False
    
    def store_caption_records_batch(self, records: List[CaptionRecord]) -> int:
        """Store multiple caption records"""
        stored_count = 0
        
        try:
            # Prepare data for batch insert
            relational_data = []
            embeddings_batch = []
            metadata_batch = []
            
            for record in records:
                relational_data.append((
                    record.caption_id,
                    record.video_id,
                    record.frame_id,
                    record.timestamp.isoformat(),
                    record.raw_caption,
                    record.sanitized_caption,
                    record.created_at.isoformat()
                ))
                
                embeddings_batch.append(record.embedding)
                metadata_batch.append({
                    'caption_id': record.caption_id,
                    'video_id': record.video_id,
                    'frame_id': record.frame_id,
                    'timestamp': record.timestamp.isoformat()
                })
            
            # Batch insert into relational database
            self.conn.executemany("""
                INSERT OR REPLACE INTO captions 
                (caption_id, video_id, frame_id, timestamp, raw_caption, 
                 sanitized_caption, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, relational_data)
            
            # Batch insert into vector database
            self.embeddings.extend(embeddings_batch)
            self.vector_metadata.extend(metadata_batch)
            
            self.conn.commit()
            self._save_vector_data()
            
            stored_count = len(records)
            self.logger.info(f"Stored {stored_count} caption records")
            
        except Exception as e:
            self.logger.error(f"Failed to store caption records batch: {e}")
        
        return stored_count
    
    def get_caption_by_id(self, caption_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve caption by ID"""
        try:
            cursor = self.conn.execute("""
                SELECT * FROM captions WHERE caption_id = ?
            """, (caption_id,))
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get caption by ID: {e}")
            return None
    
    def get_captions_by_video(self, video_id: str) -> List[Dict[str, Any]]:
        """Retrieve all captions for a video"""
        try:
            cursor = self.conn.execute("""
                SELECT * FROM captions WHERE video_id = ? 
                ORDER BY timestamp
            """, (video_id,))
            
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            self.logger.error(f"Failed to get captions by video: {e}")
            return []
    
    def search_similar_captions(self, query_embedding: np.ndarray, 
                              top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar captions using embeddings"""
        try:
            if not self.embeddings:
                return []
            
            # Compute similarities
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
                    caption_data['similarity'] = similarity
                    results.append(caption_data)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search similar captions: {e}")
            return []
    
    def log_rejected_caption(self, raw_caption: str, sanitized_caption: str, 
                           reason: str):
        """Log rejected caption for auditing"""
        try:
            self.conn.execute("""
                INSERT INTO caption_audit 
                (raw_caption, sanitized_caption, rejection_reason, created_at)
                VALUES (?, ?, ?, ?)
            """, (
                raw_caption,
                sanitized_caption,
                reason,
                datetime.now().isoformat()
            ))
            self.conn.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to log rejected caption: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            cursor = self.conn.execute("SELECT COUNT(*) as total FROM captions")
            total_captions = cursor.fetchone()['total']
            
            cursor = self.conn.execute("""
                SELECT COUNT(DISTINCT video_id) as unique_videos FROM captions
            """)
            unique_videos = cursor.fetchone()['unique_videos']
            
            cursor = self.conn.execute("SELECT COUNT(*) as rejected FROM caption_audit")
            rejected_captions = cursor.fetchone()['rejected']
            
            return {
                'total_captions': total_captions,
                'unique_videos': unique_videos,
                'rejected_captions': rejected_captions,
                'vector_embeddings': len(self.embeddings)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def close(self):
        """Close database connections"""
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
            self._save_vector_data()
        except Exception as e:
            self.logger.error(f"Failed to close storage: {e}")
    
    def __del__(self):
        """Cleanup on destruction"""
        self.close()