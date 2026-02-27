"""
Data Collector Module

Gathers all required data from MongoDB and file system for report generation:
- Events (object detection, behavior analysis)
- Keyframes and their captions
- Face detections and crops
- Video metadata
- Processing statistics
"""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from datetime import timedelta

try:
    from ..database.config import DatabaseManager, get_presigned_url
except ImportError:
    # Fallback for when running directly or in different context
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from backend.database.config import DatabaseManager, get_presigned_url

from .config import ReportConfig

logger = logging.getLogger(__name__)


class DataCollector:
    """
    Collects all data required for report generation from
    MongoDB and the file system.
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """
        Initialize the data collector.
        
        Args:
            config: Report configuration (uses default if None)
        """
        self.config = config or ReportConfig()
        self.db = None
        self.db_manager = None
        self._connect_database()
    
    def _connect_database(self):
        """Connect to MongoDB."""
        if not self.config.use_database:
            logger.warning("Database disabled in config")
            return
        
        try:
            self.db_manager = DatabaseManager()
            self.db = self.db_manager.db
            logger.info("✅ Connected to MongoDB via DatabaseManager")
            
            # Ensure MinIO is connected
            try:
                if self.db_manager.minio_client:
                    logger.info("✅ MinIO client available for Report generation")
            except Exception as e:
                logger.warning(f"⚠️ MinIO client not available: {e}")
                
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self.db = None
    
    def collect_video_metadata(self, video_id: str) -> Dict[str, Any]:
        """
        Collect metadata for a video.
        
        Args:
            video_id: Video identifier
            
        Returns:
            Video metadata dictionary
        """
        metadata = {
            'video_id': video_id,
            'camera_id': 'Unknown',
            'location': 'Not specified',
            'fps': 0,
            'duration': 0,
            'resolution': 'Unknown',
            'upload_time': None,
            'processed_time': None
        }
        
        if self.db is None:
            return metadata
        
        try:
            # Try video_file collection (DetectifAI schema) first, then video_metadata
            video_doc = self.db.video_file.find_one({'video_id': video_id})
            if not video_doc:
                video_doc = self.db.video_metadata.find_one({'video_id': video_id})
            
                if self.db_manager and self.db_manager.minio_client:
                    # Try to generate presigned URL for the video
                    try:
                        # Determine bucket and key
                        bucket = video_doc.get('minio_bucket') or self.db_manager.config.minio_video_bucket
                        key = video_doc.get('minio_object_key')
                        if not key and video_doc.get('video_id'):
                            # Try constructing standard key if not saved
                            key = f"original/{video_doc.get('video_id')}/video.mp4"
                        
                        if key:
                            url = get_presigned_url(
                                self.db_manager.minio_client, 
                                bucket, 
                                key, 
                                expires=timedelta(hours=24) # 24 hour validity for reports
                            )
                            if url:
                                metadata['video_url'] = url
                                logger.info(f"Generated presigned URL for video {video_id}")
                    except Exception as e:
                        logger.warning(f"Failed to generate video URL: {e}")

            if video_doc:
                 # video_file has duration_secs, fps in meta_data; video_metadata has duration, fps directly
                duration = video_doc.get('duration_secs') or video_doc.get('duration', 0)
                fps = video_doc.get('fps') or (video_doc.get('meta_data') or {}).get('fps', 0)
                res = (video_doc.get('meta_data') or {}).get('resolution') or f"{video_doc.get('width', 0)}x{video_doc.get('height', 0)}"
                metadata.update({
                    'camera_id': video_doc.get('camera_id', 'Unknown'),
                    'location': video_doc.get('location', 'Not specified'),
                    'fps': float(fps) if fps else 0,
                    'duration': float(duration) if duration else 0,
                    'resolution': str(res) if res else 'Unknown',
                    'upload_time': video_doc.get('upload_date') or video_doc.get('upload_time'),
                    'processed_time': video_doc.get('processed_time'),
                    'original_filename': video_doc.get('original_filename', video_doc.get('filename', 'Unknown'))
                })
            
        except Exception as e:
            logger.error(f"Error collecting video metadata: {e}")
        
        return metadata
    
    def collect_events(
        self,
        video_id: str,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        event_types: Optional[List[str]] = None,
        min_threat_level: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Collect events for a video within optional time range.
        
        Args:
            video_id: Video identifier
            time_range: Optional (start, end) datetime tuple
            event_types: Optional filter for event types
            min_threat_level: Minimum threat level to include
            
        Returns:
            List of event dictionaries
        """
        events = []
        
        if self.db is None:
            return events
        
        try:
            # Build query - use event collection (DetectifAI schema: event_id, video_id, start_timestamp_ms, event_type)
            query = {'video_id': video_id}
            
            if time_range:
                start_ms = int(time_range[0].timestamp() * 1000) if hasattr(time_range[0], 'timestamp') else 0
                end_ms = int(time_range[1].timestamp() * 1000) if hasattr(time_range[1], 'timestamp') else 0
                query['start_timestamp_ms'] = {'$gte': start_ms, '$lte': end_ms}
            
            if event_types:
                query['event_type'] = {'$in': event_types}
            
            # Query event collection (not canonical_events)
            cursor = self.db.event.find(query).sort('start_timestamp_ms', 1)
            
            threat_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
            min_level = threat_order.get(min_threat_level, 0) if min_threat_level else 0
            
            for doc in cursor:
                start_ms = doc.get('start_timestamp_ms', 0)
                ts = datetime.utcfromtimestamp(start_ms / 1000.0) if start_ms else None
                desc = doc.get('description', '')
                if not desc and doc.get('event_type'):
                    desc = f"Event: {doc.get('event_type', 'unknown')}"
                event = {
                    'event_id': str(doc.get('event_id', doc.get('_id', ''))),
                    'event_type': doc.get('event_type', 'unknown'),
                    'timestamp': ts,
                    'frame_number': 0,
                    'threat_level': 'medium',
                    'confidence': float(doc.get('confidence_score', 0)),
                    'caption': desc,
                    'description': desc,
                    'keyframe_id': None,
                    'keyframe_path': None,
                    'detections': doc.get('bounding_boxes', []) if isinstance(doc.get('bounding_boxes'), list) else doc.get('bounding_boxes', {}).get('detections', []),
                    'metadata': {}
                }
                
                event_level = threat_order.get(event['threat_level'], 2)
                if event_level >= min_level:
                    events.append(event)
            
            logger.info(f"Collected {len(events)} events for video {video_id}")
            
        except Exception as e:
            logger.error(f"Error collecting events: {e}")
        
        return events
    
    def collect_keyframes(
        self,
        video_id: str,
        event_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Collect keyframes and their captions.
        
        Args:
            video_id: Video identifier
            event_ids: Optional list of event IDs to filter by
            
        Returns:
            List of keyframe dictionaries
        """
        keyframes = []
        
        if self.db is None:
            return keyframes
        
        try:
            query = {'video_id': video_id}
            
            if event_ids:
                query['event_id'] = {'$in': event_ids}
            
            # Sort by frame_number or created_at (DetectifAI keyframes use frame_number, created_at)
            cursor = self.db.keyframes.find(query).sort('frame_number', 1)
            for doc in cursor:
                ts = doc.get('timestamp') or doc.get('created_at')
                if isinstance(ts, (int, float)):
                    ts = datetime.utcfromtimestamp(ts)
                frame_num = doc.get('frame_number') or doc.get('frame_index', 0)
                image_path = doc.get('image_path') or doc.get('minio_path', '')
                # Generate MinIO URL if available
                image_url = None
                if self.db_manager and self.db_manager.minio_client:
                    minio_path = doc.get('minio_path')
                    bucket = doc.get('bucket') or doc.get('minio_bucket') or 'detectifai-keyframes'
                    
                    if minio_path:
                        image_url = get_presigned_url(
                            self.db_manager.minio_client,
                            bucket,
                            minio_path,
                            expires=timedelta(hours=24)
                        )
                
                keyframe = {
                    'keyframe_id': str(doc.get('_id', doc.get('keyframe_id', ''))),
                    'video_id': doc.get('video_id'),
                    'timestamp': ts,
                    'frame_number': int(frame_num) if frame_num is not None else 0,
                    'caption': doc.get('caption', ''),
                    'image_path': image_path,
                    'image_url': image_url,  # Add URL
                    'bucket': bucket,
                    'minio_path': doc.get('minio_path'),
                    'event_id': doc.get('event_id'),
                    'detections': doc.get('objects_detected', doc.get('detections', []))
                }
                keyframes.append(keyframe)
            
            logger.info(f"Collected {len(keyframes)} keyframes for video {video_id}")
            
        except Exception as e:
            logger.error(f"Error collecting keyframes: {e}")
        
        return keyframes
    
    def collect_face_detections(
        self,
        video_id: str,
        include_crops: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Collect face detections and optionally their crop paths.
        
        Args:
            video_id: Video identifier
            include_crops: Whether to include crop file paths
            
        Returns:
            List of face detection dictionaries
        """
        faces = []
        
        if self.db is None:
            return faces
        
        try:
            query = {'video_id': video_id}
            # DetectifAI uses detected_faces collection
            coll = self.db.detected_faces
            cursor = coll.find(query).sort('timestamp', 1)
            
            for doc in cursor:
                # Generate MinIO URL for face crop if available
                crop_url = None
                minio_path = doc.get('minio_object_key') or doc.get('face_image_path')
                # If face_image_path is a path but not a minio key, we might need to be careful
                # DetectifAI usually stores minio path in face_image_path if uploaded
                
                if self.db_manager and self.db_manager.minio_client and minio_path and not os.path.isabs(minio_path):
                     bucket = doc.get('minio_bucket') or 'detectifai-keyframes' # Faces often in keyframes bucket in subdir
                     crop_url = get_presigned_url(
                        self.db_manager.minio_client,
                        bucket,
                        minio_path,
                        expires=timedelta(hours=24)
                     )

                face = {
                    'face_id': str(doc.get('_id', doc.get('face_id', ''))),
                    'video_id': doc.get('video_id'),
                    'timestamp': datetime.utcfromtimestamp(doc.get('timestamp')) if isinstance(doc.get('timestamp'), (int, float)) else doc.get('timestamp'),
                    'frame_number': doc.get('frame_number', 0),
                    'confidence': doc.get('confidence', 0),
                    'bbox': doc.get('bbox', {}),
                    'person_id': doc.get('person_id'),
                    'crop_path': doc.get('crop_path', '') if include_crops else None,
                    'minio_path': minio_path,
                    'crop_url': crop_url
                }
                faces.append(face)
            
            logger.info(f"Collected {len(faces)} face detections for video {video_id}")
            
        except Exception as e:
            logger.error(f"Error collecting face detections: {e}")
        
        return faces
    
    def collect_captions(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Collect video captions from the captioning module.
        
        Args:
            video_id: Video identifier
            
        Returns:
            List of caption dictionaries
        """
        captions = []
        
        if self.db is None:
            return captions
        
        try:
            # Try video_captions collection
            cursor = self.db.video_captions.find({'video_id': video_id}).sort('timestamp', 1)
            
            for doc in cursor:
                caption = {
                    'caption_id': str(doc.get('_id', '')),
                    'video_id': doc.get('video_id'),
                    'timestamp': datetime.utcfromtimestamp(doc.get('timestamp')) if isinstance(doc.get('timestamp'), (int, float)) else doc.get('timestamp'),
                    'frame_number': doc.get('frame_number', 0),
                    'caption': doc.get('caption', ''),
                    'keyframe_id': doc.get('keyframe_id'),
                    'confidence': doc.get('confidence', 0)
                }
                captions.append(caption)
            
            logger.info(f"Collected {len(captions)} captions for video {video_id}")
            
        except Exception as e:
            logger.error(f"Error collecting captions: {e}")
        
        return captions
    
    def collect_all_report_data(
        self,
        video_id: str,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """
        Collect all data needed for report generation.
        
        Args:
            video_id: Video identifier
            time_range: Optional time range filter
            
        Returns:
            Dictionary with all collected data
        """
        logger.info(f"Collecting all report data for video: {video_id}")
        
        # Collect all data types
        metadata = self.collect_video_metadata(video_id)
        events = self.collect_events(video_id, time_range)
        keyframes = self.collect_keyframes(video_id)
        faces = self.collect_face_detections(video_id)
        captions = self.collect_captions(video_id)
        
        # Merge captions into keyframes where possible
        caption_map = {c.get('keyframe_id'): c.get('caption') for c in captions if c.get('keyframe_id')}
        for kf in keyframes:
            if not kf.get('caption') and kf.get('keyframe_id') in caption_map:
                kf['caption'] = caption_map[kf['keyframe_id']]
        
        # Compute statistics
        threat_levels = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        event_types = {}
        
        for event in events:
            level = event.get('threat_level', 'low')
            threat_levels[level] = threat_levels.get(level, 0) + 1
            
            etype = event.get('event_type', 'unknown')
            event_types[etype] = event_types.get(etype, 0) + 1
        
        # Compute patterns for observations
        patterns = self._compute_patterns(events, faces)
        
        # Determine time range from data if not specified
        if not time_range and events:
            timestamps = [e.get('timestamp') for e in events if e.get('timestamp')]
            if timestamps:
                time_range = (min(timestamps), max(timestamps))
        
        report_data = {
            'video_id': video_id,
            'metadata': metadata,
            'events': events,
            'keyframes': keyframes,
            'faces': faces,
            'captions': captions,
            'statistics': {
                'total_events': len(events),
                'threat_levels': threat_levels,
                'event_types': event_types,
                'total_keyframes': len(keyframes),
                'total_faces': len(faces),
                'duration_minutes': metadata.get('duration', 0) / 60
            },
            'patterns': patterns,
            'time_range': time_range,
            'collection_time': datetime.utcnow()
        }
        
        logger.info(f"Report data collection complete: {len(events)} events, "
                   f"{len(keyframes)} keyframes, {len(faces)} faces")
        
        return report_data
    
    def _compute_patterns(
        self,
        events: List[Dict[str, Any]],
        faces: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute patterns from events and faces for observations.
        
        Args:
            events: List of events
            faces: List of face detections
            
        Returns:
            Dictionary of computed patterns
        """
        patterns = {
            'time_clusters': [],
            'escalation': None,
            'repeated_faces': {},
            'event_correlations': []
        }
        
        # Count face appearances
        face_counts = {}
        for face in faces:
            pid = face.get('person_id') or face.get('face_id', 'unknown')
            face_counts[pid] = face_counts.get(pid, 0) + 1
        
        # Find repeated faces (appearing more than once)
        patterns['repeated_faces'] = {
            fid: count for fid, count in face_counts.items() if count > 1
        }
        
        # Detect time clusters (events within 60 seconds of each other)
        if events:
            clusters = []
            current_cluster = []
            
            sorted_events = sorted(
                [e for e in events if e.get('timestamp')],
                key=lambda x: x['timestamp']
            )
            
            for event in sorted_events:
                if not current_cluster:
                    current_cluster = [event]
                else:
                    time_diff = (event['timestamp'] - current_cluster[-1]['timestamp']).total_seconds()
                    if time_diff <= 60:
                        current_cluster.append(event)
                    else:
                        if len(current_cluster) >= 2:
                            clusters.append({
                                'start': current_cluster[0]['timestamp'],
                                'end': current_cluster[-1]['timestamp'],
                                'event_count': len(current_cluster)
                            })
                        current_cluster = [event]
            
            # Don't forget last cluster
            if len(current_cluster) >= 2:
                clusters.append({
                    'start': current_cluster[0]['timestamp'],
                    'end': current_cluster[-1]['timestamp'],
                    'event_count': len(current_cluster)
                })
            
            patterns['time_clusters'] = clusters
        
        # Detect escalation (increasing threat levels over time)
        if len(events) >= 3:
            threat_order = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
            threat_sequence = [
                threat_order.get(e.get('threat_level', 'low'), 1)
                for e in sorted_events if e.get('threat_level')
            ]
            
            if len(threat_sequence) >= 3:
                # Check if generally increasing
                increasing = sum(1 for i in range(len(threat_sequence)-1) 
                               if threat_sequence[i+1] >= threat_sequence[i])
                
                if increasing / (len(threat_sequence) - 1) > 0.6:
                    patterns['escalation'] = 'increasing'
                elif increasing / (len(threat_sequence) - 1) < 0.4:
                    patterns['escalation'] = 'decreasing'
                else:
                    patterns['escalation'] = 'stable'
        
        return patterns
    
    def get_image_path(
        self,
        image_id: str,
        image_type: str = 'keyframe'
    ) -> Optional[str]:
        """
        Get the file path for an image.
        
        Args:
            image_id: Image identifier
            image_type: 'keyframe' or 'face'
            
        Returns:
            File path or None if not found
        """
        if self.db is None:
            return None
        
        try:
            if image_type == 'keyframe':
                doc = self.db.keyframes.find_one({'keyframe_id': image_id})
                if doc:
                    # Prefer URL if available via minio link
                    if self.db_manager and self.db_manager.minio_client and doc.get('minio_path'):
                         bucket = doc.get('bucket') or doc.get('minio_bucket') or 'detectifai-keyframes'
                         url = get_presigned_url(self.db_manager.minio_client, bucket, doc['minio_path'], timedelta(hours=24))
                         if url:
                             return url
                    return doc.get('image_path')
            elif image_type == 'face':
                doc = self.db.detected_faces.find_one({'face_id': image_id})
                if doc:
                    # Prefer URL
                    minio_path = doc.get('minio_object_key') or doc.get('face_image_path')
                    if self.db_manager and self.db_manager.minio_client and minio_path and not os.path.isabs(minio_path):
                         bucket = doc.get('minio_bucket') or 'detectifai-keyframes'
                         url = get_presigned_url(self.db_manager.minio_client, bucket, minio_path, timedelta(hours=24))
                         if url:
                             return url
                    return doc.get('crop_path')
        except Exception as e:
            logger.error(f"Error getting image path: {e}")
        
        return None
