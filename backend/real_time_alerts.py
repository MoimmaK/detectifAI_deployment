"""
Real-Time Alert Engine for DetectifAI

This module provides the core alert engine for processing live stream detections
and generating real-time alerts with:
- Threat classification (critical, high, medium, low)
- Suspicious person re-appearance tracking via MinIO face store
- Alert deduplication and cooldown management
- Alert queue for SSE broadcast to frontend clients
- False positive feedback loop for improving accuracy

Alert Types:
- Object Detection: gun, knife, fire
- Behavior Detection: fight, accident, wall_climb  
- Suspicious Person Re-appearance: previously flagged face detected again
"""

import uuid
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


# ========================================
# Alert Enums & Data Models
# ========================================

class AlertSeverity(Enum):
    CRITICAL = "critical"   # Immediate danger: fire, gun
    HIGH = "high"           # Serious threat: knife, fight
    MEDIUM = "medium"       # Suspicious: wall_climb, accident
    LOW = "low"             # Informational: suspicious person re-appearance
    

class AlertType(Enum):
    OBJECT_DETECTION = "object_detection"
    BEHAVIOR_DETECTION = "behavior_detection"
    SUSPICIOUS_PERSON = "suspicious_person"
    

class AlertStatus(Enum):
    PENDING = "pending"         # Awaiting user confirmation
    CONFIRMED = "confirmed"     # User confirmed as real threat
    DISMISSED = "dismissed"     # User dismissed as false positive
    AUTO_EXPIRED = "auto_expired"  # No response within timeout
    

# Threat classification mapping
THREAT_CLASSIFICATION = {
    # Object detections
    "fire": {"severity": AlertSeverity.CRITICAL, "type": AlertType.OBJECT_DETECTION, 
             "display_name": "🔥 Fire Detected", "description": "Fire/flames detected in camera feed",
             "requires_confirmation": True},
    "gun": {"severity": AlertSeverity.CRITICAL, "type": AlertType.OBJECT_DETECTION,
            "display_name": "🔫 Weapon (Gun) Detected", "description": "Firearm detected in camera feed",
            "requires_confirmation": True},
    "knife": {"severity": AlertSeverity.HIGH, "type": AlertType.OBJECT_DETECTION,
              "display_name": "🔪 Weapon (Knife) Detected", "description": "Knife/blade detected in camera feed",
              "requires_confirmation": True},
    
    # Behavior detections
    "fighting": {"severity": AlertSeverity.HIGH, "type": AlertType.BEHAVIOR_DETECTION,
                 "display_name": "👊 Fight Detected", "description": "Physical altercation detected",
                 "requires_confirmation": True},
    "road_accident": {"severity": AlertSeverity.MEDIUM, "type": AlertType.BEHAVIOR_DETECTION,
                      "display_name": "🚗 Accident Detected", "description": "Vehicle/road accident detected",
                      "requires_confirmation": True},
    "wallclimb": {"severity": AlertSeverity.MEDIUM, "type": AlertType.BEHAVIOR_DETECTION,
                  "display_name": "🧗 Wall Climbing Detected", "description": "Unauthorized climbing/trespassing detected",
                  "requires_confirmation": True},
    
    # Suspicious person re-appearance
    "suspicious_reappearance": {"severity": AlertSeverity.LOW, "type": AlertType.SUSPICIOUS_PERSON,
                                "display_name": "👤 Suspicious Person Re-appeared", 
                                "description": "A previously flagged person has been detected again",
                                "requires_confirmation": True},
}


@dataclass
class RealTimeAlert:
    """Single real-time alert with all metadata"""
    alert_id: str
    camera_id: str
    alert_type: str          # From AlertType enum value
    detection_class: str     # e.g., 'fire', 'gun', 'fighting'
    severity: str            # From AlertSeverity enum value  
    display_name: str
    description: str
    confidence: float
    timestamp: float         # Unix timestamp
    timestamp_iso: str       # ISO formatted datetime string
    status: str = "pending"  # From AlertStatus enum value
    
    # Detection details
    bounding_boxes: List[Dict] = field(default_factory=list)
    frame_snapshot_path: Optional[str] = None  # MinIO path to frame snapshot
    frame_snapshot_url: Optional[str] = None   # Presigned URL for frontend
    
    # Suspicious person tracking  
    face_id: Optional[str] = None
    face_match_score: Optional[float] = None
    previous_events: List[str] = field(default_factory=list)  # Previous event IDs involving this person
    
    # User feedback
    confirmed_by: Optional[str] = None
    confirmed_at: Optional[str] = None
    feedback_note: Optional[str] = None
    
    # Linked event in MongoDB
    event_id: Optional[str] = None
    video_id: Optional[str] = None
    
    requires_confirmation: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization and MongoDB storage"""
        data = asdict(self)
        return data
    
    def to_sse_payload(self) -> Dict[str, Any]:
        """Convert to lightweight SSE payload for frontend"""
        return {
            "alert_id": self.alert_id,
            "camera_id": self.camera_id,
            "alert_type": self.alert_type,
            "detection_class": self.detection_class,
            "severity": self.severity,
            "display_name": self.display_name,
            "description": self.description,
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp,
            "timestamp_iso": self.timestamp_iso,
            "status": self.status,
            "bounding_boxes": self.bounding_boxes,
            "frame_snapshot_url": self.frame_snapshot_url,
            "face_id": self.face_id,
            "face_match_score": self.face_match_score,
            "requires_confirmation": self.requires_confirmation,
            "event_id": self.event_id,
        }


# ========================================
# Alert Engine (Singleton)
# ========================================

class RealTimeAlertEngine:
    """
    Central alert engine that processes detections from the live stream pipeline
    and manages the alert lifecycle:
    
    1. Detection comes in from LiveStreamProcessor
    2. Engine classifies threat severity
    3. Checks for suspicious person re-appearance
    4. Deduplicates against recent alerts (cooldown)
    5. Stores snapshot frame in MinIO
    6. Pushes alert to SSE broadcast queue
    7. Persists alert to MongoDB
    8. Handles user confirmation/dismissal feedback
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern — one alert engine for the whole app"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        # Alert queue for SSE broadcast (thread-safe deque)
        self._alert_queue: deque = deque(maxlen=500)
        self._alert_subscribers: List[Any] = []  # SSE subscriber queues
        self._subscriber_lock = threading.Lock()
        
        # Active alerts (pending user confirmation)
        self._active_alerts: Dict[str, RealTimeAlert] = {}
        self._alert_history: deque = deque(maxlen=1000)
        
        # Cooldown tracking to prevent duplicate alerts
        # Key: (camera_id, detection_class), Value: last_alert_timestamp
        self._cooldown_tracker: Dict[Tuple[str, str], float] = {}
        self._cooldown_seconds = {
            AlertSeverity.CRITICAL.value: 10,   # 10s cooldown for critical (fire, gun)
            AlertSeverity.HIGH.value: 15,        # 15s for high
            AlertSeverity.MEDIUM.value: 20,      # 20s for medium
            AlertSeverity.LOW.value: 30,         # 30s for low
        }
        
        # Suspicious person tracking
        self._flagged_faces: Dict[str, Dict] = {}  # face_id -> metadata
        
        # Database connections (lazy loaded)
        self._db_manager = None
        self._minio_client = None
        
        # Statistics
        self.stats = {
            "total_alerts": 0,
            "confirmed_alerts": 0,
            "dismissed_alerts": 0,
            "pending_alerts": 0,
            "alerts_by_type": {},
            "alerts_by_severity": {},
        }
        
        logger.info("✅ Real-Time Alert Engine initialized")
    
    @property
    def db_manager(self):
        """Lazy-load database manager"""
        if self._db_manager is None:
            from database.config import DatabaseManager
            self._db_manager = DatabaseManager()
        return self._db_manager
    
    @property
    def alerts_collection(self):
        """Get MongoDB alerts collection"""
        return self.db_manager.db.real_time_alerts
    
    @property
    def minio_client(self):
        """Lazy-load MinIO client"""
        if self._minio_client is None:
            self._minio_client = self.db_manager.minio_client
        return self._minio_client
    
    # ========================================
    # SSE Subscription Management
    # ========================================
    
    def subscribe(self):
        """
        Create a new SSE subscriber queue.
        Returns a queue that the SSE endpoint will read from.
        """
        import queue
        q = queue.Queue(maxsize=100)
        with self._subscriber_lock:
            self._alert_subscribers.append(q)
        logger.info(f"📡 New SSE subscriber connected (total: {len(self._alert_subscribers)})")
        return q
    
    def unsubscribe(self, q):
        """Remove an SSE subscriber queue"""
        with self._subscriber_lock:
            if q in self._alert_subscribers:
                self._alert_subscribers.remove(q)
        logger.info(f"📡 SSE subscriber disconnected (total: {len(self._alert_subscribers)})")
    
    def _broadcast_alert(self, alert: RealTimeAlert):
        """Push alert to all SSE subscribers"""
        payload = alert.to_sse_payload()
        dead_subscribers = []
        
        with self._subscriber_lock:
            for q in self._alert_subscribers:
                try:
                    q.put_nowait(payload)
                except Exception:
                    dead_subscribers.append(q)
            
            # Clean up dead subscribers
            for q in dead_subscribers:
                self._alert_subscribers.remove(q)
    
    def _broadcast_update(self, alert_id: str, update_data: Dict):
        """Broadcast alert status update to all subscribers"""
        payload = {"type": "alert_update", "alert_id": alert_id, **update_data}
        dead_subscribers = []
        
        with self._subscriber_lock:
            for q in self._alert_subscribers:
                try:
                    q.put_nowait(payload)
                except Exception:
                    dead_subscribers.append(q)
            
            for q in dead_subscribers:
                self._alert_subscribers.remove(q)
    
    # ========================================
    # Core Alert Processing
    # ========================================
    
    def process_detection(
        self,
        camera_id: str,
        detection_class: str,
        confidence: float,
        bounding_boxes: List[Dict] = None,
        frame: Any = None,
        timestamp: float = None,
        face_id: str = None,
        face_match_score: float = None,
        video_id: str = None,
    ) -> Optional[RealTimeAlert]:
        """
        Process a detection from the live stream and potentially create an alert.
        
        Args:
            camera_id: Camera identifier
            detection_class: Type of detection (e.g., 'fire', 'gun', 'fighting')
            confidence: Detection confidence (0.0 - 1.0)
            bounding_boxes: List of bounding box dicts
            frame: OpenCV frame (numpy array) for snapshot
            timestamp: Detection timestamp
            face_id: Face ID if facial recognition matched
            face_match_score: Face match similarity score
            video_id: Associated video ID
            
        Returns:
            RealTimeAlert if alert was created, None if suppressed by cooldown
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Normalize detection class
        detection_key = detection_class.lower().strip()
        
        # Look up threat classification
        threat_info = THREAT_CLASSIFICATION.get(detection_key)
        if threat_info is None:
            logger.debug(f"Unknown detection class '{detection_key}', skipping alert")
            return None
        
        # Check cooldown
        if self._is_on_cooldown(camera_id, detection_key, threat_info["severity"].value):
            logger.debug(f"Alert suppressed (cooldown): {detection_key} on {camera_id}")
            return None
        
        # Check confidence threshold
        min_confidence = self._get_min_confidence(detection_key)
        if confidence < min_confidence:
            logger.debug(f"Alert suppressed (low confidence {confidence:.2f} < {min_confidence}): {detection_key}")
            return None
        
        # Create alert
        now = datetime.utcnow()
        alert = RealTimeAlert(
            alert_id=f"alert_{uuid.uuid4().hex[:12]}",
            camera_id=camera_id,
            alert_type=threat_info["type"].value,
            detection_class=detection_key,
            severity=threat_info["severity"].value,
            display_name=threat_info["display_name"],
            description=threat_info["description"],
            confidence=float(confidence),
            timestamp=timestamp,
            timestamp_iso=now.isoformat() + "Z",
            status=AlertStatus.PENDING.value,
            bounding_boxes=bounding_boxes or [],
            requires_confirmation=threat_info["requires_confirmation"],
            video_id=video_id or f"live_{camera_id}",
            face_id=face_id,
            face_match_score=float(face_match_score) if face_match_score else None,
        )
        
        # Save frame snapshot to MinIO
        if frame is not None:
            snapshot_path = self._save_frame_snapshot(camera_id, alert.alert_id, frame)
            if snapshot_path:
                alert.frame_snapshot_path = snapshot_path
                alert.frame_snapshot_url = self._get_snapshot_url(snapshot_path)
        
        # Check suspicious person re-appearance
        if face_id and face_match_score:
            previous = self._check_suspicious_person(face_id)
            if previous:
                alert.previous_events = previous.get("event_ids", [])
                # Upgrade alert info for re-appearance
                alert.description = (
                    f"{threat_info['description']}. "
                    f"⚠️ This person was previously involved in {len(previous.get('event_ids', []))} incident(s)."
                )
        
        # Store in active alerts and history
        self._active_alerts[alert.alert_id] = alert
        self._alert_history.appendleft(alert)
        
        # Update cooldown
        self._cooldown_tracker[(camera_id, detection_key)] = timestamp
        
        # Update stats
        self.stats["total_alerts"] += 1
        self.stats["pending_alerts"] += 1
        self.stats["alerts_by_type"][detection_key] = self.stats["alerts_by_type"].get(detection_key, 0) + 1
        self.stats["alerts_by_severity"][alert.severity] = self.stats["alerts_by_severity"].get(alert.severity, 0) + 1
        
        # Persist to MongoDB (async)
        threading.Thread(target=self._persist_alert, args=(alert,), daemon=True).start()
        
        # Broadcast to SSE subscribers
        self._broadcast_alert(alert)
        
        logger.info(
            f"🚨 ALERT: [{alert.severity.upper()}] {alert.display_name} "
            f"(confidence: {confidence:.2f}) on camera {camera_id}"
        )
        
        return alert
    
    def process_suspicious_person(
        self,
        camera_id: str,
        face_id: str,
        face_match_score: float,
        frame: Any = None,
        timestamp: float = None,
        matched_person_info: Dict = None,
    ) -> Optional[RealTimeAlert]:
        """
        Process a suspicious person re-appearance detection.
        Called when facial recognition matches a previously flagged face.
        
        Args:
            camera_id: Camera identifier
            face_id: Matched face ID
            face_match_score: Similarity score (0.0-1.0)
            frame: Current frame
            timestamp: Detection timestamp
            matched_person_info: Previous incident info for this person
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Only alert if we have a meaningful match
        if face_match_score < 0.6:
            return None
        
        # Check cooldown for this specific face
        cooldown_key = (camera_id, f"face_{face_id}")
        last_alert_time = self._cooldown_tracker.get(cooldown_key, 0)
        if (timestamp - last_alert_time) < 60:  # 60s cooldown per face
            return None
        
        # Create alert
        return self.process_detection(
            camera_id=camera_id,
            detection_class="suspicious_reappearance",
            confidence=face_match_score,
            frame=frame,
            timestamp=timestamp,
            face_id=face_id,
            face_match_score=face_match_score,
        )
    
    # ========================================
    # User Feedback (Confirm / Dismiss)
    # ========================================
    
    def confirm_alert(self, alert_id: str, user_id: str = None, note: str = None) -> Optional[Dict]:
        """
        User confirms alert as real threat.
        Updates MongoDB, stats, and broadcasts update.
        """
        alert = self._active_alerts.get(alert_id)
        if not alert:
            # Try loading from DB
            alert = self._load_alert_from_db(alert_id)
            if not alert:
                logger.warning(f"Alert not found: {alert_id}")
                return None
        
        alert.status = AlertStatus.CONFIRMED.value
        alert.confirmed_by = user_id
        alert.confirmed_at = datetime.utcnow().isoformat() + "Z"
        alert.feedback_note = note
        
        # Update stats
        self.stats["confirmed_alerts"] += 1
        self.stats["pending_alerts"] = max(0, self.stats["pending_alerts"] - 1)
        
        # Flag the person as suspicious for future tracking
        if alert.face_id:
            self._flag_suspicious_person(alert.face_id, alert)
        
        # Update in MongoDB
        threading.Thread(
            target=self._update_alert_in_db, 
            args=(alert_id, {
                "status": alert.status,
                "confirmed_by": user_id,
                "confirmed_at": datetime.utcnow(),
                "feedback_note": note,
                "is_verified": True,
                "is_false_positive": False,
            }),
            daemon=True
        ).start()
        
        # Also update the linked event in the event collection
        if alert.event_id:
            threading.Thread(
                target=self._update_linked_event,
                args=(alert.event_id, True, False),
                daemon=True
            ).start()
        
        # Broadcast update
        self._broadcast_update(alert_id, {
            "status": "confirmed",
            "confirmed_by": user_id,
            "confirmed_at": alert.confirmed_at,
        })
        
        logger.info(f"✅ Alert CONFIRMED: {alert_id} ({alert.display_name}) by {user_id}")
        return alert.to_dict()
    
    def dismiss_alert(self, alert_id: str, user_id: str = None, note: str = None) -> Optional[Dict]:
        """
        User dismisses alert as false positive.
        Updates MongoDB, stats, and broadcasts update.
        """
        alert = self._active_alerts.get(alert_id)
        if not alert:
            alert = self._load_alert_from_db(alert_id)
            if not alert:
                logger.warning(f"Alert not found: {alert_id}")
                return None
        
        alert.status = AlertStatus.DISMISSED.value
        alert.confirmed_by = user_id
        alert.confirmed_at = datetime.utcnow().isoformat() + "Z"
        alert.feedback_note = note
        
        # Update stats
        self.stats["dismissed_alerts"] += 1
        self.stats["pending_alerts"] = max(0, self.stats["pending_alerts"] - 1)
        
        # Update in MongoDB
        threading.Thread(
            target=self._update_alert_in_db,
            args=(alert_id, {
                "status": alert.status,
                "confirmed_by": user_id,
                "confirmed_at": datetime.utcnow(),
                "feedback_note": note,
                "is_verified": True,
                "is_false_positive": True,
            }),
            daemon=True
        ).start()
        
        # Also mark linked event as false positive
        if alert.event_id:
            threading.Thread(
                target=self._update_linked_event,
                args=(alert.event_id, True, True),
                daemon=True
            ).start()
        
        # Broadcast update
        self._broadcast_update(alert_id, {
            "status": "dismissed",
            "confirmed_by": user_id,
            "confirmed_at": alert.confirmed_at,
        })
        
        logger.info(f"❌ Alert DISMISSED: {alert_id} ({alert.display_name}) by {user_id}")
        return alert.to_dict()
    
    # ========================================
    # Alert Queries
    # ========================================
    
    def get_active_alerts(self, camera_id: str = None) -> List[Dict]:
        """Get all pending (unconfirmed) alerts, optionally filtered by camera"""
        alerts = []
        for alert in self._active_alerts.values():
            if alert.status == AlertStatus.PENDING.value:
                if camera_id is None or alert.camera_id == camera_id:
                    alerts.append(alert.to_sse_payload())
        return sorted(alerts, key=lambda a: a["timestamp"], reverse=True)
    
    def get_alert_history(self, limit: int = 50, camera_id: str = None, 
                          severity: str = None, status: str = None) -> List[Dict]:
        """Get alert history with optional filters"""
        alerts = []
        for alert in self._alert_history:
            if camera_id and alert.camera_id != camera_id:
                continue
            if severity and alert.severity != severity:
                continue
            if status and alert.status != status:
                continue
            alerts.append(alert.to_dict())
            if len(alerts) >= limit:
                break
        return alerts
    
    def get_alert_by_id(self, alert_id: str) -> Optional[Dict]:
        """Get a single alert by ID"""
        alert = self._active_alerts.get(alert_id)
        if alert:
            return alert.to_dict()
        # Try DB
        loaded = self._load_alert_from_db(alert_id)
        if loaded:
            return loaded.to_dict()
        return None
    
    def get_stats(self) -> Dict:
        """Get alert statistics"""
        return {
            **self.stats,
            "active_subscribers": len(self._alert_subscribers),
            "active_pending_count": sum(
                1 for a in self._active_alerts.values() 
                if a.status == AlertStatus.PENDING.value
            ),
        }
    
    # ========================================
    # Suspicious Person Tracking
    # ========================================
    
    def _flag_suspicious_person(self, face_id: str, alert: RealTimeAlert):
        """Flag a person as suspicious for future re-appearance tracking"""
        if face_id not in self._flagged_faces:
            self._flagged_faces[face_id] = {
                "face_id": face_id,
                "flagged_at": datetime.utcnow().isoformat(),
                "event_ids": [],
                "alert_ids": [],
                "incident_count": 0,
            }
        
        entry = self._flagged_faces[face_id]
        entry["event_ids"].append(alert.event_id or alert.alert_id)
        entry["alert_ids"].append(alert.alert_id)
        entry["incident_count"] += 1
        entry["last_seen"] = datetime.utcnow().isoformat()
        
        # Also persist to MongoDB for cross-session tracking
        threading.Thread(
            target=self._persist_flagged_person, args=(face_id, entry), daemon=True
        ).start()
        
        logger.info(f"🏷️ Person {face_id[:8]}... flagged as suspicious (incidents: {entry['incident_count']})")
    
    def _check_suspicious_person(self, face_id: str) -> Optional[Dict]:
        """Check if a face belongs to a previously flagged person"""
        # Check in-memory cache first
        if face_id in self._flagged_faces:
            return self._flagged_faces[face_id]
        
        # Check MongoDB
        try:
            doc = self.alerts_collection.find_one(
                {"face_id": face_id, "status": "confirmed"},
                sort=[("timestamp", -1)]
            )
            if doc:
                return {
                    "face_id": face_id,
                    "event_ids": [doc.get("event_id", "")],
                    "incident_count": 1,
                }
        except Exception as e:
            logger.warning(f"Error checking suspicious person: {e}")
        
        return None
    
    def _persist_flagged_person(self, face_id: str, entry: Dict):
        """Persist flagged person to MongoDB"""
        try:
            self.db_manager.db.flagged_persons.update_one(
                {"face_id": face_id},
                {"$set": entry, "$setOnInsert": {"created_at": datetime.utcnow()}},
                upsert=True
            )
        except Exception as e:
            logger.error(f"Error persisting flagged person: {e}")
    
    def load_flagged_persons(self):
        """Load flagged persons from MongoDB on startup"""
        try:
            docs = self.db_manager.db.flagged_persons.find({})
            for doc in docs:
                face_id = doc.get("face_id")
                if face_id:
                    self._flagged_faces[face_id] = {
                        "face_id": face_id,
                        "flagged_at": doc.get("flagged_at", ""),
                        "event_ids": doc.get("event_ids", []),
                        "alert_ids": doc.get("alert_ids", []),
                        "incident_count": doc.get("incident_count", 0),
                        "last_seen": doc.get("last_seen", ""),
                    }
            logger.info(f"📋 Loaded {len(self._flagged_faces)} flagged persons from database")
        except Exception as e:
            logger.warning(f"Could not load flagged persons: {e}")
    
    # ========================================
    # Internal Helpers
    # ========================================
    
    def _is_on_cooldown(self, camera_id: str, detection_class: str, severity: str) -> bool:
        """Check if a detection is within cooldown period"""
        key = (camera_id, detection_class)
        last_time = self._cooldown_tracker.get(key, 0)
        cooldown = self._cooldown_seconds.get(severity, 15)
        return (time.time() - last_time) < cooldown
    
    def _get_min_confidence(self, detection_class: str) -> float:
        """Get minimum confidence threshold for alerting"""
        thresholds = {
            "fire": 0.65,
            "gun": 0.60,
            "knife": 0.60,
            "fighting": 0.55,
            "road_accident": 0.50,
            "wallclimb": 0.50,
            "suspicious_reappearance": 0.55,
        }
        return thresholds.get(detection_class, 0.50)
    
    def _save_frame_snapshot(self, camera_id: str, alert_id: str, frame) -> Optional[str]:
        """Save alert frame snapshot to MinIO"""
        try:
            import cv2
            from io import BytesIO
            
            # Encode frame
            is_success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not is_success:
                return None
            
            frame_bytes = buffer.tobytes()
            timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            object_name = f"alerts/{camera_id}/{alert_id}_{timestamp_str}.jpg"
            bucket = self.db_manager.config.minio_keyframe_bucket
            
            frame_buffer = BytesIO(frame_bytes)
            self.minio_client.put_object(
                bucket,
                object_name,
                frame_buffer,
                length=len(frame_bytes),
                content_type="image/jpeg",
                metadata={"alert_id": alert_id, "camera_id": camera_id}
            )
            
            return f"{bucket}/{object_name}"
            
        except Exception as e:
            logger.warning(f"Failed to save alert snapshot: {e}")
            return None
    
    def _get_snapshot_url(self, snapshot_path: str) -> Optional[str]:
        """Generate presigned URL for alert snapshot"""
        try:
            parts = snapshot_path.split("/", 1)
            if len(parts) != 2:
                return None
            bucket, object_name = parts
            url = self.minio_client.presigned_get_object(
                bucket, object_name, expires=timedelta(hours=2)
            )
            return url
        except Exception as e:
            logger.warning(f"Failed to generate snapshot URL: {e}")
            return None
    
    def _persist_alert(self, alert: RealTimeAlert):
        """Persist alert to MongoDB"""
        try:
            doc = alert.to_dict()
            doc["created_at"] = datetime.utcnow()
            self.alerts_collection.insert_one(doc)
            logger.debug(f"Persisted alert to MongoDB: {alert.alert_id}")
        except Exception as e:
            logger.error(f"Failed to persist alert: {e}")
    
    def _update_alert_in_db(self, alert_id: str, update_data: Dict):
        """Update alert in MongoDB"""
        try:
            update_data["updated_at"] = datetime.utcnow()
            self.alerts_collection.update_one(
                {"alert_id": alert_id},
                {"$set": update_data}
            )
        except Exception as e:
            logger.error(f"Failed to update alert in DB: {e}")
    
    def _update_linked_event(self, event_id: str, is_verified: bool, is_false_positive: bool):
        """Update the linked event in the main event collection"""
        try:
            self.db_manager.db.event.update_one(
                {"event_id": event_id},
                {"$set": {
                    "is_verified": is_verified,
                    "is_false_positive": is_false_positive,
                    "verified_at": datetime.utcnow(),
                }}
            )
        except Exception as e:
            logger.error(f"Failed to update linked event: {e}")
    
    def _load_alert_from_db(self, alert_id: str) -> Optional[RealTimeAlert]:
        """Load alert from MongoDB"""
        try:
            doc = self.alerts_collection.find_one({"alert_id": alert_id})
            if doc:
                # Remove MongoDB _id field
                doc.pop("_id", None)
                doc.pop("created_at", None)
                doc.pop("updated_at", None)
                return RealTimeAlert(**{k: v for k, v in doc.items() if k in RealTimeAlert.__dataclass_fields__})
        except Exception as e:
            logger.error(f"Failed to load alert from DB: {e}")
        return None


# ========================================
# Module-level convenience functions
# ========================================

def get_alert_engine() -> RealTimeAlertEngine:
    """Get the singleton alert engine instance"""
    return RealTimeAlertEngine()


def process_live_detection(camera_id: str, detection_class: str, confidence: float, 
                           frame=None, **kwargs) -> Optional[RealTimeAlert]:
    """Convenience function to process a detection and potentially generate an alert"""
    engine = get_alert_engine()
    return engine.process_detection(
        camera_id=camera_id,
        detection_class=detection_class,
        confidence=confidence,
        frame=frame,
        **kwargs
    )
