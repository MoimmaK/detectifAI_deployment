
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime

# Add the directory to sys.path to allow imports
sys.path.append(os.getcwd())

# Mock the database before importing DataCollector if possible, 
# or patch it after.
# We need to mock pymongo.MongoClient inside data_collector.

class TestReportGenerationFixes(unittest.TestCase):
    
    def setUp(self):
        # Patch MongoClient
        self.mongo_patcher = patch('report_generation.data_collector.MongoClient')
        self.mock_mongo = self.mongo_patcher.start()
        
        # Setup mock db
        self.mock_db = MagicMock()
        self.mock_mongo.return_value.get_default_database.return_value = self.mock_db
        
        # Import DataCollector here to ensure patches work or just use sys.modules
        from report_generation.data_collector import DataCollector, ReportConfig
        
        self.config = ReportConfig(use_database=True)
        self.collector = DataCollector(self.config)
        # Manually set db just in case
        self.collector.db = self.mock_db

    def tearDown(self):
        self.mongo_patcher.stop()

    def test_collect_events_with_list_bounding_boxes(self):
        print("Testing collect_events with list bounding_boxes...")
        # Mock event data with bounding_boxes as list
        mock_cursor = [
            {
                'event_id': 'evt1',
                'video_id': 'vid1',
                'start_timestamp_ms': 1672531200000, # 2023-01-01 00:00:00 UTC
                'event_type': 'test_event',
                'threat_level': 'medium',
                'bounding_boxes': [{'box': [0, 0, 10, 10], 'class': 'person'}] # LIST here
            }
        ]
        self.mock_db.event.find.return_value.sort.return_value = mock_cursor
        
        events = self.collector.collect_events('vid1')
        
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['event_id'], 'evt1')
        self.assertIsInstance(events[0]['detections'], list)
        self.assertEqual(len(events[0]['detections']), 1)
        print("✓ collect_events passed")

    def test_collect_faces_with_float_timestamp(self):
        print("Testing collect_face_detections with float timestamp...")
        # Mock face data with float timestamp
        mock_cursor = [
            {
                'face_id': 'face1',
                'video_id': 'vid1',
                'timestamp': 1672531205.5, # Float timestamp
                'confidence': 0.9
            }
        ]
        self.mock_db.detected_faces.find.return_value.sort.return_value = mock_cursor
        
        faces = self.collector.collect_face_detections('vid1')
        
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0]['face_id'], 'face1')
        self.assertIsInstance(faces[0]['timestamp'], datetime)
        print(f"✓ timestamp type: {type(faces[0]['timestamp'])}")
        print("✓ collect_face_detections passed")

    def test_collect_keyframes_with_float_timestamp(self):
        print("Testing collect_keyframes with float timestamp...")
        mock_cursor = [
            {
                'keyframe_id': 'kf1',
                'video_id': 'vid1',
                'timestamp': 1672531210.0,
                'image_path': '/tmp/img.jpg'
            }
        ]
        self.mock_db.keyframes.find.return_value.sort.return_value = mock_cursor
        
        keyframes = self.collector.collect_keyframes('vid1')
        
        self.assertEqual(len(keyframes), 1)
        self.assertIsInstance(keyframes[0]['timestamp'], datetime)
        print("✓ collect_keyframes passed")

if __name__ == '__main__':
    unittest.main()
