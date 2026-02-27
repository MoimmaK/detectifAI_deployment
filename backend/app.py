"""
DetectifAI Flask Backend - AI-Powered CCTV Surveillance System

Enhanced Flask API for:
- Video upload and processing with DetectifAI security focus
- Real-time processing status and results
- Object detection with fire/weapon recognition
- Security event analysis and threat assessment
- Frontend integration for surveillance dashboard
- Automated forensic report generation
"""

from flask import Flask, request, jsonify, send_file, send_from_directory, Response, redirect
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import threading
import json
from datetime import datetime, timedelta
import logging
import uuid
import time
import urllib.parse
from typing import List, Dict, Any

# Import DetectifAI components
from main_pipeline import CompleteVideoProcessingPipeline
from config import get_security_focused_config, VideoProcessingConfig

# Import Report Generation components
try:
    from report_generation import ReportGenerator, ReportConfig
    REPORT_GENERATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Report generation not available: {e}")
    REPORT_GENERATION_AVAILABLE = False
    ReportGenerator = None
    ReportConfig = None

# Import database-integrated service
from database_video_service import DatabaseIntegratedVideoService

# Try to import DetectifAI-specific components
try:
    from detectifai_events import DetectifAIEventType, ThreatLevel
    DETECTIFAI_EVENTS_AVAILABLE = True
except ImportError:
    DETECTIFAI_EVENTS_AVAILABLE = False
    logging.warning("DetectifAI events module not available - using basic functionality")

# Try to import caption search (optional - may not be available)
try:
    import sys
    import os
    # Add DetectifAI_db to path for imports
    detectifai_db_path = os.path.join(os.path.dirname(__file__), 'DetectifAI_db')
    if detectifai_db_path not in sys.path:
        sys.path.insert(0, detectifai_db_path)
    from caption_search import get_caption_search_engine
    CAPTION_SEARCH_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Caption search not available: {e}")
    CAPTION_SEARCH_AVAILABLE = False
    get_caption_search_engine = None

# Import subscription middleware for feature gating
try:
    from subscription_middleware import (
        SubscriptionMiddleware,
        require_subscription,
        require_feature,
        check_usage_limit
    )
    SUBSCRIPTION_MIDDLEWARE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Subscription middleware not available: {e}")
    SUBSCRIPTION_MIDDLEWARE_AVAILABLE = False
    # Create dummy decorators that do nothing
    def require_subscription(plan=None):
        def decorator(f):
            return f
        return decorator
    def require_feature(feature):
        def decorator(f):
            return f
        return decorator
    def check_usage_limit(limit_type, auto_increment=True):
        def decorator(f):
            return f
        return decorator

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/detectifai_api.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration - use absolute paths to handle different working directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'video_processing_outputs')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Store processing status in memory (use Redis in production)
processing_status = {}

# Initialize database-integrated video service
try:
    db_video_service = DatabaseIntegratedVideoService(get_security_focused_config())
    DATABASE_ENABLED = True
    # Initialize DETECTIFAI_DB for subscription middleware
    app.config['DETECTIFAI_DB'] = db_video_service.db_manager.db
    logger.info("✅ Database-integrated video service initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize database service: {e}")
    DATABASE_ENABLED = False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_detectifai_results(pipeline_results):
    """Extract DetectifAI-specific results from pipeline output"""
    try:
        detectifai_results = {
            # Basic video metrics
            'video_info': {
                'total_keyframes': pipeline_results['outputs'].get('total_keyframes', 0),
                'processing_time': pipeline_results['processing_stats'].get('total_processing_time', 0),
                'output_directory': pipeline_results['outputs'].get('output_directory', '')
            },
            
            # Security detection results
            'security_detection': {
                'total_object_detections': pipeline_results['outputs'].get('total_object_detections', 0),
                'total_object_events': pipeline_results['outputs'].get('total_object_events', 0),
                'detectifai_events': pipeline_results['outputs'].get('detectifai_events', 0),
                'fire_detections': 0,  # Will be populated from actual results
                'weapon_detections': 0,
                'security_alerts': []
            },
            
            # Event analysis
            'event_analysis': {
                'canonical_events': pipeline_results['outputs'].get('canonical_events', 0),
                'total_motion_events': pipeline_results['outputs'].get('total_motion_events', 0),
                'high_priority_events': 0,
                'critical_events': 0
            },
            
            # Output files
            'output_files': {
                'keyframes_directory': os.path.join(pipeline_results['outputs'].get('output_directory', ''), 'frames'),
                'reports': pipeline_results['outputs'].get('reports', {}),
                'highlight_reels': pipeline_results['outputs'].get('highlight_reels', {}),
                'compressed_video': pipeline_results['outputs'].get('compressed_video', '')
            },
            
            # System performance
            'performance': {
                'frames_processed': pipeline_results['processing_stats'].get('frames_processed', 0),
                'frames_enhanced': pipeline_results['processing_stats'].get('frames_enhanced', 0),
                'gpu_acceleration': pipeline_results['processing_stats'].get('gpu_used', False)
            }
        }
        
        return detectifai_results
        
    except Exception as e:
        logger.error(f"Error extracting DetectifAI results: {e}")
        return {'error': 'Failed to extract results'}

def process_video_async(video_id, video_path, config_type='detectifai'):
    """Process video in background thread with DetectifAI focus"""
    try:
        processing_status[video_id]['status'] = 'processing'
        processing_status[video_id]['progress'] = 0
        processing_status[video_id]['message'] = 'Initializing DetectifAI processing...'
        
        # Select configuration with DetectifAI optimizations
        if config_type == 'detectifai' or config_type == 'security':
            config = get_security_focused_config()
        # Removed robbery detection - using security focused config as default
        elif config_type == 'high_recall':
            try:
                from config import get_high_recall_config
                config = get_high_recall_config()
            except ImportError:
                config = get_security_focused_config()
        elif config_type == 'balanced':
            try:
                from config import get_balanced_config
                config = get_balanced_config()
            except ImportError:
                config = VideoProcessingConfig()
        else:
            config = VideoProcessingConfig()
        
        # DetectifAI-specific configuration enhancements
        config.enable_object_detection = True
        config.enable_facial_recognition = True
        config.enable_video_captioning = True # Re-enabled with improved error handling and timeouts
        config.keyframe_extraction_fps = 1.0  # Extract 1 frame per second for surveillance
        config.enable_adaptive_processing = True
        
        # Set custom output directory for this video
        config.output_base_dir = os.path.join(OUTPUT_FOLDER, video_id)
        
        # Initialize pipeline with database manager for MongoDB integration
        db_manager = None
        if DATABASE_ENABLED:
            db_manager = db_video_service.db_manager
        
        pipeline = CompleteVideoProcessingPipeline(config, db_manager=db_manager)
        
        # Update progress
        processing_status[video_id]['progress'] = 10
        processing_status[video_id]['message'] = 'Extracting keyframes for security analysis...'
        
        # Process video with DetectifAI (with error tolerance)
        output_name = os.path.splitext(os.path.basename(video_path))[0]
        results = None
        processing_errors = []
        
        try:
            results = pipeline.process_video_complete(video_path, output_name)
            logger.info(f"✅ Core pipeline processing completed for {video_id}")
        except Exception as pipeline_error:
            logger.error(f"⚠️ Pipeline error (but continuing): {str(pipeline_error)}")
            processing_errors.append(f"Pipeline: {str(pipeline_error)}")
            # Create minimal results structure
            results = {
                'outputs': {
                    'total_keyframes': 0,
                    'total_events': 0,
                    'total_motion_events': 0,
                    'total_object_events': 0,
                    'total_object_detections': 0,
                    'canonical_events': [],
                    'total_segments': 1,
                    'highlight_reels': {},
                    'reports': {},
                    'compressed_video': ''
                },
                'processing_stats': {'total_processing_time': 0}
            }
        
        # Extract DetectifAI-specific results (with error tolerance)
        detectifai_results = {}
        try:
            detectifai_results = extract_detectifai_results(results)
        except Exception as extract_error:
            logger.error(f"⚠️ Result extraction error (but continuing): {str(extract_error)}")
            processing_errors.append(f"Extraction: {str(extract_error)}")
            detectifai_results = {'security_detection': {}, 'event_analysis': {}, 'performance': {}}
        
        # Always mark as completed (even with errors)
        processing_status[video_id]['status'] = 'completed'
        processing_status[video_id]['progress'] = 100
        completion_message = 'DetectifAI processing completed successfully!'
        if processing_errors:
            completion_message = f'DetectifAI processing completed with warnings: {len(processing_errors)} non-critical errors'
        processing_status[video_id]['message'] = completion_message
        processing_status[video_id]['results'] = {
            # Original results for backward compatibility
            'total_keyframes': results['outputs']['total_keyframes'],
            'total_events': results['outputs']['total_events'],
            'total_motion_events': results['outputs'].get('total_motion_events', 0),
            'total_object_events': results['outputs'].get('total_object_events', 0),
            'total_object_detections': results['outputs'].get('total_object_detections', 0),
            'canonical_events': results['outputs']['canonical_events'],
            'total_segments': results['outputs']['total_segments'],
            'processing_time': results['processing_stats']['total_processing_time'],
            'highlight_reels': results['outputs'].get('highlight_reels', {}),
            'reports': results['outputs'].get('reports', {}),
            'compressed_video': results['outputs'].get('compressed_video', ''),
            'output_directory': config.output_base_dir,
            'object_detection_enabled': config.enable_object_detection,
            
            # DetectifAI-specific results
            'detectifai_results': detectifai_results,
            'security_detection': detectifai_results.get('security_detection', {}),
            'event_analysis': detectifai_results.get('event_analysis', {}),
            'performance': detectifai_results.get('performance', {}),
            
            # Processing status
            'processing_errors': processing_errors,
            'has_warnings': len(processing_errors) > 0
        }
        
        logger.info(f"Video {video_id} processed successfully")
        
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}")
        processing_status[video_id]['status'] = 'failed'
        processing_status[video_id]['message'] = f'Error: {str(e)}'
        processing_status[video_id]['error'] = str(e)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'timestamp': datetime.now().isoformat(),
        'database_enabled': DATABASE_ENABLED
    })

# ====== SUBSCRIPTION & FEATURE GATING ENDPOINTS ======

@app.route('/api/feature/check', methods=['GET'])
def check_feature_access():
    """
    Check if user has access to specific feature based on subscription plan.
    Used by frontend to determine feature visibility.
    """
    try:
        user_id = request.args.get('user_id')
        feature = request.args.get('feature')
        
        if not user_id or not feature:
            return jsonify({
                'success': False,
                'error': 'user_id and feature required'
            }), 400
        
        if not SUBSCRIPTION_MIDDLEWARE_AVAILABLE:
            # If middleware not available, allow all (dev mode)
            return jsonify({
                'success': True,
                'feature': feature,
                'has_access': True,
                'current_plan': 'dev_mode',
                'message': 'Subscription middleware not available - all features enabled'
            }), 200
        
        db = app.config.get('DETECTIFAI_DB')
        middleware = SubscriptionMiddleware(db)
        
        has_access = middleware.check_feature_access(user_id, feature)
        plan_name = middleware.get_user_plan_name(user_id)
        
        return jsonify({
            'success': True,
            'feature': feature,
            'has_access': has_access,
            'current_plan': plan_name
        }), 200
        
    except Exception as e:
        logger.error(f"Error checking feature access: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/usage/summary', methods=['GET'])
def get_usage_summary():
    """
    Get user's current usage statistics and limits based on subscription.
    Returns usage for video processing, searches, etc.
    """
    try:
        user_id = request.args.get('user_id')
        
        if not user_id:
            return jsonify({
                'success': False,
                'error': 'user_id required'
            }), 400
        
        if not SUBSCRIPTION_MIDDLEWARE_AVAILABLE:
            # If middleware not available, return unlimited (dev mode)
            return jsonify({
                'success': True,
                'usage': {
                    'has_subscription': True,
                    'plan': 'dev_mode',
                    'plan_name': 'Development Mode',
                    'status': 'active',
                    'message': 'Subscription middleware not available - unlimited usage'
                }
            }), 200
        
        db = app.config.get('DETECTIFAI_DB')
        middleware = SubscriptionMiddleware(db)
        
        summary = middleware.get_usage_summary(user_id)
        
        return jsonify({
            'success': True,
            'usage': summary
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting usage summary: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/usage/increment', methods=['POST'])
def increment_usage():
    """
    Manually increment usage counter for a user.
    Called after successful operations that should count toward limits.
    """
    try:
        data = request.get_json() or {}
        user_id = data.get('user_id')
        limit_type = data.get('limit_type')
        amount = data.get('amount', 1)
        
        if not user_id or not limit_type:
            return jsonify({
                'success': False,
                'error': 'user_id and limit_type required'
            }), 400
        
        if not SUBSCRIPTION_MIDDLEWARE_AVAILABLE:
            return jsonify({
                'success': True,
                'message': 'Usage tracking not available in dev mode'
            }), 200
        
        db = app.config.get('DETECTIFAI_DB')
        middleware = SubscriptionMiddleware(db)
        
        success = middleware.increment_usage(user_id, limit_type, amount)
        
        return jsonify({
            'success': success,
            'message': 'Usage incremented' if success else 'Failed to increment usage'
        }), 200 if success else 500
        
    except Exception as e:
        logger.error(f"Error incrementing usage: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500



# ====== REPORT GENERATION ENDPOINTS ======

@app.route('/api/video/reports/generate', methods=['POST'])
@require_subscription()
@check_usage_limit('report_generation')
def generate_report():
    """Generate forensic report for a video and upload to MinIO"""
    if not REPORT_GENERATION_AVAILABLE:
        return jsonify({'error': 'Report generation service not available'}), 503
        
    try:
        data = request.get_json()
        video_id = data.get('video_id')
        
        if not video_id:
             return jsonify({'error': 'video_id required'}), 400
             
        # Initialize generator
        config = ReportConfig()
        # Use existing model path or default
        if os.path.exists(os.path.join(BASE_DIR, 'report_generation', 'models', 'qwen2.5-3b-instruct-q4_k_m.gguf')):
             # Config should pick it up automatically if in expected path
             pass
        
        generator = ReportGenerator(config)
        
        # Generate report
        logger.info(f"Generating report for video: {video_id}")
        report = generator.generate_report(video_id=video_id)
        
        # Define report output directory (local temporary storage)
        report_dir = os.path.join(OUTPUT_FOLDER, video_id, 'reports')
        os.makedirs(report_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"report_{timestamp}.pdf"
        html_filename = f"report_{timestamp}.html"
        
        pdf_path = os.path.join(report_dir, pdf_filename)
        html_path = os.path.join(report_dir, html_filename)
        
        # Export HTML (always available)
        final_html_path = generator.export_html(report, output_path=html_path)
        logger.info(f"✅ HTML report exported locally: {final_html_path}")
        
        # Try to export PDF (optional - may fail if WeasyPrint dependencies missing)
        final_pdf_path = None
        try:
            final_pdf_path = generator.export_pdf(report, output_path=pdf_path)
            logger.info(f"✅ PDF report exported locally: {final_pdf_path}")
        except Exception as pdf_error:
            logger.warning(f"⚠️ PDF export failed (HTML report still available): {pdf_error}")
            # Try fallback SimplePDFExporter if available
            try:
                from report_generation.pdf_exporter import SimplePDFExporter
                simple_exporter = SimplePDFExporter(config)
                final_pdf_path = simple_exporter.export(report, output_path=pdf_path)
                logger.info(f"✅ PDF exported using SimplePDFExporter: {final_pdf_path}")
            except Exception as fallback_error:
                logger.warning(f"⚠️ SimplePDFExporter also failed: {fallback_error}")
                # Continue without PDF - HTML is still available
                final_pdf_path = None
        
        # Upload reports to MinIO and get presigned URLs
        html_url = None
        pdf_url = None
        
        try:
            # Initialize ReportRepository
            from database.config import DatabaseManager
            from database.repositories import ReportRepository
            
            db_manager = DatabaseManager()
            report_repo = ReportRepository(db_manager)
            
            # Upload HTML to MinIO
            logger.info(f"📤 Uploading HTML report to MinIO...")
            html_minio_path = report_repo.upload_report_to_minio(final_html_path, video_id, html_filename)
            html_url = report_repo.get_report_presigned_url(video_id, html_filename, expires=timedelta(hours=24))
            logger.info(f"✅ HTML report uploaded to MinIO: {html_minio_path}")
            
            # Upload PDF to MinIO if available
            if final_pdf_path and os.path.exists(final_pdf_path):
                logger.info(f"📤 Uploading PDF report to MinIO...")
                pdf_minio_path = report_repo.upload_report_to_minio(final_pdf_path, video_id, pdf_filename)
                pdf_url = report_repo.get_report_presigned_url(video_id, pdf_filename, expires=timedelta(hours=24))
                logger.info(f"✅ PDF report uploaded to MinIO: {pdf_minio_path}")
            
        except Exception as minio_error:
            logger.error(f"❌ Failed to upload reports to MinIO: {minio_error}")
            # Fall back to local file serving if MinIO upload fails
            html_url = f"/api/video/reports/download/{video_id}/{html_filename}"
            if final_pdf_path:
                pdf_url = f"/api/video/reports/download/{video_id}/{pdf_filename}"
        
        response_data = {
            'success': True,
            'report_id': report.report_id,
            'html_url': html_url,
            'pdf_available': pdf_url is not None
        }
        
        if pdf_url:
            response_data['pdf_url'] = pdf_url
        
        logger.info(f"✅ Report generation complete for {video_id}")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Report generation error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/video/reports/download/<video_id>/<filename>', methods=['GET'])
def download_report(video_id, filename):
    """Download generated report file"""
    try:
        report_dir = os.path.join(OUTPUT_FOLDER, video_id, 'reports')
        return send_from_directory(report_dir, filename, as_attachment=True)
    except Exception as e:
         return jsonify({'error': 'File not found'}), 404


# ====== DATABASE-INTEGRATED ENDPOINTS ======

@app.route('/api/v2/video/upload', methods=['POST'])
@require_subscription()  # Requires any active subscription (Basic or Pro)
@check_usage_limit('video_processing')  # Check and increment video processing limit
def upload_video_db():
    """Enhanced video upload with database storage. Requires: Active subscription"""
    if not DATABASE_ENABLED:
        return jsonify({'error': 'Database service not available'}), 503
    
    try:
        # Check if file is present
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv, wmv, flv'}), 400
        
        # Generate video ID with consistent format
        video_id = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"
        
        # Save temporary file with original extension
        filename = secure_filename(file.filename)
        base, ext = os.path.splitext(filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{video_id}/video{ext}")
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        file.save(temp_path)
        
        # Get user ID (if authenticated) - TODO: implement proper authentication
        user_id = request.form.get('user_id', None)
        
        # STEP 1: Extract video metadata FIRST (before MongoDB record)
        try:
            import cv2
            cap = cv2.VideoCapture(temp_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            file_size = os.path.getsize(temp_path)
            resolution = f"{width}x{height}"
        except Exception as e:
            logger.warning(f"Could not extract video metadata: {e}")
            fps = 30.0
            duration = 0
            file_size = os.path.getsize(temp_path)
            resolution = "unknown"
        
        # STEP 2: Create MongoDB record FIRST (before MinIO upload)
        video_record = {
            "video_id": video_id,
            "user_id": user_id or "system",
            "file_path": f"videos/{video_id}/video{ext}",
            "minio_object_key": f"original/{video_id}/video{ext}",  # Will be confirmed after MinIO upload
            "minio_bucket": db_video_service.video_repo.video_bucket,
            "codec": "h264",  # Default, can be updated later
            "fps": float(fps),
            "upload_date": datetime.utcnow(),
            "duration_secs": int(duration),
            "file_size_bytes": int(file_size),
            "meta_data": {
                "filename": filename,
                "original_name": file.filename,
                "resolution": resolution,
                "processing_status": "uploading",
                "processing_progress": 0,
                "processing_message": "Creating database record..."
            }
        }
        
        # Create MongoDB record immediately
        try:
            video_doc_id = db_video_service.video_repo.create_video_record(video_record)
            logger.info(f"✅ Created MongoDB record for video: {video_id}")
        except Exception as e:
            logger.error(f"❌ Failed to create MongoDB record: {e}")
            return jsonify({'error': f'Failed to create database record: {str(e)}'}), 500
        
        # STEP 3: Upload video to MinIO immediately (after MongoDB record exists)
        try:
            db_video_service.video_repo.update_metadata(video_id, {
                "processing_progress": 5,
                "processing_message": "Uploading video to MinIO..."
            })
            
            minio_path = db_video_service.video_repo.upload_video_to_minio(temp_path, video_id)
            
            # STEP 4: Update MongoDB with MinIO path (link metadata)
            db_video_service.video_repo.collection.update_one(
                {"video_id": video_id},
                {"$set": {
                    "minio_object_key": minio_path,
                    "meta_data.minio_original_path": minio_path
                }}
            )
            logger.info(f"✅ Uploaded video to MinIO and linked in MongoDB: {minio_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to upload to MinIO: {e}")
            db_video_service.video_repo.update_metadata(video_id, {
                "processing_status": "failed",
                "error_message": f"MinIO upload failed: {str(e)}"
            })
            return jsonify({'error': f'Failed to upload to MinIO: {str(e)}'}), 500
        
        # STEP 5: Start background processing (frames, detection, etc.)
        try:
            thread = threading.Thread(
                target=db_video_service.process_video_with_database_storage,
                args=(temp_path, video_id, user_id),
                daemon=True
            )
            thread.start()
            
            return jsonify({
                'success': True,
                'video_id': video_id,
                'message': 'Video uploaded successfully. Processing started with database storage.',
                'status_url': f'/api/v2/video/status/{video_id}'
            }), 201
            
        except Exception as process_error:
            logger.error(f"Failed to start video processing: {process_error}")
            # Update status in database
            db_video_service.video_repo.update_metadata(video_id, {
                "processing_status": "failed",
                "error_message": str(process_error)
            })
            raise
            
    except Exception as e:
        logger.error(f"Database upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    except Exception as e:
        logger.error(f"Database upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v2/video/status/<video_id>', methods=['GET'])
def get_video_status_db(video_id):
    """Get processing status from database with fallback to in-memory status"""
    if not DATABASE_ENABLED:
        # Fallback to in-memory status if database not available
        if video_id in processing_status:
            return jsonify(processing_status[video_id]), 200
        return jsonify({'error': 'Database service not available and video not found in memory'}), 503

    try:
        status_data = db_video_service.get_video_status(video_id)

        if 'error' in status_data:
            # Fallback to in-memory status if database lookup fails
            if video_id in processing_status:
                logger.info(f"Database lookup failed for {video_id}, falling back to in-memory status")
                return jsonify(processing_status[video_id]), 200
            return jsonify(status_data), 404

        return jsonify(status_data), 200

    except Exception as e:
        logger.error(f"Database status check error: {str(e)}")
        # Fallback to in-memory status on exception
        if video_id in processing_status:
            logger.info(f"Database error for {video_id}, falling back to in-memory status")
            return jsonify(processing_status[video_id]), 200
        return jsonify({'error': str(e)}), 500

@app.route('/api/v2/video/keyframes/<video_id>', methods=['GET'])
def get_video_keyframes_db(video_id):
    """Get keyframes from database with MinIO URLs"""
    if not DATABASE_ENABLED:
        return jsonify({'error': 'Database service not available'}), 503
    
    try:
        # Get query parameters
        filter_detections = request.args.get('filter_detections', 'false').lower() == 'true'
        limit = request.args.get('limit', type=int)
        
        keyframes_data = db_video_service.get_video_keyframes(
            video_id, filter_detections=filter_detections, limit=limit
        )
        
        return jsonify(keyframes_data), 200
        
    except Exception as e:
        logger.error(f"Database keyframes retrieval error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v2/video/events/<video_id>', methods=['GET'])
def get_video_events_db(video_id):
    """Get events from database"""
    if not DATABASE_ENABLED:
        return jsonify({'error': 'Database service not available'}), 503
    
    try:
        event_type = request.args.get('type')  # motion, object_detection, face_recognition
        
        events_data = db_video_service.get_video_events(video_id, event_type)
        
        return jsonify(events_data), 200
        
    except Exception as e:
        logger.error(f"Database events retrieval error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v2/video/detections/<video_id>', methods=['GET'])
def get_video_detections_db(video_id):
    """Get object detections from database"""
    if not DATABASE_ENABLED:
        return jsonify({'error': 'Database service not available'}), 503
    
    try:
        class_filter = request.args.get('class')  # fire, knife, gun, smoke
        
        detections_data = db_video_service.get_video_detections(video_id, class_filter)
        
        return jsonify(detections_data), 200
        
    except Exception as e:
        logger.error(f"Database detections retrieval error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v2/video/faces/<video_id>', methods=['GET'])
def get_video_faces_db(video_id):
    """Get detected faces from database for a video"""
    if not DATABASE_ENABLED:
        return jsonify({'error': 'Database service not available'}), 503
    
    try:
        faces_data = db_video_service.get_video_faces(video_id)
        
        return jsonify(faces_data), 200
        
    except Exception as e:
        logger.error(f"Database faces retrieval error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v2/video/results/<video_id>', methods=['GET'])
def get_video_results_db(video_id):
    """Get comprehensive video results from database"""
    if not DATABASE_ENABLED:
        return jsonify({'error': 'Database service not available'}), 503
    
    try:
        # Get video status and basic info
        status_data = db_video_service.get_video_status(video_id)
        
        if 'error' in status_data:
            logger.warning(f"Video not found in database: {video_id}")
            return jsonify(status_data), 404
        
        # Check if processing is completed (check multiple possible status fields)
        processing_status = status_data.get('status') or status_data.get('meta_data', {}).get('processing_status', 'unknown')
        
        # Log status for debugging
        logger.info(f"Video {video_id} status: {processing_status}, progress: {status_data.get('processing_progress')}")
        
        # Allow results even if status is not exactly 'completed' - check if we have detections/events
        meta_data = status_data.get('meta_data', {})
        has_detections = meta_data.get('detection_count', 0) > 0 or status_data.get('detection_count', 0) > 0
        has_events = meta_data.get('event_count', 0) > 0 or status_data.get('event_count', 0) > 0
        
        if processing_status not in ['completed', 'done'] and not (has_detections or has_events):
            return jsonify({
                'error': 'Processing not completed',
                'current_status': processing_status,
                'progress': status_data.get('processing_progress') or meta_data.get('processing_progress', 0),
                'message': status_data.get('processing_message') or meta_data.get('processing_message', '')
            }), 400
        
        # Get keyframes, events, and detections
        keyframes_data = db_video_service.get_video_keyframes(video_id, limit=50)
        events_data = db_video_service.get_video_events(video_id)
        detections_data = db_video_service.get_video_detections(video_id)
        
        # Extract behavior analysis events
        all_events = events_data.get('events', [])
        behavior_events = [e for e in all_events if e.get('event_type', '').startswith('behavior_')]
        
        # Summarize behavior detections
        behavior_summary = _summarize_behaviors(behavior_events)
        
        # Get compressed video URL from status
        compressed_video_url = status_data.get('compressed_video_url') or f'/api/video/compressed/{video_id}'
        compressed_video_available = bool(status_data.get('compressed_video_url') or status_data.get('meta_data', {}).get('minio_compressed_path'))
        
        # Compile comprehensive results
        results = {
            'video_info': status_data,
            'compressed_video_available': compressed_video_available,
            'compressed_video_url': compressed_video_url,
            'keyframes_available': len(keyframes_data.get('keyframes', [])) > 0,
            'keyframes_count': keyframes_data.get('total_keyframes', 0),
            'keyframes_sample': keyframes_data.get('keyframes', [])[:10],  # First 10 keyframes
            'events_available': len(events_data.get('events', [])) > 0,
            'events_count': events_data.get('total_events', 0),
            'events_summary': _summarize_events(events_data.get('events', [])),
            'detections_available': len(detections_data.get('detections', [])) > 0,
            'detections_count': detections_data.get('total_detections', 0),
            'detections_summary': _summarize_detections(detections_data.get('detections', [])),
            'behaviors_available': len(behavior_events) > 0,
            'behaviors_count': len(behavior_events),
            'behaviors_summary': behavior_summary,
            'behavior_events': behavior_events[:10],  # First 10 behavior events
            'threat_assessment': _assess_threat_level(events_data.get('events', []), detections_data.get('detections', []))
        }
        
        return jsonify(results), 200
        
    except Exception as e:
        logger.error(f"Database results retrieval error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/video/upload', methods=['POST'])
@app.route('/api/upload', methods=['POST'])
@require_subscription()  # Requires any active subscription (Basic or Pro)
@check_usage_limit('video_processing')  # Check and increment video processing limit
def upload_video():
    """Upload video endpoint. Requires: Active subscription"""
    try:
        # Check if file is present
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv, wmv, flv'}), 400
        
        # Get processing configuration (default to DetectifAI optimized)
        config_type = request.form.get('config_type', 'detectifai')
        
        # Generate unique video ID
        video_id = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{video_id}_{filename}")
        file.save(video_path)
        
        # Initialize processing status
        processing_status[video_id] = {
            'video_id': video_id,
            'filename': filename,
            'status': 'queued',
            'progress': 0,
            'message': 'Video uploaded successfully. Processing queued.',
            'uploaded_at': datetime.now().isoformat(),
            'config_type': config_type
        }
        
        # Start background processing
        thread = threading.Thread(
            target=process_video_async,
            args=(video_id, video_path, config_type)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'video_id': video_id,
            'message': 'Video uploaded successfully. Processing started.',
            'status_url': f'/api/status/{video_id}'
        }), 200
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/video/status/<video_id>', methods=['GET'])
@app.route('/api/status/<video_id>', methods=['GET'])
def get_status(video_id):
    """Get processing status for a video"""
    # Check memory first
    if video_id in processing_status:
        return jsonify(processing_status[video_id]), 200
    
    # Check if video files exist on disk (recovered processing)
    output_dir = os.path.join(OUTPUT_FOLDER, video_id)
    if os.path.exists(output_dir):
        # Recover status from disk
        recovered_status = {
            'video_id': video_id,
            'status': 'completed',
            'progress': 100,
            'message': 'Processing completed (recovered from disk)',
            'uploaded_at': '',
            'filename': f"{video_id}.avi"
        }
        
        # Add back to memory for future requests
        processing_status[video_id] = recovered_status
        
        logger.info(f"🔄 Recovered status for {video_id} from disk")
        return jsonify(recovered_status), 200
    
    return jsonify({'error': 'Video not found'}), 404

@app.route('/api/results/<video_id>', methods=['GET'])
def get_results(video_id):
    """Get processing results for a video"""
    if video_id not in processing_status:
        return jsonify({'error': 'Video not found'}), 404
    
    status = processing_status[video_id]
    
    if status['status'] != 'completed':
        return jsonify({
            'error': 'Processing not completed',
            'current_status': status['status']
        }), 400
    
    return jsonify(status.get('results', {})), 200

@app.route('/api/video/results/<video_id>', methods=['GET'])
def get_video_results(video_id):
    """Get video processing results with availability flags"""
    # First check if video is in memory status
    if video_id in processing_status:
        status = processing_status[video_id]

        if status['status'] == 'processing':
            # Return partial results while processing
            return jsonify({
                'video_id': video_id,
                'status': 'processing',
                'progress': status.get('progress', 0),
                'message': status.get('message', 'Processing...'),
                'compressed_video_available': False,
                'keyframes_available': False,
                'reports_available': False
            }), 200

        if status['status'] == 'failed':
            return jsonify({
                'error': 'Processing failed',
                'message': status.get('message', 'Unknown error'),
                'current_status': status['status']
            }), 400

        # Check if status has results structure (normal processing)
        if 'results' in status and 'output_directory' in status['results']:
            output_dir = status['results']['output_directory']
        else:
            # Fallback to standard directory structure
            output_dir = os.path.join(OUTPUT_FOLDER, video_id)
    else:
        # Check database for video status (for database-integrated processing)
        if DATABASE_ENABLED:
            try:
                db_status = db_video_service.get_video_status(video_id)
                if 'error' not in db_status:
                    # Video found in database, construct results from database metadata
                    meta_data = db_status.get('meta_data', {})

                    # Check for compressed video in MinIO
                    compressed_video_available = bool(meta_data.get('minio_compressed_path'))
                    compressed_video_url = f'/api/video/compressed/{video_id}' if compressed_video_available else None

                    # Check for keyframes
                    keyframes_available = meta_data.get('keyframe_count', 0) > 0
                    keyframes_count = meta_data.get('keyframe_count', 0)

                    # Check for reports (assume available if processing completed)
                    reports_available = db_status.get('status') == 'completed'

                    return jsonify({
                        'video_id': video_id,
                        'status': db_status.get('status', 'unknown'),
                        'compressed_video_available': compressed_video_available,
                        'compressed_video_url': compressed_video_url,
                        'keyframes_available': keyframes_available,
                        'keyframes_count': keyframes_count,
                        'keyframes_url': f'/api/v2/video/keyframes/{video_id}',  # Use v2 endpoint for database
                        'reports_available': reports_available,
                        'reports': []  # Database doesn't store report files locally
                    }), 200
            except Exception as e:
                logger.warning(f"Database lookup failed for results: {e}")

        # Check if video files exist on disk (for recovered/restarted servers)
        output_dir = os.path.join(OUTPUT_FOLDER, video_id)
        if not os.path.exists(output_dir):
            return jsonify({'error': 'Video not found'}), 404

        logger.info(f"📁 Found video files on disk for {video_id}, recovering results")

    # Check for compressed video
    compressed_dir = os.path.join(output_dir, 'compressed')
    compressed_video_available = False
    compressed_video_url = None

    if os.path.exists(compressed_dir):
        video_files = [f for f in os.listdir(compressed_dir) if f.endswith('.mp4')]
        if video_files:
            compressed_video_available = True
            compressed_video_url = f'/api/video/compressed/{video_id}'

    # Check for keyframes
    frames_dir = os.path.join(output_dir, 'frames')
    keyframes_available = os.path.exists(frames_dir) and len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')]) > 0
    keyframes_count = len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')]) if keyframes_available else 0

    # Check for reports
    reports_dir = os.path.join(output_dir, 'reports')
    reports_available = os.path.exists(reports_dir)
    report_files = []
    if reports_available:
        report_files = [f for f in os.listdir(reports_dir) if f.endswith('.json')]

    return jsonify({
        'video_id': video_id,
        'compressed_video_available': compressed_video_available,
        'compressed_video_url': compressed_video_url,
        'keyframes_available': keyframes_available,
        'keyframes_count': keyframes_count,
        'keyframes_url': f'/api/video/keyframes/{video_id}',
        'reports_available': reports_available,
        'reports': report_files
    }), 200

@app.route('/api/download/<video_id>/<file_type>', methods=['GET'])
def download_file(video_id, file_type):
    """Download processed files"""
    if video_id not in processing_status:
        return jsonify({'error': 'Video not found'}), 404
    
    status = processing_status[video_id]
    
    if status['status'] != 'completed':
        return jsonify({'error': 'Processing not completed'}), 400
    
    output_dir = status['results']['output_directory']
    
    try:
        if file_type == 'highlight_event':
            file_path = status['results']['highlight_reels'].get('event_aware', '')
        elif file_type == 'highlight_comprehensive':
            file_path = status['results']['highlight_reels'].get('ultra_comprehensive', '')
        elif file_type == 'highlight_quality':
            file_path = status['results']['highlight_reels'].get('quality_focused', '')
        elif file_type == 'compressed_video':
            file_path = status['results']['compressed_video']
        elif file_type == 'report_processing':
            file_path = status['results']['reports'].get('processing_results', '')
        elif file_type == 'report_events':
            file_path = status['results']['reports'].get('canonical_events', '')
        elif file_type == 'html_gallery':
            file_path = status['results']['reports'].get('html_gallery', '')
        else:
            return jsonify({'error': 'Invalid file type'}), 400
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/video/keyframes/<video_id>', methods=['GET'])
@app.route('/api/keyframes/<video_id>', methods=['GET'])
def get_keyframes(video_id):
    """Get list of extracted keyframes with DetectifAI annotations"""
    if video_id not in processing_status:
        return jsonify({'error': 'Video not found'}), 404
    
    status = processing_status[video_id]
    
    if status['status'] != 'completed':
        return jsonify({'error': 'Processing not completed'}), 400
    
    output_dir = status['results']['output_directory']
    frames_dir = os.path.join(output_dir, 'frames')
    
    if not os.path.exists(frames_dir):
        return jsonify({'error': 'Frames directory not found'}), 404
    
    # Load detection metadata if available
    detection_metadata = {}
    detection_metadata_path = os.path.join(output_dir, 'detection_metadata.json')
    if os.path.exists(detection_metadata_path):
        try:
            with open(detection_metadata_path, 'r') as f:
                detection_metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load detection metadata: {e}")
    
    # Get filter parameter
    filter_detections = request.args.get('filter_detections', 'false').lower() == 'true'
    
    keyframes = []
    frames_with_detections = {item['original_path']: item for item in detection_metadata.get('detection_summary', [])}
    
    for filename in sorted(os.listdir(frames_dir)):
        if filename.endswith('.jpg') and not filename.endswith('_annotated.jpg'):
            # Extract timestamp from filename
            timestamp = 0.0
            try:
                if '_' in filename:
                    timestamp_part = filename.split('_')[1].replace('s', '').replace('.jpg', '')
                    timestamp = float(timestamp_part)
            except:
                pass
            
            frame_path = os.path.join(frames_dir, filename)
            has_detections = frame_path in frames_with_detections
            
            # Skip frames without detections if filtering is enabled
            if filter_detections and not has_detections:
                continue
            
            keyframe_data = {
                'filename': filename,
                'timestamp': timestamp,
                'url': f'/api/video/{video_id}/keyframe/{filename}',
                'minio_url': f'/api/minio/image/detectifai-keyframes/{video_id}/keyframes/{filename}',
                'has_detections': has_detections
            }
            
            # Add detection details if available
            if has_detections:
                detection_info = frames_with_detections[frame_path]
                keyframe_data.update({
                    'detection_count': detection_info.get('detection_count', 0),
                    'objects': detection_info.get('objects', []),
                    'confidence_avg': detection_info.get('confidence_avg', 0.0)
                })
            
            keyframes.append(keyframe_data)
    
    return jsonify({
        'video_id': video_id,
        'total_keyframes': detection_metadata.get('total_keyframes', len(keyframes)),
        'keyframes_with_detections': detection_metadata.get('frames_with_detections', 0),
        'keyframes': keyframes,
        'objects_detected': detection_metadata.get('objects_detected', {}),
        'filter_applied': filter_detections
    }), 200

@app.route('/api/keyframe/<video_id>/<filename>', methods=['GET'])
def get_keyframe_image(video_id, filename):
    """Serve keyframe image"""
    if video_id not in processing_status:
        return jsonify({'error': 'Video not found'}), 404
    
    status = processing_status[video_id]
    output_dir = status['results']['output_directory']
    frames_dir = os.path.join(output_dir, 'frames')
    
    return send_from_directory(frames_dir, filename)

@app.route('/api/video/compressed/<video_id>', methods=['GET'])
def get_compressed_video(video_id):
    """Serve compressed video"""
    if video_id not in processing_status:
        return jsonify({'error': 'Video not found'}), 404
    
    status = processing_status[video_id]
    
    if status['status'] != 'completed':
        return jsonify({'error': 'Processing not completed'}), 400
    
    output_dir = status['results']['output_directory']
    compressed_dir = os.path.join(output_dir, 'compressed')
    
    if not os.path.exists(compressed_dir):
        return jsonify({'error': 'Compressed video directory not found'}), 404
    
    # Find the compressed video file
    video_files = [f for f in os.listdir(compressed_dir) if f.endswith('.mp4')]
    
    if not video_files:
        return jsonify({'error': 'Compressed video file not found'}), 404
    
    # Use the first video file found (should only be one)
    video_filename = video_files[0]
    
    return send_from_directory(compressed_dir, video_filename)

@app.route('/api/videos', methods=['GET'])
def list_videos():
    """List all processed videos"""
    videos = []
    for video_id, status in processing_status.items():
        videos.append({
            'video_id': video_id,
            'filename': status.get('filename', ''),
            'status': status.get('status', ''),
            'uploaded_at': status.get('uploaded_at', ''),
            'progress': status.get('progress', 0)
        })
    
    return jsonify({'videos': videos}), 200

@app.route('/api/video/processing-summary/<video_id>', methods=['GET'])
@app.route('/api/processing-summary/<video_id>', methods=['GET'])
def get_processing_summary(video_id):
    """Get detailed processing summary for a video"""
    if video_id not in processing_status:
        return jsonify({'error': 'Video not found'}), 404
    
    status = processing_status[video_id]
    
    if status['status'] != 'completed':
        return jsonify({'error': 'Processing not completed'}), 400
    
    output_dir = status['results']['output_directory']
    
    # Load detection metadata
    detection_metadata = {}
    detection_metadata_path = os.path.join(output_dir, 'detection_metadata.json')
    if os.path.exists(detection_metadata_path):
        try:
            with open(detection_metadata_path, 'r') as f:
                detection_metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load detection metadata: {e}")
    
    # Get processing stats from status
    processing_stats = status['results'].get('processing_stats', {})
    
    summary = {
        'video_id': video_id,
        'filename': status.get('filename', ''),
        'processing_time': processing_stats.get('total_processing_time', 0),
        'keyframes_extracted': detection_metadata.get('total_keyframes', 0),
        'keyframes_with_detections': detection_metadata.get('frames_with_detections', 0),
        'objects_detected': detection_metadata.get('objects_detected', {}),
        'total_objects': sum(detection_metadata.get('objects_detected', {}).values()),
        'component_times': processing_stats.get('component_times', {}),
        'output_files': {
            'compressed_video': status['results'].get('compressed_video_path', ''),
            'frames_directory': os.path.join(output_dir, 'frames'),
            'reports_directory': os.path.join(output_dir, 'reports')
        }
    }
    
    return jsonify(summary), 200

@app.route('/api/delete/<video_id>', methods=['DELETE'])
def delete_video(video_id):
    """Delete video and its processing results"""
    if video_id not in processing_status:
        return jsonify({'error': 'Video not found'}), 404
    
    try:
        # Remove from status
        status = processing_status.pop(video_id)
        
        # Delete output directory
        if 'results' in status and 'output_directory' in status['results']:
            import shutil
            output_dir = status['results']['output_directory']
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
        
        # Delete uploaded video
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            if file.startswith(video_id):
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
        
        return jsonify({'success': True, 'message': 'Video deleted successfully'}), 200
        
    except Exception as e:
        logger.error(f"Delete error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# DetectifAI-specific endpoints

@app.route('/api/detectifai/events/<video_id>', methods=['GET'])
def get_detectifai_events(video_id):
    """Get DetectifAI security events for a video"""
    if video_id not in processing_status:
        return jsonify({'error': 'Video not found'}), 404
    
    status = processing_status[video_id]
    
    if status['status'] != 'completed':
        return jsonify({'error': 'Processing not completed'}), 400
    
    results = status.get('results', {})
    security_events = results.get('security_detection', {})
    
    return jsonify({
        'video_id': video_id,
        'security_events': security_events,
        'total_detections': security_events.get('total_object_detections', 0),
        'fire_detections': security_events.get('fire_detections', 0),
        'weapon_detections': security_events.get('weapon_detections', 0),
        'security_alerts': security_events.get('security_alerts', [])
    }), 200

@app.route('/api/detectifai/demo', methods=['GET'])
def demo_detectifai():
    """Demo endpoint to process test videos (rob.mp4, fire.avi)"""
    try:
        demo_videos = []
        
        # Check for test videos
        test_files = ['rob.mp4', 'fire.avi']
        for test_file in test_files:
            if os.path.exists(test_file):
                # Create demo processing entry
                video_id = f"demo_{test_file.replace('.', '_')}_{int(datetime.now().timestamp())}"
                
                processing_status[video_id] = {
                    'video_id': video_id,
                    'filename': test_file,
                    'status': 'ready',
                    'progress': 0,
                    'message': f'Demo video {test_file} ready for DetectifAI processing',
                    'uploaded_at': datetime.now().isoformat(),
                    'video_path': test_file,
                    'is_demo': True,
                    'config_type': 'detectifai'
                }
                
                demo_videos.append({
                    'video_id': video_id,
                    'filename': test_file,
                    'process_url': f'/api/process/{video_id}'
                })
        
        return jsonify({
            'demo_videos': demo_videos,
            'message': f'Found {len(demo_videos)} demo videos ready for DetectifAI processing'
        }), 200
        
    except Exception as e:
        logger.error(f"Demo endpoint error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/process/<video_id>', methods=['POST'])
def process_existing_video(video_id):
    """Process an existing video (useful for demo videos)"""
    if video_id not in processing_status:
        return jsonify({'error': 'Video not found'}), 404
    
    status = processing_status[video_id]
    
    if status.get('status') not in ['ready', 'failed']:
        return jsonify({'error': 'Video is already being processed or completed'}), 400
    
    video_path = status.get('video_path', '')
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video file not found'}), 404
    
    config_type = status.get('config_type', 'detectifai')
    
    # Start background processing
    thread = threading.Thread(
        target=process_video_async,
        args=(video_id, video_path, config_type)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'video_id': video_id,
        'message': 'DetectifAI processing started',
        'status_url': f'/api/status/{video_id}'
    }), 200

@app.route('/api/debug/compressed/<video_id>', methods=['GET'])
def debug_compressed_video(video_id):
    """Debug endpoint to check compressed video storage and optionally serve it"""
    if not DATABASE_ENABLED:
        return jsonify({'error': 'Database not enabled'}), 503
    
    # Check if user wants to download the video
    serve_video = request.args.get('serve', 'false').lower() == 'true'
    
    try:
        video_record = db_video_service.video_repo.get_video_by_id(video_id)
        if not video_record:
            return jsonify({'error': 'Video not found'}), 404
        
        meta_data = video_record.get('meta_data', {})
        bucket = video_record.get('minio_bucket', db_video_service.video_repo.video_bucket)
        
        # Check MinIO
        minio_info = {}
        objects = []
        try:
            objects = list(db_video_service.video_repo.minio.list_objects(bucket, prefix=f"compressed/{video_id}/", recursive=True))
            minio_info['objects_found'] = len(objects)
            minio_info['objects'] = [{'name': obj.object_name, 'size': obj.size} for obj in objects]
        except Exception as e:
            minio_info['error'] = str(e)
        
        # If user wants to serve the video, try to serve it
        if serve_video and objects:
            logger.info(f"🐛 DEBUG: Attempting to serve compressed video for: {video_id}")
            try:
                # Find video.mp4 in the objects
                video_object = None
                for obj in objects:
                    if obj.object_name.endswith('video.mp4'):
                        video_object = obj
                        break
                
                if video_object:
                    logger.info(f"🐛 DEBUG: Found video object: {video_object.object_name}")
                    
                    # Get the video data
                    minio_client = db_video_service.video_repo.minio
                    video_data = minio_client.get_object(bucket, video_object.object_name)
                    
                    # Create response
                    def generate():
                        try:
                            for chunk in video_data.stream(8192):
                                yield chunk
                        finally:
                            video_data.close()
                    
                    response = Response(
                        generate(),
                        mimetype='video/mp4',
                        headers={
                            'Content-Disposition': f'inline; filename="compressed_{video_id}.mp4"',
                            'Accept-Ranges': 'bytes'
                        }
                    )
                    
                    logger.info(f"🐛 DEBUG: Successfully serving compressed video")
                    return response
                else:
                    logger.warning(f"🐛 DEBUG: No video.mp4 found in objects")
                    
            except Exception as serve_e:
                logger.error(f"🐛 DEBUG: Failed to serve video: {serve_e}")
                return jsonify({
                    'error': f'Failed to serve video: {str(serve_e)}',
                    'video_id': video_id,
                    'bucket': bucket,
                    'minio_info': minio_info
                }), 500
        
        return jsonify({
            'video_id': video_id,
            'bucket': bucket,
            'minio_compressed_path': meta_data.get('minio_compressed_path'),
            'compression_info': meta_data.get('compression_info', {}),
            'minio_info': minio_info,
            'help': 'Add ?serve=true to download the video'
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/video/<video_id>/compressed', methods=['GET'])
@app.route('/api/video/annotated/<video_id>', methods=['GET'])
@app.route('/api/v2/video/annotated/<video_id>', methods=['GET'])
def serve_annotated_video(video_id):
    """Serve annotated video with bounding boxes from MinIO or local storage"""
    logger.info(f"🎨 Request to serve annotated video: {video_id}")
    try:
        # First try to get from database/MinIO
        video_record = None
        video_exists_in_db = False
        status_data = None
        meta_data = {}
        
        if DATABASE_ENABLED:
            try:
                status_data = db_video_service.get_video_status(video_id)
                if 'error' not in status_data:
                    video_exists_in_db = True
                    logger.info(f"✅ Found video in database: {video_id}")
                    
                    # Get video record directly
                    try:
                        video_record = db_video_service.video_repo.get_video_by_id(video_id)
                    except Exception as e:
                        logger.warning(f"Could not get video record: {e}")
                    
                    # Get metadata
                    if status_data:
                        meta_data = status_data.get('meta_data', {})
                    if not meta_data and video_record:
                        meta_data = video_record.get('meta_data', {})
                    
                    # Use detectifai-videos bucket
                    video_bucket = "detectifai-videos"
                    if video_record:
                        record_bucket = video_record.get('minio_bucket')
                        if record_bucket == "detectifai-videos":
                            video_bucket = record_bucket
                    
                    # Get annotated video path from metadata
                    minio_annotated_path = meta_data.get('minio_annotated_path')
                    annotated_video_available = meta_data.get('annotated_video_available', False)
                    
                    logger.info(f"📁 MinIO annotated path: {minio_annotated_path}")
                    logger.info(f"📁 Annotated video available: {annotated_video_available}")
                    
                    # Try to serve from MinIO
                    if minio_annotated_path and annotated_video_available:
                        try:
                            from minio.error import S3Error
                            minio_client = db_video_service.video_repo.minio
                            
                            # Check if object exists
                            try:
                                minio_client.stat_object(video_bucket, minio_annotated_path)
                                
                                # Generate presigned URL
                                from datetime import timedelta
                                presigned_url = minio_client.presigned_get_object(
                                    video_bucket,
                                    minio_annotated_path,
                                    expires=timedelta(hours=1)
                                )
                                logger.info(f"✅ Generated presigned URL for annotated video: {minio_annotated_path}")
                                return redirect(presigned_url)
                            except S3Error as e:
                                if e.code == 'NoSuchKey':
                                    logger.warning(f"⚠️ Annotated video not found in MinIO: {minio_annotated_path}")
                                else:
                                    logger.error(f"❌ MinIO error: {e}")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to get annotated video from MinIO: {e}")
                    
                    # Try local file
                    annotated_video_path = meta_data.get('annotated_video_path')
                    if annotated_video_path and os.path.exists(annotated_video_path):
                        logger.info(f"✅ Serving annotated video from local path: {annotated_video_path}")
                        return send_file(annotated_video_path, mimetype='video/mp4')
                    
            except Exception as e:
                logger.error(f"❌ Error getting video status: {e}")
        
        # Fallback: check local storage
        output_dir = os.path.join(OUTPUT_FOLDER, video_id)
        annotated_dir = os.path.join(output_dir, 'annotated')
        
        if os.path.exists(annotated_dir):
            video_files = [f for f in os.listdir(annotated_dir) if f.endswith('.mp4')]
            if video_files:
                video_filename = video_files[0]
                logger.info(f"✅ Serving annotated video from local directory: {annotated_dir}/{video_filename}")
                return send_from_directory(annotated_dir, video_filename)
        
        # If no annotated video, fallback to compressed or original
        logger.warning(f"⚠️ Annotated video not found for {video_id}, falling back to compressed")
        return serve_compressed_video(video_id)
        
    except Exception as e:
        logger.error(f"❌ Error serving annotated video: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Failed to serve annotated video: {str(e)}'}), 500

@app.route('/api/video/compressed/<video_id>', methods=['GET'])
@app.route('/api/v2/video/compressed/<video_id>', methods=['GET'])
def serve_compressed_video(video_id):
    """Serve compressed processed video from MinIO or local storage"""
    logger.info(f"🎬 Request to serve compressed video: {video_id}")
    
    # QUICK FIX: Redirect to working V3 endpoint
    logger.info(f"🔄 Redirecting to working V3 endpoint: {video_id}")
    return redirect(f'/api/v3/video/compressed/{video_id}')
    
    # ORIGINAL COMPLEX LOGIC (fallback if simple approach fails)
    try:
        # First try to get from database/MinIO
        video_record = None
        video_exists_in_db = False
        status_data = None
        meta_data = {}
        
        if DATABASE_ENABLED:
            try:
                status_data = db_video_service.get_video_status(video_id)
                if 'error' not in status_data:
                    video_exists_in_db = True
                    logger.info(f"✅ Found video in database: {video_id}")
                    logger.info(f"📊 Status data keys: {list(status_data.keys())}")
                    
                    # Get video record directly to access all fields including bucket
                    try:
                        video_record = db_video_service.video_repo.get_video_by_id(video_id)
                        if video_record:
                            logger.info(f"📁 Retrieved video record from database")
                    except Exception as e:
                        logger.warning(f"Could not get video record: {e}")
                else:
                    logger.warning(f"⚠️ Video not found in database status, but will still try MinIO: {video_id}")
                    # Still try to get video record directly
                    try:
                        video_record = db_video_service.video_repo.get_video_by_id(video_id)
                        if video_record:
                            video_exists_in_db = True
                            logger.info(f"✅ Found video record directly (status lookup failed)")
                    except Exception as e:
                        logger.warning(f"Could not get video record: {e}")
                    
                    # Try to get from MinIO directly
                    # meta_data might be nested or at root level
                    if status_data:
                        meta_data = status_data.get('meta_data', {})
                    if not meta_data and video_record:
                        meta_data = video_record.get('meta_data', {})
                        logger.info(f"📁 Retrieved meta_data from video record")
                    
                    # Get bucket from video record (should be "detectifai-videos")
                    # Always use detectifai-videos bucket as confirmed by user
                    video_bucket = "detectifai-videos"
                    if video_record:
                        record_bucket = video_record.get('minio_bucket')
                        if record_bucket:
                            logger.info(f"📦 Video bucket from record: {record_bucket}")
                            # Use record bucket if it's detectifai-videos, otherwise use default
                            if record_bucket == "detectifai-videos":
                                video_bucket = record_bucket
                            else:
                                logger.warning(f"⚠️ Record bucket ({record_bucket}) doesn't match expected (detectifai-videos), using detectifai-videos")
                                video_bucket = "detectifai-videos"
                        else:
                            logger.info(f"📦 No bucket in record, using detectifai-videos")
                    else:
                        logger.info(f"📦 No video record, using detectifai-videos bucket")
                    
                    # Ensure we're using the correct bucket
                    if video_bucket != "detectifai-videos":
                        logger.warning(f"⚠️ Bucket mismatch! Expected 'detectifai-videos', got '{video_bucket}'. Forcing to 'detectifai-videos'")
                        video_bucket = "detectifai-videos"
                    
                    logger.info(f"📦 Final video bucket: {video_bucket}")
                    
                    minio_compressed_path = meta_data.get('minio_compressed_path') if meta_data else None
                    
                    # Also check compression_info for the path
                    if not minio_compressed_path and meta_data:
                        compression_info = meta_data.get('compression_info', {})
                        minio_compressed_path = compression_info.get('minio_path')
                    
                    logger.info(f"📁 MinIO compressed path from metadata: {minio_compressed_path}")
                    logger.info(f"📁 Processing status: {meta_data.get('processing_status') if meta_data else 'N/A'}")
                    logger.info(f"📁 Full meta_data keys: {list(meta_data.keys()) if meta_data else 'N/A'}")
            except Exception as e:
                logger.warning(f"⚠️ Database lookup failed, but will still try MinIO: {e}")
                import traceback
                logger.debug(f"Database lookup traceback: {traceback.format_exc()}")
        
        # Always try MinIO first (even if database lookup failed, try standard path)
        # This ensures we can serve videos even if database is temporarily unavailable
        try:
            from io import BytesIO
            from minio.error import S3Error
            
            # Use detectifai-videos bucket as confirmed by user
            video_bucket = "detectifai-videos"
            
            # Get minio_compressed_path from metadata if available
            minio_compressed_path = meta_data.get('minio_compressed_path') if meta_data else None
            if not minio_compressed_path and meta_data:
                compression_info = meta_data.get('compression_info', {})
                minio_compressed_path = compression_info.get('minio_path')
            
            # Get compressed video path from metadata or use standard path
            # User confirmed: bucket is "detectifai-videos" and folder is "compressed"
            # Standard path format: compressed/{video_id}/video.mp4
            possible_paths = []
            
            # First, try the path from metadata if available
            if minio_compressed_path:
                # Normalize path - remove leading slash if present
                normalized_path = minio_compressed_path.lstrip('/')
                possible_paths.append(normalized_path)
                logger.info(f"📁 Using path from metadata: {normalized_path}")
            
            # Always try the standard path format (user confirmed this is correct)
            standard_path = f"compressed/{video_id}/video.mp4"
            if standard_path not in possible_paths:
                possible_paths.insert(0, standard_path)  # Try standard path first
            
            # Also try alternative formats as fallback
            alternative_paths = [
                f"compressed/{video_id}/compressed.mp4",
            ]
            for alt_path in alternative_paths:
                if alt_path not in possible_paths:
                    possible_paths.append(alt_path)
            
            logger.info(f"🔍 Will try {len(possible_paths)} possible paths in bucket: {video_bucket}")
            for i, p in enumerate(possible_paths, 1):
                logger.info(f"   {i}. {p}")
            
            # Debug: Log if DATABASE_ENABLED and which minio client we're using
            logger.info(f"📋 DEBUG: DATABASE_ENABLED = {DATABASE_ENABLED}")
            if DATABASE_ENABLED:
                logger.info(f"📋 DEBUG: compression_bucket = {compression_bucket}")
                logger.info(f"📋 DEBUG: video_bucket = {video_bucket}")
                logger.info(f"📋 DEBUG: minio_client type = {type(minio_client)}")
                logger.info(f"📋 DEBUG: minio_client available = {minio_client is not None}")
            
            video_data = None
            successful_path = None
            
            # Try to get from video bucket (compressed videos are in same bucket as originals)
            if DATABASE_ENABLED:
                compression_bucket = db_video_service.compression_service.bucket
                minio_client = db_video_service.video_repo.minio
            else:
                compression_bucket = video_bucket
                # Need to create a MinIO client if database is not enabled
                from database.config import DatabaseManager
                db_manager = DatabaseManager()
                minio_client = db_manager.minio_client
            
            # Try each possible path in the video bucket first
            logger.info(f"🔍 Trying video bucket: {video_bucket}")
            for minio_path in possible_paths:
                try:
                    logger.info(f"   Attempting: {video_bucket}/{minio_path}")
                    # Verify bucket exists first
                    if not minio_client.bucket_exists(video_bucket):
                        logger.error(f"❌ Bucket '{video_bucket}' does not exist!")
                        raise Exception(f"Bucket '{video_bucket}' does not exist")
                    
                    video_data = minio_client.get_object(
                        video_bucket,
                        minio_path
                    )
                    successful_path = minio_path
                    logger.info(f"✅ Found compressed video in video bucket: {video_bucket} at {minio_path}")
                    break
                except S3Error as s3_err:
                    error_code = getattr(s3_err, 'code', 'Unknown')
                    error_msg = str(s3_err)
                    logger.warning(f"   ❌ S3Error ({error_code}): {error_msg[:200]}")
                    if error_code == 'NoSuchKey':
                        logger.info(f"   ℹ️ Object '{minio_path}' not found in bucket '{video_bucket}'")
                    
                    # DEBUG: Let's list what's actually in the bucket at this path
                    if error_code == 'NoSuchKey':
                        try:
                            prefix = '/'.join(minio_path.split('/')[:-1]) + '/'  # Get directory path
                            logger.info(f"   🔍 DEBUG: Listing objects with prefix '{prefix}' in bucket '{video_bucket}'")
                            debug_objects = list(minio_client.list_objects(video_bucket, prefix=prefix, recursive=True))
                            if debug_objects:
                                logger.info(f"   📦 DEBUG: Found {len(debug_objects)} objects:")
                                for obj in debug_objects[:5]:  # Show first 5
                                    logger.info(f"      - {obj.object_name} ({obj.size} bytes)")
                            else:
                                logger.info(f"   📦 DEBUG: No objects found with prefix '{prefix}'")
                        except Exception as debug_e:
                            logger.warning(f"   ⚠️ DEBUG: Failed to list objects: {debug_e}")
                    continue
                except Exception as e1:
                    error_msg = str(e1)
                    logger.warning(f"   ❌ Failed: {error_msg[:200]}")
                    import traceback
                    logger.debug(f"   Traceback: {traceback.format_exc()}")
                    continue
            
            # If not found in video bucket, try compression bucket (should be same, but check anyway)
            if not video_data and compression_bucket != video_bucket and DATABASE_ENABLED:
                logger.info(f"🔍 Trying compression bucket: {compression_bucket}")
                compression_minio = db_video_service.compression_service.minio
                for minio_path in possible_paths:
                    try:
                        logger.info(f"   Attempting: {compression_bucket}/{minio_path}")
                        if not compression_minio.bucket_exists(compression_bucket):
                            logger.error(f"❌ Compression bucket '{compression_bucket}' does not exist!")
                            continue
                        
                        video_data = compression_minio.get_object(
                            compression_bucket,
                            minio_path
                        )
                        successful_path = minio_path
                        logger.info(f"✅ Found compressed video in compression bucket: {compression_bucket} at {minio_path}")
                        break
                    except S3Error as s3_err:
                        error_code = getattr(s3_err, 'code', 'Unknown')
                        logger.warning(f"   ❌ S3Error ({error_code}): {str(s3_err)[:200]}")
                        continue
                    except Exception as e2:
                        logger.warning(f"   ❌ Failed: {str(e2)[:200]}")
                        continue
            elif not video_data and compression_bucket == video_bucket:
                logger.info(f"ℹ️ Compression bucket is same as video bucket, skipping duplicate check")
            
            # If still not found, try listing objects to see what's available
            if not video_data:
                logger.warning(f"⚠️ Could not find video with standard paths, listing objects in bucket '{video_bucket}'...")
                try:
                    # List all objects with compressed prefix for this video
                    search_prefix = f"compressed/{video_id}/"
                    logger.info(f"🔍 Listing objects in '{video_bucket}' with prefix '{search_prefix}'")
                    
                    if not minio_client.bucket_exists(video_bucket):
                        logger.error(f"❌ Bucket '{video_bucket}' does not exist! Cannot list objects.")
                    else:
                        objects = list(minio_client.list_objects(video_bucket, prefix=search_prefix, recursive=True))
                        logger.info(f"📦 Found {len(objects)} objects in video bucket '{video_bucket}' with prefix '{search_prefix}'")
                        
                        if objects:
                            logger.info(f"📋 Available objects:")
                            for obj in objects:
                                logger.info(f"   - {obj.object_name} ({obj.size} bytes, modified: {obj.last_modified})")
                            
                            # Try the first object found
                            actual_path = objects[0].object_name
                            logger.info(f"🔄 Trying first object found: {actual_path}")
                            try:
                                video_data = minio_client.get_object(video_bucket, actual_path)
                                successful_path = actual_path
                                logger.info(f"✅ Successfully retrieved video from path: {actual_path}")
                            except Exception as get_err:
                                logger.error(f"❌ Failed to get object '{actual_path}': {get_err}")
                        else:
                            logger.warning(f"⚠️ No objects found with prefix '{search_prefix}' in bucket '{video_bucket}'")
                            
                            # Try listing all objects in compressed folder
                            logger.info(f"🔍 Listing all objects in 'compressed/' folder...")
                            all_compressed = list(minio_client.list_objects(video_bucket, prefix="compressed/", recursive=True))
                            logger.info(f"📦 Found {len(all_compressed)} total objects in 'compressed/' folder")
                            if all_compressed:
                                logger.info(f"📋 Sample objects in compressed folder:")
                                for obj in all_compressed[:10]:  # Show first 10
                                    logger.info(f"   - {obj.object_name}")
                    
                    # Also check compression bucket if different
                    if not video_data and compression_bucket != video_bucket and DATABASE_ENABLED:
                        logger.info(f"🔍 Listing objects in compression bucket '{compression_bucket}' with prefix '{search_prefix}'")
                        compression_minio = db_video_service.compression_service.minio
                        if compression_minio.bucket_exists(compression_bucket):
                            objects2 = list(compression_minio.list_objects(compression_bucket, prefix=search_prefix, recursive=True))
                            logger.info(f"📦 Found {len(objects2)} objects in compression bucket")
                            if objects2:
                                for obj in objects2:
                                    logger.info(f"   - {obj.object_name} ({obj.size} bytes)")
                                actual_path = objects2[0].object_name
                                logger.info(f"🔄 Trying actual path found: {actual_path}")
                                video_data = compression_minio.get_object(compression_bucket, actual_path)
                                successful_path = actual_path
                except Exception as list_err:
                    logger.error(f"❌ Failed to list objects: {list_err}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
            
            if video_data:
                # Successfully found video in MinIO
                video_bytes = video_data.read()
                video_data.close()
                video_data.release_conn()
                
                response = send_file(
                    BytesIO(video_bytes),
                    mimetype='video/mp4',
                    as_attachment=False,
                    download_name=f"{video_id}_compressed.mp4"
                )
                response.headers['Accept-Ranges'] = 'bytes'
                response.headers['Cache-Control'] = 'no-cache'
                response.headers['Access-Control-Allow-Origin'] = '*'
                response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
                response.headers['Access-Control-Allow-Headers'] = 'Range'
                response.headers['Content-Type'] = 'video/mp4'
                logger.info(f"✅ Served compressed video from MinIO for {video_id}")
                return response
            else:
                logger.warning(f"⚠️ Could not retrieve video from MinIO. Tried {len(possible_paths)} paths in buckets {video_bucket} and {compression_bucket}")
                # Fall through to local storage check
        except S3Error as e:
            logger.warning(f"⚠️ MinIO retrieval failed (S3Error), falling back to local storage: {e}")
            import traceback
            logger.error(f"S3Error traceback: {traceback.format_exc()}")
            # Don't return, continue to local fallback
        except Exception as e:
            logger.warning(f"⚠️ MinIO retrieval failed, falling back to local storage: {e}")
            import traceback
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            # Don't return, continue to local fallback
        
        # Fallback: Find the compressed video file locally (ALWAYS try this, even if database lookup failed)
        logger.info(f"🔍 Searching local file system for compressed video: {video_id}")
        
        # Get the local path from compression service if available
        local_path_from_service = None
        if DATABASE_ENABLED:
            try:
                # Try to get local path from compression service result
                if not video_record:
                    video_record = db_video_service.video_repo.get_video_by_id(video_id)
                if video_record:
                    meta_data = video_record.get('meta_data', {})
                    # Check if we have compression info with local path
                    compression_info = meta_data.get('compression_info', {})
                    if compression_info and 'local_path' in compression_info:
                        local_path_from_service = compression_info['local_path']
                        logger.info(f"📁 Found local path from compression info: {local_path_from_service}")
                    # Also check for compressed_path in compression_info (alternative field name)
                    elif compression_info and 'compressed_path' in compression_info:
                        local_path_from_service = compression_info['compressed_path']
                        logger.info(f"📁 Found local path from compression_info.compressed_path: {local_path_from_service}")
                    # Also check minio_compressed_path - might be a local path
                    elif meta_data.get('minio_compressed_path'):
                        potential_path = meta_data.get('minio_compressed_path')
                        if os.path.exists(potential_path) and not potential_path.startswith('compressed/'):
                            local_path_from_service = potential_path
                            logger.info(f"📁 Found local path from minio_compressed_path: {local_path_from_service}")
            except Exception as e:
                logger.debug(f"Could not get local path from service: {e}")
        
        # List of possible local directories to check
        possible_dirs = []
        
        # Add path from compression service if available
        if local_path_from_service:
            if os.path.exists(local_path_from_service):
                possible_dirs.append(os.path.dirname(local_path_from_service))
            elif os.path.exists(local_path_from_service):
                # If it's a file path, use its directory
                possible_dirs.append(os.path.dirname(local_path_from_service))
        
        # Add standard locations (check multiple possible locations)
        possible_dirs.extend([
            os.path.join("video_processing_outputs", "compressed", video_id),  # Standard location from compression service
            os.path.join(OUTPUT_FOLDER, video_id, 'compressed'),
            os.path.join("video_processing_outputs", video_id, "compressed"),
            os.path.join("backend", "video_processing_outputs", "compressed", video_id),  # If running from root
            os.path.join(".", "video_processing_outputs", "compressed", video_id),  # Current directory
            os.path.join("video_processing_outputs", "compressed"),  # Check root compressed dir
            os.path.join(OUTPUT_FOLDER, "compressed", video_id),  # Alternative location
        ])
        
        # Also add direct file paths that might be stored in metadata
        possible_file_paths = [
            os.path.join("video_processing_outputs", "compressed", f"{video_id}_compressed.mp4"),
            os.path.join(OUTPUT_FOLDER, "compressed", f"{video_id}_compressed.mp4"),
            os.path.join("video_processing_outputs", "compressed", video_id, "video.mp4"),
            os.path.join(OUTPUT_FOLDER, video_id, "compressed", "video.mp4"),
        ]
        
        # Check direct file paths first
        for file_path in possible_file_paths:
            if os.path.exists(file_path) and os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
                logger.info(f"✅ Found compressed video file: {file_path} ({os.path.getsize(file_path)} bytes)")
                try:
                    response = send_file(
                        file_path,
                        mimetype='video/mp4',
                        as_attachment=False,
                        download_name=os.path.basename(file_path)
                    )
                    response.headers['Accept-Ranges'] = 'bytes'
                    response.headers['Cache-Control'] = 'no-cache'
                    response.headers['Access-Control-Allow-Origin'] = '*'
                    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
                    response.headers['Access-Control-Allow-Headers'] = 'Range'
                    response.headers['Content-Type'] = 'video/mp4'
                    logger.info(f"✅ Serving compressed video from file path: {file_path}")
                    return response
                except Exception as e:
                    logger.warning(f"Failed to serve from file path {file_path}: {e}")
                    continue
        
        # Also check if local_path_from_service is a direct file path
        if local_path_from_service and os.path.exists(local_path_from_service) and os.path.isfile(local_path_from_service):
            logger.info(f"✅ Found compressed video file directly: {local_path_from_service}")
            try:
                response = send_file(
                    local_path_from_service,
                    mimetype='video/mp4',
                    as_attachment=False,
                    download_name=os.path.basename(local_path_from_service)
                )
                response.headers['Accept-Ranges'] = 'bytes'
                response.headers['Cache-Control'] = 'no-cache'
                response.headers['Access-Control-Allow-Origin'] = '*'
                response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
                response.headers['Access-Control-Allow-Headers'] = 'Range'
                response.headers['Content-Type'] = 'video/mp4'
                logger.info(f"✅ Serving compressed video from direct path: {local_path_from_service}")
                return response
            except Exception as e:
                logger.warning(f"Failed to serve from direct path: {e}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_dirs = []
        for d in possible_dirs:
            if d not in seen:
                seen.add(d)
                unique_dirs.append(d)
        
        logger.info(f"🔍 Checking {len(unique_dirs)} possible local directories")
        
        for output_dir in unique_dirs:
            logger.info(f"🔍 Checking directory: {output_dir}")
            if os.path.exists(output_dir):
                # Look for compressed video files
                try:
                    files = os.listdir(output_dir)
                    logger.info(f"📁 Files in {output_dir}: {files}")
                    
                    for file in files:
                        if file.endswith('.mp4'):
                            video_path = os.path.join(output_dir, file)
                            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                                logger.info(f"✅ Found compressed video locally: {video_path} ({os.path.getsize(video_path)} bytes)")
                                response = send_file(
                                    video_path,
                                    mimetype='video/mp4',
                                    as_attachment=False,
                                    download_name=file
                                )
                                # Add headers for video playback and streaming
                                response.headers['Accept-Ranges'] = 'bytes'
                                response.headers['Cache-Control'] = 'no-cache'
                                response.headers['Access-Control-Allow-Origin'] = '*'
                                response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
                                response.headers['Access-Control-Allow-Headers'] = 'Range'
                                response.headers['Content-Type'] = 'video/mp4'
                                logger.info(f"✅ Serving compressed video from local storage: {video_path}")
                                return response
                except Exception as dir_err:
                    logger.warning(f"⚠️ Error reading directory {output_dir}: {dir_err}")
                    continue
        
        logger.error(f"❌ No compressed video found for {video_id} in any location")
        logger.error(f"   Checked {len(unique_dirs)} directories: {unique_dirs}")
        
        # Use video_exists_in_db from earlier check, or check again if not set
        if not video_exists_in_db and DATABASE_ENABLED:
            try:
                if not video_record:
                    video_record = db_video_service.video_repo.get_video_by_id(video_id)
                video_exists_in_db = video_record is not None
            except Exception as e:
                logger.warning(f"Could not check if video exists: {e}")
        
        if not video_exists_in_db:
            logger.error(f"❌ Video {video_id} does not exist in database")
            return jsonify({'error': 'Video not found', 'video_id': video_id}), 404
        else:
            processing_status = 'unknown'
            if video_record:
                processing_status = video_record.get('meta_data', {}).get('processing_status', 'unknown')
            logger.error(f"❌ Video {video_id} exists but compressed video not found")
            logger.error(f"   Processing status: {processing_status}")
            logger.error(f"   Checked {len(unique_dirs)} directories: {unique_dirs}")
            return jsonify({
                'error': 'Compressed video not found', 
                'video_id': video_id, 
                'checked_dirs': unique_dirs,
                'processing_status': processing_status,
                'message': 'Video exists but compressed version not available. Processing may still be in progress or compression may have failed.'
            }), 404
        
    except Exception as e:
        logger.error(f"Error serving compressed video: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/video/<video_id>/keyframes', methods=['GET'])
def get_video_keyframes(video_id):
    """Get list of keyframes with detection results"""
    try:
        frames_dir = os.path.join(OUTPUT_FOLDER, video_id, 'frames')
        if not os.path.exists(frames_dir):
            return jsonify({'error': 'Keyframes not found'}), 404
        
        # Load detection metadata
        detection_metadata = {}
        detection_metadata_path = os.path.join(OUTPUT_FOLDER, video_id, 'detection_metadata.json')
        if os.path.exists(detection_metadata_path):
            try:
                with open(detection_metadata_path, 'r') as f:
                    detection_metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load detection metadata: {e}")
        
        # Build detection lookup dictionary
        detection_lookup = {}
        for item in detection_metadata.get('detection_summary', []):
            original_filename = os.path.basename(item['original_path'])
            annotated_filename = os.path.basename(item['annotated_path']) if 'annotated_path' in item else None
            detection_lookup[original_filename] = {
                'has_detections': True,
                'detection_count': item.get('detection_count', 0),
                'objects': item.get('objects', []),
                'confidence_avg': item.get('confidence_avg', 0.0),
                'annotated_filename': annotated_filename
            }
            
        keyframes = []
        for file in os.listdir(frames_dir):
            # Filter out annotated versions - only include original keyframes
            if file.endswith('.jpg') and not file.endswith('_annotated.jpg'):
                # Extract timestamp safely
                timestamp = 0.0
                try:
                    if '_' in file:
                        timestamp_part = file.split('_')[1].replace('s', '').replace('.jpg', '')
                        timestamp = float(timestamp_part)
                except (ValueError, IndexError):
                    timestamp = 0.0
                
                # Build keyframe data with detection info
                keyframe_data = {
                    'filename': file,
                    'url': f'/api/video/{video_id}/keyframe/{file}',
                    'timestamp': timestamp,
                    'has_detections': file in detection_lookup
                }
                
                # Add detection details and annotated frame URL if available
                if file in detection_lookup:
                    detection_info = detection_lookup[file]
                    keyframe_data['detection_count'] = detection_info['detection_count']
                    keyframe_data['objects'] = detection_info['objects']
                    keyframe_data['confidence_avg'] = detection_info['confidence_avg']
                    
                    # Provide annotated frame URL if it exists
                    if detection_info['annotated_filename']:
                        keyframe_data['annotated_url'] = f'/api/video/{video_id}/keyframe/{detection_info["annotated_filename"]}'
                
                keyframes.append(keyframe_data)
        
        # Sort by timestamp
        keyframes.sort(key=lambda x: x['timestamp'])
        
        return jsonify({
            'video_id': video_id,
            'keyframes': keyframes,
            'total_keyframes': len(keyframes),
            'keyframes_with_detections': detection_metadata.get('frames_with_detections', 0),
            'objects_detected': detection_metadata.get('objects_detected', {})
        })
        
    except Exception as e:
        logger.error(f"Error getting keyframes: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/video/<video_id>/keyframe/<filename>', methods=['GET'])
@app.route('/api/v2/video/keyframe/<video_id>/<filename>', methods=['GET'])
def serve_keyframe(video_id, filename):
    """Serve individual keyframe image from MinIO or local storage"""
    try:
        # First try to get from MinIO (database-integrated)
        if DATABASE_ENABLED:
            try:
                # Construct MinIO path from filename
                # Filename format: frame_000001.jpg
                # Try both path patterns (keyframes subfolder and flat)
                from io import BytesIO
                from minio.error import S3Error
                
                minio_paths_to_try = [
                    f"{video_id}/keyframes/{filename}",
                    f"{video_id}/{filename}",
                ]
                
                keyframe_bytes = None
                for minio_path in minio_paths_to_try:
                    try:
                        keyframe_data = db_video_service.keyframe_repo.minio.get_object(
                            db_video_service.keyframe_repo.bucket,
                            minio_path
                        )
                        keyframe_bytes = keyframe_data.read()
                        keyframe_data.close()
                        keyframe_data.release_conn()
                        logger.info(f"✅ Served keyframe from MinIO: {minio_path}")
                        break
                    except S3Error:
                        continue
                
                if keyframe_bytes:
                    response = send_file(
                        BytesIO(keyframe_bytes),
                        mimetype='image/jpeg',
                        as_attachment=False
                    )
                    response.headers['Cache-Control'] = 'public, max-age=3600'
                    response.headers['Access-Control-Allow-Origin'] = '*'
                    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
                    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
                    return response
                else:
                    logger.warning(f"Keyframe not found in MinIO for any path: {minio_paths_to_try}")
            except Exception as e:
                logger.warning(f"MinIO retrieval failed, trying local: {e}")
        
        # Fallback: Try local filesystem (multiple possible locations)
        local_paths_to_try = [
            os.path.join(OUTPUT_FOLDER, video_id, 'frames', filename),
            os.path.join('video_processing_outputs', 'keyframes', video_id, filename),
            os.path.join(OUTPUT_FOLDER, video_id, filename),
        ]
        for keyframe_path in local_paths_to_try:
            if os.path.exists(keyframe_path):
                response = send_file(
                    keyframe_path,
                    mimetype='image/jpeg',
                    as_attachment=False
                )
                response.headers['Access-Control-Allow-Origin'] = '*'
                return response
        
        return jsonify({'error': 'Keyframe not found'}), 404
        
    except Exception as e:
        logger.error(f"Error serving keyframe: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/minio/image/<bucket>/<path:object_path>', methods=['GET'])
def serve_minio_image(bucket, object_path):
    """
    Unified endpoint to serve images from MinIO buckets
    Supports:
    - Keyframes: detectifai-keyframes/{video_id}/keyframes/frame_*.jpg
    - Live stream keyframes: detectifai-keyframes/live/{camera_id}/*.jpg
    - NLP images: nlp-images/*.jpg
    - Face images: detectifai-faces/*.jpg
    """
    try:
        from io import BytesIO
        from minio.error import S3Error
        
        if not DATABASE_ENABLED:
            return jsonify({'error': 'Database service not available'}), 503
        
        # Get MinIO client
        minio_client = db_video_service.db_manager.minio_client
        
        # Verify bucket exists
        if not minio_client.bucket_exists(bucket):
            logger.warning(f"Bucket {bucket} does not exist")
            return jsonify({'error': f'Bucket {bucket} not found'}), 404
        
        try:
            # Get object from MinIO
            image_data = minio_client.get_object(bucket, object_path)
            image_bytes = image_data.read()
            image_data.close()
            image_data.release_conn()
            
            # Determine content type from file extension
            content_type = 'image/jpeg'
            if object_path.lower().endswith('.png'):
                content_type = 'image/png'
            elif object_path.lower().endswith('.webp'):
                content_type = 'image/webp'
            elif object_path.lower().endswith('.gif'):
                content_type = 'image/gif'
            
            response = send_file(
                BytesIO(image_bytes),
                mimetype=content_type,
                as_attachment=False
            )
            response.headers['Cache-Control'] = 'public, max-age=3600'
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            
            logger.info(f"✅ Served image from MinIO: {bucket}/{object_path}")
            return response
            
        except S3Error as e:
            logger.error(f"MinIO error retrieving {bucket}/{object_path}: {e}")
            if e.code == 'NoSuchKey':
                return jsonify({'error': 'Image not found in MinIO'}), 404
            return jsonify({'error': f'MinIO error: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Error serving MinIO image: {e}")
        return jsonify({'error': f'Error serving image: {str(e)}'}), 500



@app.route('/api/v3/video/compressed/<video_id>', methods=['GET'])
def serve_compressed_video_v3(video_id):
    """NEW: Simple working compressed video endpoint with local fallback"""
    logger.info(f"🆕 V3 Request to serve compressed video: {video_id}")
    
    # 1. Try MinIO if database is enabled
    if DATABASE_ENABLED:
        try:
            # Get video record
            video_record = db_video_service.video_repo.get_video_by_id(video_id)
            if video_record:
                logger.info(f"🆕 Found video record for: {video_id}")
                
                # Get MinIO client and bucket
                minio_client = db_video_service.video_repo.minio
                bucket = "detectifai-videos"
                
                # Standard path where compressed videos should be
                minio_path = f"compressed/{video_id}/video.mp4"
                
                logger.info(f"🆕 Attempting to generate presigned URL for MinIO: {bucket}/{minio_path}")
                
                # Check if object exists first
                stat = minio_client.stat_object(bucket, minio_path)
                
                # Generate presigned URL (valid for 1 hour)
                from datetime import timedelta
                presigned_url = minio_client.presigned_get_object(
                    bucket, 
                    minio_path, 
                    expires=timedelta(hours=1),
                    response_headers={
                        'response-content-disposition': f'inline; filename="compressed_{video_id}.mp4"',
                        'response-content-type': 'video/mp4'
                    }
                )
                
                # Fix for Docker vs Localhost networking issues
                # If running locally but MinIO is in docker/internal network, URL might be unreachabled
                # We assume if request comes to localhost, MinIO is also on localhost
                if 'localhost' in request.host or '127.0.0.1' in request.host:
                   # Replace internal hostname (like 'minio') with localhost if present in URL
                   # This is a heuristic fix for common dev setups
                   # Extract port from presigned URL keys
                   parsed_url = urllib.parse.urlparse(presigned_url)
                   if parsed_url.hostname not in ['localhost', '127.0.0.1']:
                       new_netloc = parsed_url.netloc.replace(parsed_url.hostname, 'localhost')
                       presigned_url = parsed_url._replace(netloc=new_netloc).geturl()
                       logger.info(f"🔄 Adjusted presigned URL for localhost: {presigned_url}")
                
                logger.info(f"🆕 Redirecting to presigned URL for video: {video_id}")
                return redirect(presigned_url, code=302)

            else:
                logger.warning(f"🆕 Video record not found in DB for: {video_id}")
                
        except Exception as minio_e:
            logger.warning(f"🆕 MinIO compressed video failed: {minio_e}")
            
            # Fallback to original video if compressed doesn't exist
            try:
                logger.info(f"🔄 Trying original video as fallback for: {video_id}")
                
                # Get video record to find original path
                video_record = db_video_service.video_repo.get_video_by_id(video_id)
                if video_record and 'minio_object_key' in video_record:
                    original_path = video_record['minio_object_key']
                    bucket = video_record.get('minio_bucket', 'detectifai-videos')
                    
                    logger.info(f"🆕 Attempting original video from MinIO: {bucket}/{original_path}")
                    
                    # Check if original exists
                    stat = minio_client.stat_object(bucket, original_path)
                    
                    # Generate presigned URL for original
                    from datetime import timedelta
                    presigned_url = minio_client.presigned_get_object(
                        bucket, 
                        original_path, 
                        expires=timedelta(hours=1),
                        response_headers={
                            'response-content-disposition': f'inline; filename="video_{video_id}.mp4"',
                            'response-content-type': 'video/mp4'
                        }
                    )
                    
                    # Fix for localhost
                    if 'localhost' in request.host or '127.0.0.1' in request.host:
                        parsed_url = urllib.parse.urlparse(presigned_url)
                        if parsed_url.hostname not in ['localhost', '127.0.0.1']:
                            new_netloc = parsed_url.netloc.replace(parsed_url.hostname, 'localhost')
                            presigned_url = parsed_url._replace(netloc=new_netloc).geturl()
                    
                    logger.info(f"✅ Redirecting to ORIGINAL video for: {video_id}")
                    return redirect(presigned_url, code=302)
                    
            except Exception as original_e:
                logger.warning(f"🆕 Original video fallback also failed: {original_e}")

    # 2. Fallback: Try local filesystem
    logger.info(f"🔄 V3 Fallback: Checking local filesystem for video {video_id}")
    
    try:
        # Possible local paths
        possible_paths = [
            os.path.join(OUTPUT_FOLDER, video_id, 'compressed', 'video.mp4'),
            os.path.join(OUTPUT_FOLDER, video_id, 'compressed', f'{video_id}_compressed.mp4'),
            os.path.join("video_processing_outputs", video_id, "compressed", "video.mp4"),
            os.path.join(OUTPUT_FOLDER, "compressed", video_id, "video.mp4"),
            # Also check upload folder if it was just uploaded but not fully processed
            os.path.join(app.config['UPLOAD_FOLDER'], video_id, 'compressed', 'video.mp4')
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                logger.info(f"✅ Found compressed video locally: {path} ({os.path.getsize(path)} bytes)")
                response = send_file(
                    path,
                    mimetype='video/mp4',
                    as_attachment=False,
                    download_name=f"compressed_{video_id}.mp4"
                )
                # Add headers for video playback and streaming
                response.headers['Accept-Ranges'] = 'bytes'
                response.headers['Cache-Control'] = 'public, max-age=3600'
                response.headers['Access-Control-Allow-Origin'] = '*'
                response.headers['Content-Type'] = 'video/mp4'
                logger.info(f"✅ Serving compressed video from local fallback: {path}")
                return response
        
        logger.error(f"❌ No compressed video found for {video_id} in local fallback paths")
        return jsonify({'error': 'Video not found locally or in cloud'}), 404
        
    except Exception as local_e:
        logger.error(f"❌ Local fallback error: {local_e}")
        return jsonify({'error': str(local_e)}), 500

@app.route('/api/minio/presigned/<bucket>/<path:object_path>', methods=['GET'])
def get_minio_presigned_url(bucket, object_path):
    """
    Generate presigned URL for MinIO object
    Useful for direct client access to images
    """
    try:
        from datetime import timedelta
        from minio.error import S3Error
        
        if not DATABASE_ENABLED:
            return jsonify({'error': 'Database service not available'}), 503
        
        # Get expiration time from query parameter (default 1 hour)
        expires_hours = request.args.get('expires', 1, type=int)
        expires = timedelta(hours=expires_hours)
        
        # Get MinIO client
        minio_client = db_video_service.db_manager.minio_client
        
        # Verify bucket exists
        if not minio_client.bucket_exists(bucket):
            return jsonify({'error': f'Bucket {bucket} not found'}), 404
        
        try:
            # Generate presigned URL
            presigned_url = minio_client.presigned_get_object(
                bucket,
                object_path,
                expires=expires
            )
            
            return jsonify({
                'success': True,
                'url': presigned_url,
                'bucket': bucket,
                'object_path': object_path,
                'expires_in_hours': expires_hours
            })
            
        except S3Error as e:
            logger.error(f"MinIO error generating presigned URL: {e}")
            return jsonify({'error': f'MinIO error: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Error generating presigned URL: {e}")
        return jsonify({'error': f'Error: {str(e)}'}), 500

# ====== HELPER FUNCTIONS ======

def _summarize_behaviors(behavior_events: List[Dict]) -> Dict:
    """Summarize behavior analysis results"""
    if not behavior_events:
        return {
            'total_behaviors': 0,
            'by_type': {},
            'most_common': None,
            'average_confidence': 0.0,
            'behavior_types': []
        }
    
    # Count behaviors by type
    behavior_counts = {}
    confidences = []
    behavior_types = []
    
    for event in behavior_events:
        event_type = event.get('event_type', '')
        # Extract behavior type from "behavior_fighting" -> "fighting"
        if event_type.startswith('behavior_'):
            behavior_type = event_type.replace('behavior_', '')
            behavior_types.append(behavior_type)
            behavior_counts[behavior_type] = behavior_counts.get(behavior_type, 0) + 1
            
            confidence = event.get('confidence_score', 0.0)
            if confidence:
                confidences.append(float(confidence))
    
    # Get most common behavior
    most_common = None
    if behavior_counts:
        most_common = max(behavior_counts.items(), key=lambda x: x[1])[0]
    
    return {
        'total_behaviors': len(behavior_events),
        'by_type': behavior_counts,
        'most_common': most_common,
        'average_confidence': sum(confidences) / len(confidences) if confidences else 0.0,
        'behavior_types': list(set(behavior_types))
    }

def _summarize_events(events: List[Dict]) -> Dict:
    """Summarize events by type and threat level"""
    summary = {
        'by_type': {},
        'by_threat_level': {},
        'total_duration': 0.0,
        'highest_confidence': 0.0
    }
    
    for event in events:
        # Count by type
        event_type = event.get('event_type', 'unknown')
        summary['by_type'][event_type] = summary['by_type'].get(event_type, 0) + 1
        
        # Count by threat level
        threat_level = event.get('threat_level', 'low')
        summary['by_threat_level'][threat_level] = summary['by_threat_level'].get(threat_level, 0) + 1
        
        # Calculate duration
        start = event.get('start_timestamp', 0)
        end = event.get('end_timestamp', 0)
        summary['total_duration'] += (end - start)
        
        # Track highest confidence
        confidence = event.get('confidence', 0)
        summary['highest_confidence'] = max(summary['highest_confidence'], confidence)
    
    return summary

def _summarize_detections(detections: List[Dict]) -> Dict:
    """Summarize object detections by class and confidence"""
    summary = {
        'by_class': {},
        'average_confidence': 0.0,
        'highest_confidence': 0.0,
        'threat_objects': []
    }
    
    if not detections:
        return summary
    
    total_confidence = 0.0
    threat_classes = ['fire', 'gun', 'knife', 'smoke']
    
    for detection in detections:
        # Count by class
        class_name = detection.get('class_name', 'unknown')
        summary['by_class'][class_name] = summary['by_class'].get(class_name, 0) + 1
        
        # Calculate confidence stats
        confidence = detection.get('confidence', 0)
        total_confidence += confidence
        summary['highest_confidence'] = max(summary['highest_confidence'], confidence)
        
        # Track threat objects
        if class_name in threat_classes and class_name not in summary['threat_objects']:
            summary['threat_objects'].append(class_name)
    
    # Calculate average confidence
    summary['average_confidence'] = total_confidence / len(detections) if detections else 0.0
    
    return summary

def _assess_threat_level(events: List[Dict], detections: List[Dict]) -> Dict:
    """Assess overall threat level based on events and detections"""
    assessment = {
        'overall_level': 'low',
        'confidence_score': 0.0,
        'risk_factors': [],
        'recommendation': 'No immediate action required'
    }
    
    risk_score = 0.0
    risk_factors = []
    
    # Analyze events
    critical_events = sum(1 for e in events if e.get('threat_level') == 'critical')
    high_events = sum(1 for e in events if e.get('threat_level') == 'high')
    
    if critical_events > 0:
        risk_score += critical_events * 10.0
        risk_factors.append(f"{critical_events} critical events detected")
    
    if high_events > 0:
        risk_score += high_events * 5.0
        risk_factors.append(f"{high_events} high-risk events detected")
    
    # Analyze detections
    critical_objects = sum(1 for d in detections if d.get('class_name') in ['fire', 'gun'])
    high_objects = sum(1 for d in detections if d.get('class_name') == 'knife')
    
    if critical_objects > 0:
        risk_score += critical_objects * 8.0
        risk_factors.append(f"{critical_objects} critical objects detected (fire/gun)")
    
    if high_objects > 0:
        risk_score += high_objects * 4.0
        risk_factors.append(f"{high_objects} weapons detected (knife)")
    
    # Calculate overall threat level
    if risk_score >= 20.0:
        assessment['overall_level'] = 'critical'
        assessment['recommendation'] = 'Immediate response required - potential emergency situation'
    elif risk_score >= 10.0:
        assessment['overall_level'] = 'high'
        assessment['recommendation'] = 'Investigation recommended - elevated security concern'
    elif risk_score >= 5.0:
        assessment['overall_level'] = 'medium'
        assessment['recommendation'] = 'Monitor situation - potential security interest'
    else:
        assessment['overall_level'] = 'low'
        assessment['recommendation'] = 'Normal activity - routine monitoring sufficient'
    
    assessment['confidence_score'] = min(risk_score / 20.0, 1.0)  # Normalize to 0-1
    assessment['risk_factors'] = risk_factors
    
    return assessment

@app.route('/api/search/person-by-image', methods=['POST'])
# @require_feature('image_search')  # Pro plan feature - Temporarily disabled for development
def search_person_by_image():
    """
    Search for a person by uploading their image.
    Uses facial recognition to find similar faces in the database.
    Requires: Pro plan (image_search feature)
    """
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No image file selected'
            }), 400
        
        # Validate file type
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Please upload an image file.'
            }), 400
        
        # Save uploaded image temporarily
        filename = secure_filename(f"search_{int(time.time())}_{file.filename}")
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(temp_path)
        
        try:
            # Initialize facial recognition system
            from facial_recognition import FacialRecognitionIntegrated
            from config import VideoProcessingConfig
            
            config = VideoProcessingConfig()
            config.enable_facial_recognition = True
            
            face_recognizer = FacialRecognitionIntegrated(config)
            
            if not face_recognizer.enabled:
                return jsonify({
                    'success': False,
                    'error': 'Facial recognition system is not enabled or properly configured'
                }), 500
            
            # Get search parameters from request
            threshold = float(request.form.get('threshold', 0.6))
            max_results = int(request.form.get('max_results', 10))
            
            # Perform image search
            search_results = face_recognizer.search_person_by_image(
                temp_path, 
                k=max_results, 
                threshold=threshold
            )
            
            # Format results for frontend and enrich with event/video info from MongoDB
            formatted_results = []
            for result in search_results:
                face_id = result['face_id']
                event_id = None
                video_id = None
                start_timestamp = result.get('timestamp', 0.0)
                end_timestamp = start_timestamp + 5.0  # Default 5 second clip
                
                # Try to extract event_id from face_id (format: face_{person}_{event}_{frame}_{index}_{uuid})
                # Example: face_unknown_event_obj_detection_1234_000000_00_abc12345
                face_id_parts = face_id.split('_')
                if 'event' in face_id_parts:
                    try:
                        event_idx = face_id_parts.index('event')
                        # Extract event type and timestamp
                        event_type = '_'.join(face_id_parts[event_idx+1:event_idx+3])  # e.g., "obj_detection"
                        event_timestamp = face_id_parts[event_idx+3] if len(face_id_parts) > event_idx+3 else None
                        
                        # Try to construct event_id
                        if event_timestamp:
                            potential_event_id = f"event_{event_type}_{event_timestamp}"
                            logger.info(f"Extracted potential event_id from face_id: {potential_event_id}")
                    except Exception as e:
                        logger.warning(f"Could not parse event info from face_id {face_id}: {e}")
                
                # Try to get event_id and video_id from MongoDB
                if DATABASE_ENABLED:
                    try:
                        # Query detected_faces collection for this face_id
                        faces_collection = db_video_service.db_manager.db.detected_faces
                        face_doc = faces_collection.find_one({"face_id": face_id})
                        
                        if face_doc:
                            event_id = face_doc.get('event_id')
                            logger.info(f"Found face_doc with event_id: {event_id}")
                        else:
                            logger.warning(f"No face document found for face_id: {face_id}")
                            
                            # Try alternative queries
                            # Query by partial face_id match
                            face_doc = faces_collection.find_one({"face_id": {"$regex": f"^{face_id[:20]}"}})
                            if face_doc:
                                event_id = face_doc.get('event_id')
                                logger.info(f"Found face via regex with event_id: {event_id}")
                        
                        # Query events collection for video_id
                        if event_id:
                            from bson.objectid import ObjectId
                            events_collection = db_video_service.db_manager.db.event
                            # Try _id first (ObjectId), fallback to event_id field
                            try:
                                event_doc = events_collection.find_one({"_id": ObjectId(event_id)})
                            except:
                                event_doc = events_collection.find_one({"event_id": event_id})
                            
                            if event_doc:
                                video_id = event_doc.get('video_id')
                                # Get actual timestamps from event
                                start_timestamp = event_doc.get('start_timestamp_ms', 0) / 1000.0
                                end_timestamp = event_doc.get('end_timestamp_ms', 0) / 1000.0
                                logger.info(f"Found event with video_id: {video_id}, timestamps: {start_timestamp}-{end_timestamp}")
                            else:
                                logger.warning(f"No event document found for event_id: {event_id}")
                        else:
                            logger.info("No event_id found, clip will not be available")
                    except Exception as e:
                        logger.warning(f"Could not fetch event/video info for face {face_id}: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Get face detections for this face_id to enable annotation
                face_detections_count = 0
                if DATABASE_ENABLED and face_id:
                    try:
                        faces_collection = db_video_service.db_manager.db.detected_faces
                        if video_id:
                            face_detections_count = faces_collection.count_documents({
                                "face_id": face_id,
                                "video_id": video_id
                            })
                        elif event_id:
                            face_detections_count = faces_collection.count_documents({
                                "face_id": face_id,
                                "event_id": event_id
                            })
                    except Exception as e:
                        logger.warning(f"Could not count face detections: {e}")
                
                # Build thumbnail URL - ensure face image exists
                thumbnail_url = None
                if result.get('face_image_path') and os.path.exists(result['face_image_path']):
                    thumbnail_url = f"/api/face-image/{face_id}"
                    logger.info(f"✅ Face image exists at {result['face_image_path']}, thumbnail URL: {thumbnail_url}")
                else:
                    logger.warning(f"❌ Face image not found at {result.get('face_image_path')}")
                
                # Determine if clip is available
                clip_is_available = event_id is not None and video_id is not None
                logger.info(f"📹 Clip status for {face_id}: available={clip_is_available} (event_id={event_id}, video_id={video_id})")
                
                formatted_result = {
                    'id': face_id,
                    'face_id': face_id,
                    'event_id': event_id,
                    'video_id': video_id,
                    'person_name': result['person_name'],
                    'confidence': round(result['similarity_score'], 3),
                    'person_confidence': round(result['person_confidence'], 3) if result.get('person_confidence') else 0.0,
                    'timestamp': result['timestamp'],
                    'start_timestamp': start_timestamp,
                    'end_timestamp': end_timestamp,
                    'event_context': result['event_context'],
                    'detection_context': result['detection_context'],
                    'thumbnail': thumbnail_url,
                    'description': f"{result['person_name']} detected in {result['detection_context'].lower()}",
                    'zone': 'Security Zone',  # Placeholder
                    'has_face_image': thumbnail_url is not None,
                    'clip_available': event_id is not None and video_id is not None,
                    'annotated_clip_available': face_detections_count > 0 and event_id is not None and video_id is not None,
                    'annotated_clip_url': (
                        f"/api/event/clip/{event_id}/annotated?face_id={face_id}&person_name={urllib.parse.quote(result['person_name'])}" 
                        if (event_id and face_id and result.get('person_name'))
                        else (f"/api/event/clip/{event_id}/annotated?face_id={face_id}" if (event_id and face_id) else None)
                    )
                }
                formatted_results.append(formatted_result)
            
            # Get system statistics
            stats = face_recognizer.get_detection_stats()
            
            response_data = {
                'success': True,
                'results': formatted_results,
                'total_matches': len(formatted_results),
                'search_parameters': {
                    'similarity_threshold': threshold,
                    'max_results': max_results
                },
                'system_stats': {
                    'total_faces_in_database': stats.get('total_faces_in_database', 0),
                    'implementation_mode': stats.get('implementation_mode', 'unknown')
                },
                'message': f"Found {len(formatted_results)} matches with similarity >= {threshold}"
            }
            
            return jsonify(response_data)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        logger.error(f"Error in person image search: {e}")
        return jsonify({'error': str(e)}), 500


# ===== VIDEO CAPTIONING ENDPOINTS =====

@app.route('/api/captions/search', methods=['POST'])
# @require_feature('nlp_search')  # Pro plan feature - Temporarily disabled for development
def search_captions():
    """Search video captions using semantic similarity. Requires: Pro plan (nlp_search feature)"""
    try:
        data = request.get_json()
        query = data.get('query')
        video_id = data.get('video_id')  # Optional filter
        top_k = data.get('top_k', 10)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Import and initialize captioning integrator
        from video_captioning_integrator import VideoCaptioningIntegrator
        from config import VideoProcessingConfig
        
        config = VideoProcessingConfig(enable_video_captioning=True)
        captioning_integrator = VideoCaptioningIntegrator(config)
        
        if not captioning_integrator.enabled:
            return jsonify({'error': 'Video captioning is not enabled'}), 503
        
        # Search captions
        results = captioning_integrator.search_captions(query, video_id=video_id, top_k=top_k)
        
        return jsonify({
            'success': True,
            'query': query,
            'total_results': len(results),
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error searching captions: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/captions/video/<video_id>', methods=['GET'])
def get_video_captions(video_id):
    """Get all captions for a specific video"""
    try:
        # Import and initialize captioning integrator
        from video_captioning_integrator import VideoCaptioningIntegrator
        from config import VideoProcessingConfig
        
        config = VideoProcessingConfig(enable_video_captioning=True)
        captioning_integrator = VideoCaptioningIntegrator(config)
        
        if not captioning_integrator.enabled:
            return jsonify({'error': 'Video captioning is not enabled'}), 503
        
        # Get captions for video
        captions = captioning_integrator.get_video_captions(video_id)
        
        return jsonify({
            'success': True,
            'video_id': video_id,
            'total_captions': len(captions),
            'captions': captions
        })
        
    except Exception as e:
        logger.error(f"Error getting video captions: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/captions/statistics', methods=['GET'])
def get_captioning_statistics():
    """Get video captioning service statistics"""
    try:
        # Import and initialize captioning integrator
        from video_captioning_integrator import VideoCaptioningIntegrator
        from config import VideoProcessingConfig
        
        config = VideoProcessingConfig(enable_video_captioning=True)
        captioning_integrator = VideoCaptioningIntegrator(config)
        
        if not captioning_integrator.enabled:
            return jsonify({'error': 'Video captioning is not enabled'}), 503
        
        # Get statistics
        stats = captioning_integrator.get_statistics()
        
        return jsonify({
            'success': True,
            'statistics': stats
        })
        
    except Exception as e:
        logger.error(f"Error getting captioning statistics: {e}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/event/clip/<event_id>/annotated', methods=['GET'])
def get_annotated_event_clip(event_id):
    """
    Generate and serve annotated event clip with face bounding boxes for a specific person
    Query params: face_id (required), person_name (optional)
    """
    try:
        if not DATABASE_ENABLED:
            return jsonify({'error': 'Database not enabled'}), 500
        
        face_id = request.args.get('face_id')
        person_name = request.args.get('person_name')
        
        if not face_id:
            return jsonify({'error': 'face_id parameter is required'}), 400
        
        # Get event from database (using singular 'event' collection)
        from bson.objectid import ObjectId
        events_collection = db_video_service.db_manager.db.event
        # Try _id first (ObjectId), fallback to event_id field
        try:
            event = events_collection.find_one({"_id": ObjectId(event_id)})
        except:
            event = events_collection.find_one({"event_id": event_id})
        
        if not event:
            return jsonify({'error': 'Event not found'}), 404
        
        video_id = event.get('video_id')
        start_timestamp_ms = int(event.get('start_timestamp_ms', 0))
        end_timestamp_ms = int(event.get('end_timestamp_ms', 0))
        
        start_time = start_timestamp_ms / 1000.0
        end_time = end_timestamp_ms / 1000.0
        
        # Get all face detections for this face_id in this video
        faces_collection = db_video_service.db_manager.db.detected_faces
        
        # Try to get face detections with video_id first
        face_detections = list(faces_collection.find({
            "face_id": face_id,
            "video_id": video_id
        }))
        
        if not face_detections:
            # Fallback: try to get from event_id
            face_detections = list(faces_collection.find({
                "face_id": face_id,
                "event_id": event_id
            }))
        
        if not face_detections:
            # Last resort: get all detections for this face_id
            face_detections = list(faces_collection.find({
                "face_id": face_id
            }))
        
        logger.info(f"Found {len(face_detections)} face detections for face_id {face_id}")
        
        # Get video path (same logic as get_event_clip)
        video_record = db_video_service.video_repo.get_video_by_id(video_id)
        if not video_record:
            return jsonify({'error': 'Video not found'}), 404
        
        video_path = None
        minio_key = video_record.get('minio_object_key')
        if minio_key:
            try:
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_path = temp_file.name
                temp_file.close()
                
                db_video_service.video_repo.minio.fget_object(
                    video_record.get('minio_bucket', db_video_service.video_repo.video_bucket),
                    minio_key,
                    temp_path
                )
                video_path = temp_path
            except Exception as e:
                logger.warning(f"Could not get video from MinIO: {e}")
        
        if not video_path:
            # Try local compressed video
            local_compressed = os.path.join('video_processing_outputs', 'compressed', video_id, 'video.mp4')
            logger.info(f"Checking local compressed path: {os.path.abspath(local_compressed)}")
            if os.path.exists(local_compressed):
                video_path = local_compressed
                logger.info(f"✅ Using local compressed video: {local_compressed}")
            else:
                logger.warning(f"❌ Local compressed video not found at: {os.path.abspath(local_compressed)}")
                # Try database file_path
                file_path = video_record.get('file_path')
                if file_path and os.path.exists(file_path):
                    video_path = file_path
                    logger.info(f"Using file_path: {file_path}")
                else:
                    # Try uploads folder
                    uploads_path = os.path.join(UPLOAD_FOLDER, video_id, 'video.mp4')
                    if os.path.exists(uploads_path):
                        video_path = uploads_path
                        logger.info(f"Using uploads path: {uploads_path}")
        
        if not video_path or not os.path.exists(video_path):
            logger.error(f"❌ Video file not found for video_id: {video_id}")
            return jsonify({'error': 'Video file not found'}), 404
        
        # Convert face detections to list of dicts
        from database.models import convert_objectid_to_string
        face_detections_list = [convert_objectid_to_string(det) for det in face_detections]
        
        # Generate annotated clip
        from event_clip_generator import EventClipGenerator
        clip_generator = EventClipGenerator()
        clip_path = clip_generator.extract_annotated_clip(
            video_path, start_time, end_time, face_id, face_detections_list, video_id, person_name
        )
        
        if not clip_path or not os.path.exists(clip_path):
            return jsonify({'error': 'Failed to generate annotated clip'}), 500
        
        # Serve the clip
        return send_file(clip_path, mimetype='video/mp4')
        
    except Exception as e:
        logger.error(f"Error generating annotated event clip: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/event/clip/<event_id>', methods=['GET'])
def get_event_clip(event_id):
    """
    Generate and serve event clip for viewing/playing
    """
    try:
        if not DATABASE_ENABLED:
            return jsonify({'error': 'Database not enabled'}), 500
        
        # Get event from database (using singular 'event' collection)
        from bson.objectid import ObjectId
        events_collection = db_video_service.db_manager.db.event
        # Try _id first (ObjectId), fallback to event_id field
        try:
            event = events_collection.find_one({"_id": ObjectId(event_id)})
        except:
            event = events_collection.find_one({"event_id": event_id})
        
        if not event:
            return jsonify({'error': 'Event not found'}), 404
        
        video_id = event.get('video_id')
        start_timestamp_ms = int(event.get('start_timestamp_ms', 0))
        end_timestamp_ms = int(event.get('end_timestamp_ms', 0))
        
        start_time = start_timestamp_ms / 1000.0
        end_time = end_timestamp_ms / 1000.0
        
        # Get video path
        video_record = db_video_service.video_repo.get_video_by_id(video_id)
        if not video_record:
            return jsonify({'error': 'Video not found'}), 404
        
        # Try to get video path from MinIO or local storage
        video_path = None
        
        # Try MinIO first
        minio_key = video_record.get('minio_object_key')
        if minio_key:
            try:
                # Download from MinIO to temp file
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_path = temp_file.name
                temp_file.close()
                
                db_video_service.video_repo.minio.fget_object(
                    video_record.get('minio_bucket', db_video_service.video_repo.video_bucket),
                    minio_key,
                    temp_path
                )
                video_path = temp_path
            except Exception as e:
                logger.warning(f"Could not get video from MinIO: {e}")
        
        # Fallback to local path
        if not video_path:
            # Try local compressed video
            local_compressed = os.path.join('video_processing_outputs', 'compressed', video_id, 'video.mp4')
            if os.path.exists(local_compressed):
                video_path = local_compressed
                logger.info(f"Using local compressed video: {local_compressed}")
            else:
                # Try database file_path
                file_path = video_record.get('file_path')
                if file_path and os.path.exists(file_path):
                    video_path = file_path
                else:
                    # Try uploads folder
                    uploads_path = os.path.join(UPLOAD_FOLDER, video_id, 'video.mp4')
                    if os.path.exists(uploads_path):
                        video_path = uploads_path
        
        if not video_path or not os.path.exists(video_path):
            return jsonify({'error': 'Video file not found'}), 404
        
        # Generate clip
        from event_clip_generator import EventClipGenerator
        clip_generator = EventClipGenerator()
        clip_path = clip_generator.extract_clip(
            video_path, start_time, end_time, event_id, video_id
        )
        
        if not clip_path or not os.path.exists(clip_path):
            return jsonify({'error': 'Failed to generate clip'}), 500
        
        # Serve the clip
        return send_file(clip_path, mimetype='video/mp4')
        
    except Exception as e:
        logger.error(f"Error generating event clip: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/event/clip/<event_id>/download', methods=['GET'])
def download_event_clip(event_id):
    """
    Download event clip
    """
    try:
        if not DATABASE_ENABLED:
            return jsonify({'error': 'Database not enabled'}), 500
        
        # Get event from database (using singular 'event' collection)
        from bson.objectid import ObjectId
        events_collection = db_video_service.db_manager.db.event
        # Try _id first (ObjectId), fallback to event_id field
        try:
            event = events_collection.find_one({"_id": ObjectId(event_id)})
        except:
            event = events_collection.find_one({"event_id": event_id})
        
        if not event:
            return jsonify({'error': 'Event not found'}), 404
        
        video_id = event.get('video_id')
        start_timestamp_ms = int(event.get('start_timestamp_ms', 0))
        end_timestamp_ms = int(event.get('end_timestamp_ms', 0))
        
        start_time = start_timestamp_ms / 1000.0
        end_time = end_timestamp_ms / 1000.0
        
        # Get video path (same logic as get_event_clip)
        video_record = db_video_service.video_repo.get_video_by_id(video_id)
        if not video_record:
            return jsonify({'error': 'Video not found'}), 404
        
        video_path = None
        minio_key = video_record.get('minio_object_key')
        if minio_key:
            try:
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_path = temp_file.name
                temp_file.close()
                
                db_video_service.video_repo.minio.fget_object(
                    video_record.get('minio_bucket', db_video_service.video_repo.video_bucket),
                    minio_key,
                    temp_path
                )
                video_path = temp_path
            except Exception as e:
                logger.warning(f"Could not get video from MinIO: {e}")
        
        if not video_path:
            # Try local compressed video
            local_compressed = os.path.join('video_processing_outputs', 'compressed', video_id, 'video.mp4')
            if os.path.exists(local_compressed):
                video_path = local_compressed
                logger.info(f"Using local compressed video: {local_compressed}")
            else:
                # Try database file_path
                file_path = video_record.get('file_path')
                if file_path and os.path.exists(file_path):
                    video_path = file_path
                else:
                    # Try uploads folder
                    uploads_path = os.path.join(UPLOAD_FOLDER, video_id, 'video.mp4')
                    if os.path.exists(uploads_path):
                        video_path = uploads_path
        
        if not video_path or not os.path.exists(video_path):
            return jsonify({'error': 'Video file not found'}), 404
        
        # Generate clip
        from event_clip_generator import EventClipGenerator
        clip_generator = EventClipGenerator()
        clip_path = clip_generator.extract_clip(
            video_path, start_time, end_time, event_id, video_id
        )
        
        if not clip_path or not os.path.exists(clip_path):
            return jsonify({'error': 'Failed to generate clip'}), 500
        
        # Serve as download
        return send_file(clip_path, mimetype='video/mp4', as_attachment=True, 
                        download_name=f"event_{event_id}_clip.mp4")
        
    except Exception as e:
        logger.error(f"Error downloading event clip: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/face-image/<face_id>')
def get_face_image(face_id):
    """
    Serve face images for the search results.
    """
    try:
        # Construct face image path using absolute path
        # BASE_DIR is project root, so model/faces should be at project root
        # Try project root first
        face_image_path = os.path.join(BASE_DIR, 'model', 'faces', f"{face_id}.jpg")
        if not os.path.exists(face_image_path):
            # Fallback to backend/model/faces (if model is in backend directory)
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            face_image_path = os.path.join(backend_dir, 'model', 'faces', f"{face_id}.jpg")
        if not os.path.exists(face_image_path):
            # Final fallback to relative path from current working directory
            face_image_path = os.path.join('model', 'faces', f"{face_id}.jpg")
        
        if not os.path.exists(face_image_path):
            # Return a placeholder or 404
            return jsonify({'error': 'Face image not found'}), 404
        
        return send_file(face_image_path, mimetype='image/jpeg')
        
    except Exception as e:
        logger.error(f"Error serving face image {face_id}: {e}")
        return jsonify({'error': 'Error serving face image'}), 500

@app.route("/api/search/captions", methods=["POST"])
# @require_feature('nlp_search')  # Pro plan feature - Temporarily disabled for development
def search_nlp_captions():
    """Search captions using sentence-transformer embeddings + cosine similarity.
    
    Searches both:
      - event_description: behavior-level captions (e.g., "Accident behavior detected")
      - video_captions: frame-level BLIP captions (e.g., "a car is parked in a parking lot")
    
    Requires: Pro plan (nlp_search feature)
    """
    try:
        if not CAPTION_SEARCH_AVAILABLE:
            return jsonify({
                "error": "Caption search not available",
                "message": "Caption search module not installed or not available"
            }), 503
        
        data = request.json or {}
        query_text = data.get("query", "").strip()
        top_k = data.get("top_k", 10)
        min_score = data.get("min_score", 0.0)
        
        if not query_text:
            return jsonify({"error": "query is required"}), 400
        
        # Use query_retrieval.py logic for consistent results
        try:
            from nlp_search.query_retreival import retrieve_by_threshold
            
            # Connect to MongoDB using existing database service
            if DATABASE_ENABLED and db_video_service and db_video_service.db_manager:
                db = db_video_service.db_manager.db
            else:
                return jsonify({
                    "error": "Database not available",
                    "message": "Cannot connect to MongoDB for search"
                }), 503
            
            # Use a lower default threshold (0.3) to catch semantic matches
            # e.g., "car" matching "a car is parked in a parking lot" at ~0.45
            threshold = max(min_score, 0.3) if min_score > 0 else 0.3
            
            # Perform search using query_retrieval logic (searches both collections)
            results = retrieve_by_threshold(db, query_text, threshold=threshold)
            
            # Limit results to top_k
            if top_k and len(results) > top_k:
                results = results[:top_k]
                
        except Exception as e:
            logger.error(f"Error using query_retrieval search: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return jsonify({
                "error": "Search functionality unavailable",
                "message": f"NLP search module error: {str(e)}"
            }), 503
        
        # Format results for frontend
        formatted_results = []
        
        for result in results:
            source = result.get("source", "event_description")
            video_ref = result.get("video_reference") or {}
            image_url = None
            video_id = result.get("video_id")
            
            if source == "video_captions":
                # Build keyframe URL using the MinIO proxy path.
                # The keyframe_repository saves at: {video_id}/frame_XXXXXX.jpg
                # The Next.js /api/minio/image/[bucket]/[...path] proxy already works.
                frame_id = result.get("frame_id")
                if not frame_id:
                    caption_id = result.get("description_id")
                    if caption_id:
                        vc_doc = db.video_captions.find_one(
                            {"caption_id": caption_id}, {"frame_id": 1}
                        )
                        if vc_doc:
                            frame_id = vc_doc.get("frame_id")
                
                if video_id and frame_id:
                    # Use the MinIO proxy URL pattern (works through Next.js)
                    image_url = f"/api/minio/image/detectifai-keyframes/{video_id}/{frame_id}.jpg"
                        
            elif video_ref and isinstance(video_ref, dict):
                object_name = video_ref.get("object_name", "")
                bucket = video_ref.get("bucket", "nlp-images")
                if object_name and bucket:
                    image_url = f"/api/minio/image/{bucket}/{object_name}"
            
            formatted_result = {
                "id": result.get("description_id"),
                "event_id": result.get("event_id"),
                "video_id": video_id,
                "description": result.get("caption", ""),
                "caption": result.get("caption", ""),
                "confidence": result.get("similarity", 0.0),
                "similarity_score": result.get("similarity", 0.0),
                "thumbnail": image_url,
                "video_reference": video_ref if video_ref else None,
                "start_timestamp_ms": result.get("start_timestamp_ms"),
                "end_timestamp_ms": result.get("end_timestamp_ms"),
                "timestamp": result.get("start_timestamp_ms"),
                "zone": "N/A",
                "source": source
            }
            formatted_results.append(formatted_result)
        
        return jsonify({
            "query": query_text,
            "results": formatted_results,
            "total_results": len(formatted_results),
            "threshold_used": threshold if 'threshold' in locals() else min_score
        })
        
    except Exception as e:
        logger.error(f"Error in caption search: {e}")
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

# ====== LIVE STREAM ENDPOINTS ======

@app.route('/api/live/start', methods=['POST'])
def start_live_stream():
    """Start live stream processing from webcam"""
    try:
        data = request.json or {}
        camera_id = data.get('camera_id', 'webcam_01')
        camera_index = data.get('camera_index', 0)  # 0 = default webcam
        
        from live_stream_processor import get_live_processor
        
        processor = get_live_processor(camera_id, get_security_focused_config())
        
        if processor.is_processing:
            return jsonify({
                'success': False,
                'error': f'Live stream already running for camera {camera_id}'
            }), 400
        
        # Just mark as ready - actual processing happens in feed endpoint
        processor.camera_index = camera_index
        
        return jsonify({
            'success': True,
            'camera_id': camera_id,
            'message': 'Live stream ready',
            'video_feed_url': f'/api/live/feed/{camera_id}'
        })
        
    except Exception as e:
        logger.error(f"Error starting live stream: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/live/feed/<camera_id>')
def live_video_feed(camera_id):
    """Video feed endpoint for live stream - streams frames directly"""
    logger.info(f"🎬 ===== VIDEO FEED REQUESTED ===== camera_id: {camera_id}")
    try:
        from live_stream_processor import get_live_processor
        
        processor = get_live_processor(camera_id)
        camera_index = getattr(processor, 'camera_index', 0)
        
        logger.info(f"📹 Video feed requested for camera {camera_id} (index {camera_index})")
        logger.info(f"📹 Processor is_processing: {processor.is_processing}")
        logger.info(f"📹 Processor camera_index attribute: {getattr(processor, 'camera_index', 'NOT SET')}")
        
        # The generate_frames generator will handle the camera and processing
        # This runs in the same thread as the Flask response
        def generate():
            frame_count = 0
            try:
                logger.info(f"🎬 Starting frame generation for {camera_id}")
                for frame_data in processor.generate_frames(camera_index):
                    frame_count += 1
                    if frame_count % 30 == 0:  # Log every 30 frames
                        logger.info(f"📹 Streaming frame {frame_count} for {camera_id}")
                    yield frame_data
            except Exception as gen_error:
                logger.error(f"❌ Error in frame generator: {gen_error}")
                import traceback
                logger.error(traceback.format_exc())
                # Yield an error frame
                try:
                    error_frame = processor._create_error_frame(f"Stream error: {str(gen_error)}")
                    import cv2
                    ret, buffer = cv2.imencode('.jpg', error_frame)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                except Exception as frame_error:
                    logger.error(f"❌ Could not create error frame: {frame_error}")
        
        return Response(
            generate(),
            mimetype='multipart/x-mixed-replace; boundary=frame',
            headers={
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0',
                'X-Accel-Buffering': 'no',  # Disable buffering for nginx
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',  # CORS header
                'Access-Control-Allow-Methods': 'GET',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error in video feed endpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/live/stop/<camera_id>', methods=['POST'])
def stop_live_stream(camera_id):
    """Stop live stream processing"""
    try:
        from live_stream_processor import stop_live_processor
        
        stop_live_processor(camera_id)
        
        return jsonify({
            'success': True,
            'message': f'Live stream stopped for camera {camera_id}'
        })
        
    except Exception as e:
        logger.error(f"Error stopping live stream: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/live/stats/<camera_id>', methods=['GET'])
def get_live_stats(camera_id):
    """Get live stream processing statistics"""
    try:
        from live_stream_processor import get_live_processor
        
        processor = get_live_processor(camera_id)
        stats = processor.get_stats()
        
        return jsonify({
            'success': True,
            'camera_id': camera_id,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error getting live stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/live/test-camera', methods=['GET'])
def test_camera():
    """Test if camera is available - helps debug camera issues"""
    try:
        import cv2
        
        camera_index = int(request.args.get('index', 0))
        
        logger.info(f"🔍 Testing camera {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            return jsonify({
                'success': False,
                'available': False,
                'camera_index': camera_index,
                'message': f'Camera {camera_index} could not be opened. Make sure the camera is connected and not in use by another application.'
            }), 200
        
        # Try to read a frame
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            return jsonify({
                'success': True,
                'available': True,
                'camera_index': camera_index,
                'message': f'Camera {camera_index} is working correctly',
                'frame_size': f'{frame.shape[1]}x{frame.shape[0]}',
                'frame_channels': frame.shape[2] if len(frame.shape) > 2 else 1
            })
        else:
            return jsonify({
                'success': False,
                'available': False,
                'camera_index': camera_index,
                'message': f'Camera {camera_index} opened but cannot read frames. The camera may be in use or not functioning properly.'
            }), 200
        
    except Exception as e:
        logger.error(f"Error testing camera: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'available': False,
            'error': str(e),
            'message': f'Error testing camera: {str(e)}'
        }), 500

# ========================================
# Register Subscription Routes Blueprint
# ========================================
try:
    from subscription_routes import subscription_bp
    app.register_blueprint(subscription_bp)
    logger.info("✅ Subscription routes registered successfully")
except Exception as e:
    logger.error(f"❌ Failed to register subscription routes: {e}")

# ========================================
# Register Real-Time Alert Routes Blueprint
# ========================================
try:
    from alert_routes import alert_bp
    app.register_blueprint(alert_bp)
    logger.info("✅ Real-time alert routes registered successfully")
except Exception as e:
    logger.error(f"❌ Failed to register alert routes: {e}")

if __name__ == '__main__':
    logger.info("Starting DetectifAI Flask API server...")
    app.run(host='0.0.0.0', port=5000, debug=True)