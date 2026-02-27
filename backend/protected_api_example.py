"""
Example API Routes with Feature Gating

This file demonstrates how to use subscription middleware
in your existing API endpoints.
"""

from flask import Blueprint, request, jsonify
from subscription_middleware import (
    require_subscription,
    require_feature,
    check_usage_limit,
    SubscriptionMiddleware
)

# Create blueprint
protected_api = Blueprint('protected_api', __name__, url_prefix='/api')


# Example 1: Basic subscription required (any plan)
@protected_api.route('/video/process', methods=['POST'])
@require_subscription()  # Requires any active subscription
@check_usage_limit('video_processing')  # Check and increment video processing limit
def process_video():
    """
    Process uploaded video - requires active subscription
    Usage is automatically tracked and enforced
    """
    user_id = request.json.get('user_id')
    video_path = request.json.get('video_path')
    
    # Your video processing logic here
    # The middleware has already verified:
    # 1. User has active subscription
    # 2. User hasn't exceeded video processing limit
    # 3. Usage counter will be incremented after successful completion
    
    return jsonify({
        'success': True,
        'message': 'Video processing started',
        'video_path': video_path
    }), 200


# Example 2: Pro plan required
@protected_api.route('/behavior-analysis', methods=['POST'])
@require_subscription('pro')  # Requires Pro plan specifically
def behavior_analysis():
    """
    Advanced behavior analysis - Pro plan only
    """
    user_id = request.json.get('user_id')
    video_id = request.json.get('video_id')
    
    # Your behavior analysis logic here
    # Middleware ensures user has Pro plan
    
    return jsonify({
        'success': True,
        'message': 'Behavior analysis complete',
        'video_id': video_id
    }), 200


# Example 3: Specific feature required
@protected_api.route('/nlp/search', methods=['POST'])
@require_feature('nlp_search')  # Requires nlp_search feature
@check_usage_limit('nlp_searches')  # Check NLP search limit
def nlp_search():
    """
    NLP-based event search - requires nlp_search feature (Pro plan)
    """
    user_id = request.json.get('user_id')
    query = request.json.get('query')
    
    # Your NLP search logic here
    # Middleware ensures:
    # 1. User's plan includes nlp_search feature
    # 2. User hasn't exceeded nlp_searches limit
    
    return jsonify({
        'success': True,
        'query': query,
        'results': []  # Your search results
    }), 200


# Example 4: Image search with limits
@protected_api.route('/search/image', methods=['POST'])
@require_feature('image_search')  # Requires image_search feature (Pro)
@check_usage_limit('image_searches')  # Check image search limit
def image_search():
    """
    Face/image-based search - Pro plan only
    """
    user_id = request.json.get('user_id')
    image_data = request.json.get('image')
    
    # Your image search logic here
    
    return jsonify({
        'success': True,
        'matches': []  # Your search results
    }), 200


# Example 5: Person tracking (Pro feature)
@protected_api.route('/person-tracking', methods=['POST'])
@require_feature('person_tracking')
def person_tracking():
    """
    Track person across video timeline - Pro plan only
    """
    user_id = request.json.get('user_id')
    person_id = request.json.get('person_id')
    
    # Your person tracking logic here
    
    return jsonify({
        'success': True,
        'person_id': person_id,
        'timeline': []  # Tracking timeline
    }), 200


# Example 6: Get usage statistics (no gating - informational)
@protected_api.route('/usage/summary', methods=['GET'])
def get_usage_summary():
    """
    Get user's current usage and limits
    """
    user_id = request.args.get('user_id')
    
    if not user_id:
        return jsonify({'success': False, 'error': 'user_id required'}), 400
    
    from flask import current_app
    db = current_app.config.get('DETECTIFAI_DB')
    middleware = SubscriptionMiddleware(db)
    
    summary = middleware.get_usage_summary(user_id)
    
    return jsonify({
        'success': True,
        'usage': summary
    }), 200


# Example 7: Check specific feature access (utility endpoint)
@protected_api.route('/feature/check', methods=['GET'])
def check_feature():
    """
    Check if user has access to specific feature
    """
    user_id = request.args.get('user_id')
    feature = request.args.get('feature')
    
    if not user_id or not feature:
        return jsonify({
            'success': False,
            'error': 'user_id and feature required'
        }), 400
    
    from flask import current_app
    db = current_app.config.get('DETECTIFAI_DB')
    middleware = SubscriptionMiddleware(db)
    
    has_access = middleware.check_feature_access(user_id, feature)
    plan_name = middleware.get_user_plan_name(user_id)
    
    return jsonify({
        'success': True,
        'feature': feature,
        'has_access': has_access,
        'current_plan': plan_name
    }), 200


# Example 8: Custom reports (Pro feature, no usage limit)
@protected_api.route('/reports/custom', methods=['POST'])
@require_feature('custom_reports')
def generate_custom_report():
    """
    Generate custom reports - Pro plan only (no usage limit)
    """
    user_id = request.json.get('user_id')
    report_config = request.json.get('config', {})
    
    # Your custom report generation logic here
    
    return jsonify({
        'success': True,
        'report_id': 'report_123',
        'config': report_config
    }), 200


# Helper function to manually check and increment usage
def manual_usage_check_example():
    """
    Example of manual usage checking (without decorator)
    Use this pattern when you need more control
    """
    from flask import current_app
    
    user_id = request.json.get('user_id')
    db = current_app.config.get('DETECTIFAI_DB')
    middleware = SubscriptionMiddleware(db)
    
    # Check limit manually
    limit_check = middleware.check_usage_limit(user_id, 'video_processing')
    
    if not limit_check['allowed']:
        return jsonify({
            'success': False,
            'error': 'Usage limit exceeded',
            'usage': limit_check
        }), 429
    
    # Do your work...
    
    # Increment manually after success
    middleware.increment_usage(user_id, 'video_processing', amount=1)
    
    return jsonify({'success': True}), 200


# Register blueprint in app.py:
# from protected_api_example import protected_api
# app.register_blueprint(protected_api)
