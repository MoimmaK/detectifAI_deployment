"""
Subscription Middleware - Feature Gating & Usage Limits
Enforces subscription plan restrictions and tracks usage
"""

from functools import wraps
from flask import request, jsonify
from datetime import datetime
from bson import ObjectId
import logging

logger = logging.getLogger(__name__)


class SubscriptionMiddleware:
    """Middleware for subscription-based feature gating"""
    
    def __init__(self, db):
        """
        Initialize middleware with database connection
        
        Args:
            db: MongoDB database instance
        """
        self.db = db
        self.user_subscriptions = db['user_subscriptions']
        self.subscription_plans = db['subscription_plans']
        self.subscription_usage = db['subscription_usage']
        
    
    def get_user_subscription(self, user_id):
        """
        Get active subscription for a user
        
        Args:
            user_id: User's unique identifier (could be database user_id or Google ID)
            
        Returns:
            dict: Subscription document or None
        """
        try:
            logger.info(f"🔍 get_user_subscription: Looking for subscription with user_id: {user_id}")
            
            # First, try direct lookup with the provided user_id
            subscription = self.user_subscriptions.find_one({
                'user_id': user_id,
                'status': 'active',
                'current_period_end': {'$gte': datetime.utcnow()}
            })
            
            if subscription:
                logger.info(f"✅ Found active subscription with direct user_id lookup: {subscription.get('subscription_id')}")
                # Get plan details
                plan = self.subscription_plans.find_one({
                    'plan_id': subscription.get('plan_id')
                })
                
                if plan:
                    subscription['plan_details'] = plan
                else:
                    logger.warning(f"⚠️ Plan not found for plan_id: {subscription.get('plan_id')}")
                return subscription
            
            # If not found, try to find the user in the users collection and get their actual user_id
            # This handles the case where user_id might be a Google ID instead of database user_id
            logger.info(f"⚠️ No subscription found with user_id {user_id}, trying to find user in database...")
            try:
                users_collection = self.db['users']
                user_doc = users_collection.find_one({'user_id': user_id})
                if not user_doc:
                    # Try finding by Google ID if user_id looks like a Google ID (numeric string)
                    if user_id and isinstance(user_id, str) and user_id.isdigit():
                        logger.info(f"🔍 user_id looks like Google ID, searching by google_id...")
                        user_doc = users_collection.find_one({'google_id': user_id})
                
                if user_doc:
                    actual_user_id = user_doc.get('user_id')
                    logger.info(f"✅ Found user in database, actual user_id: {actual_user_id}")
                    
                    # Now try to find subscription with the actual user_id
                    subscription = self.user_subscriptions.find_one({
                        'user_id': actual_user_id,
                        'status': 'active',
                        'current_period_end': {'$gte': datetime.utcnow()}
                    })
                    
                    if subscription:
                        logger.info(f"✅ Found active subscription with actual user_id: {subscription.get('subscription_id')}")
                        # Get plan details
                        plan = self.subscription_plans.find_one({
                            'plan_id': subscription.get('plan_id')
                        })
                        
                        if plan:
                            subscription['plan_details'] = plan
                        return subscription
                    else:
                        logger.warning(f"⚠️ No active subscription found for actual user_id: {actual_user_id}")
                else:
                    logger.warning(f"⚠️ User not found in database with user_id or google_id: {user_id}")
            except Exception as e:
                logger.error(f"❌ Error looking up user: {str(e)}")
            
            # Debug: List all subscriptions for this user_id
            all_subscriptions = list(self.user_subscriptions.find({'user_id': user_id}))
            logger.info(f"📊 Found {len(all_subscriptions)} total subscription(s) for user_id {user_id}")
            for sub in all_subscriptions:
                logger.info(f"  - Subscription ID: {sub.get('subscription_id')}, Status: {sub.get('status')}, Plan: {sub.get('plan_id')}")
                    
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting user subscription: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    
    def get_user_plan_name(self, user_id):
        """
        Get user's plan name (basic, pro, or free)
        
        Args:
            user_id: User's unique identifier
            
        Returns:
            str: Plan name ('free', 'detectifai_basic', 'detectifai_pro')
        """
        subscription = self.get_user_subscription(user_id)
        
        if not subscription:
            return 'free'
        
        return subscription.get('plan_id', 'free')
    
    
    def check_feature_access(self, user_id, feature_name):
        """
        Check if user has access to a specific feature
        
        Args:
            user_id: User's unique identifier
            feature_name: Feature to check (e.g., 'behavior_analysis', 'nlp_search')
            
        Returns:
            bool: True if user has access, False otherwise
        """
        subscription = self.get_user_subscription(user_id)
        
        # Free tier - no features
        if not subscription:
            return False
        
        plan_details = subscription.get('plan_details', {})
        features = plan_details.get('features', [])
        
        return feature_name in features
    
    
    def check_usage_limit(self, user_id, limit_type):
        """
        Check if user has exceeded their usage limit
        
        Args:
            user_id: User's unique identifier
            limit_type: Type of limit (e.g., 'video_processing', 'searches')
            
        Returns:
            dict: {'allowed': bool, 'current': int, 'limit': int, 'remaining': int}
        """
        try:
            subscription = self.get_user_subscription(user_id)
            
            # Free tier - no access
            if not subscription:
                return {
                    'allowed': False,
                    'current': 0,
                    'limit': 0,
                    'remaining': 0,
                    'message': 'Subscription required'
                }
            
            plan_details = subscription.get('plan_details', {})
            limits = plan_details.get('limits', {})
            limit_value = limits.get(limit_type, float('inf'))
            
            # Get current usage for this billing period
            usage = self.subscription_usage.find_one({
                'user_id': user_id,
                'subscription_id': str(subscription['_id']),
                'period_start': subscription.get('current_period_start'),
                'period_end': subscription.get('current_period_end')
            })
            
            current_usage = 0
            if usage:
                current_usage = usage.get('usage', {}).get(limit_type, 0)
            
            allowed = current_usage < limit_value
            remaining = max(0, limit_value - current_usage)
            
            return {
                'allowed': allowed,
                'current': current_usage,
                'limit': limit_value,
                'remaining': remaining,
                'message': 'OK' if allowed else f'{limit_type} limit exceeded'
            }
            
        except Exception as e:
            logger.error(f"Error checking usage limit: {str(e)}")
            return {
                'allowed': False,
                'current': 0,
                'limit': 0,
                'remaining': 0,
                'message': f'Error: {str(e)}'
            }
    
    
    def increment_usage(self, user_id, limit_type, amount=1):
        """
        Increment usage counter for a user
        
        Args:
            user_id: User's unique identifier
            limit_type: Type of usage to increment
            amount: Amount to increment by (default: 1)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            subscription = self.get_user_subscription(user_id)
            
            if not subscription:
                return False
            
            # Upsert usage document
            self.subscription_usage.update_one(
                {
                    'user_id': user_id,
                    'subscription_id': str(subscription['_id']),
                    'period_start': subscription.get('current_period_start'),
                    'period_end': subscription.get('current_period_end')
                },
                {
                    '$inc': {f'usage.{limit_type}': amount},
                    '$set': {
                        'last_updated': datetime.utcnow()
                    },
                    '$setOnInsert': {
                        'user_id': user_id,
                        'subscription_id': str(subscription['_id']),
                        'plan_id': subscription.get('plan_id'),
                        'period_start': subscription.get('current_period_start'),
                        'period_end': subscription.get('current_period_end'),
                        'created_at': datetime.utcnow()
                    }
                },
                upsert=True
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error incrementing usage: {str(e)}")
            return False
    
    
    def get_usage_summary(self, user_id):
        """
        Get complete usage summary for a user
        
        Args:
            user_id: User's unique identifier
            
        Returns:
            dict: Usage statistics and limits
        """
        try:
            subscription = self.get_user_subscription(user_id)
            
            if not subscription:
                return {
                    'has_subscription': False,
                    'plan': 'free',
                    'message': 'No active subscription'
                }
            
            plan_details = subscription.get('plan_details', {})
            limits = plan_details.get('limits', {})
            
            # Get current usage
            usage_doc = self.subscription_usage.find_one({
                'user_id': user_id,
                'subscription_id': str(subscription['_id']),
                'period_start': subscription.get('current_period_start'),
                'period_end': subscription.get('current_period_end')
            })
            
            current_usage = {}
            if usage_doc:
                current_usage = usage_doc.get('usage', {})
            
            # Build summary
            summary = {
                'has_subscription': True,
                'plan': subscription.get('plan_id'),
                'plan_name': plan_details.get('plan_name'),
                'status': subscription.get('status'),
                'period_start': subscription.get('current_period_start'),
                'period_end': subscription.get('current_period_end'),
                'usage': {},
                'limits': limits
            }
            
            # Calculate remaining for each limit
            for limit_type, limit_value in limits.items():
                used = current_usage.get(limit_type, 0)
                summary['usage'][limit_type] = {
                    'used': used,
                    'limit': limit_value,
                    'remaining': max(0, limit_value - used),
                    'percentage': (used / limit_value * 100) if limit_value > 0 else 0
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting usage summary: {str(e)}")
            return {
                'has_subscription': False,
                'error': str(e)
            }


# Decorator for requiring subscription
def require_subscription(plan_required=None):
    """
    Decorator to require active subscription for endpoint
    
    Args:
        plan_required: Minimum plan required ('basic' or 'pro'), None for any plan
        
    Usage:
        @app.route('/api/process-video')
        @require_subscription('basic')
        def process_video():
            ...
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            from flask import current_app
            
            # Get user_id from request (adjust based on your auth)
            user_id = request.args.get('user_id')
            
            # Try getting from form data (for multipart/form-data)
            if not user_id:
                user_id = request.form.get('user_id')
                
            # Try getting from JSON if not found yet (silent=True prevents 415 error)
            if not user_id:
                try:
                    json_data = request.get_json(silent=True)
                    if json_data:
                        user_id = json_data.get('user_id')
                except Exception:
                    pass
            
            if not user_id:
                logger.warning("⚠️ require_subscription: user_id not found in request")
                return jsonify({
                    'success': False,
                    'error': 'user_id required'
                }), 401
            
            logger.info(f"🔍 require_subscription: Checking subscription for user_id: {user_id} (type: {type(user_id).__name__})")
            
            # Initialize middleware
            db = current_app.config.get('DETECTIFAI_DB')
            if db is None:
                logger.error("❌ DETECTIFAI_DB not found in app config")
                return jsonify({
                    'success': False,
                    'error': 'Database configuration error'
                }), 500
            
            middleware = SubscriptionMiddleware(db)
            
            # If user_id looks like a Google ID (all numeric), try to find the actual database user_id
            actual_user_id = user_id
            if user_id and isinstance(user_id, str) and user_id.isdigit():
                logger.info(f"🔍 user_id appears to be a Google ID, looking up actual user_id...")
                try:
                    users_collection = db['users']
                    user_doc = users_collection.find_one({'google_id': user_id})
                    if user_doc:
                        actual_user_id = user_doc.get('user_id')
                        logger.info(f"✅ Found user, actual user_id: {actual_user_id}")
                    else:
                        # Also try by user_id in case it's already the database ID
                        user_doc = users_collection.find_one({'user_id': user_id})
                        if user_doc:
                            actual_user_id = user_id
                            logger.info(f"✅ User found with provided user_id")
                except Exception as e:
                    logger.error(f"❌ Error looking up user: {str(e)}")
            
            # Check subscription with actual_user_id
            subscription = middleware.get_user_subscription(actual_user_id)
            
            if not subscription:
                logger.warning(f"⚠️ require_subscription: No active subscription found for user_id: {user_id}")
                # Check if user exists at all
                try:
                    users_collection = db['users']
                    user_exists = users_collection.find_one({'user_id': user_id})
                    if not user_exists:
                        # Try finding by email or other identifier
                        logger.warning(f"⚠️ User with user_id {user_id} not found in users collection")
                except Exception as e:
                    logger.error(f"❌ Error checking user existence: {str(e)}")
                
                return jsonify({
                    'success': False,
                    'error': 'Active subscription required',
                    'message': 'Please subscribe to a plan to access this feature',
                    'upgrade_url': '/pricing',
                    'user_id_received': user_id
                }), 403
            
            logger.info(f"✅ require_subscription: Active subscription found for user_id: {user_id}, plan: {subscription.get('plan_id')}")
            
            # Check plan level if specified
            if plan_required:
                plan_id = subscription.get('plan_id', '')
                
                # Define plan hierarchy
                plan_hierarchy = {
                    'detectifai_basic': 1,
                    'detectifai_pro': 2
                }
                
                required_level = plan_hierarchy.get(f'detectifai_{plan_required}', 0)
                user_level = plan_hierarchy.get(plan_id, 0)
                
                if user_level < required_level:
                    return jsonify({
                        'success': False,
                        'error': f'{plan_required.title()} plan required',
                        'message': f'This feature requires {plan_required.title()} or higher plan',
                        'current_plan': plan_id,
                        'required_plan': f'detectifai_{plan_required}',
                        'upgrade_url': '/pricing'
                    }), 403
            
            # Add subscription to request context
            request.subscription = subscription
            request.subscription_middleware = middleware
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


# Decorator for requiring specific feature
def require_feature(feature_name):
    """
    Decorator to require specific feature access
    
    Args:
        feature_name: Feature required (e.g., 'behavior_analysis', 'nlp_search')
        
    Usage:
        @app.route('/api/behavior-analysis')
        @require_feature('behavior_analysis')
        def behavior_analysis():
            ...
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            from flask import current_app
            
            user_id = request.args.get('user_id')
            
            # Try getting from form data (for multipart/form-data)
            if not user_id:
                user_id = request.form.get('user_id')
                
            # Try getting from JSON if not found yet (silent=True prevents 415 error)
            if not user_id:
                try:
                    json_data = request.get_json(silent=True)
                    if json_data:
                        user_id = json_data.get('user_id')
                except Exception:
                    pass
            
            if not user_id:
                return jsonify({
                    'success': False,
                    'error': 'user_id required'
                }), 401
            
            db = current_app.config.get('DETECTIFAI_DB')
            middleware = SubscriptionMiddleware(db)
            
            # Check feature access
            has_access = middleware.check_feature_access(user_id, feature_name)
            
            if not has_access:
                subscription = middleware.get_user_subscription(user_id)
                current_plan = subscription.get('plan_id') if subscription else 'free'
                
                return jsonify({
                    'success': False,
                    'error': f'Feature not available: {feature_name}',
                    'message': f'Your {current_plan} plan does not include {feature_name}',
                    'current_plan': current_plan,
                    'upgrade_url': '/pricing'
                }), 403
            
            request.subscription_middleware = middleware
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


# Decorator for checking usage limits
def check_usage_limit(limit_type, auto_increment=True):
    """
    Decorator to check and optionally increment usage limits
    
    Args:
        limit_type: Type of limit to check (e.g., 'video_processing')
        auto_increment: Whether to automatically increment counter (default: True)
        
    Usage:
        @app.route('/api/process-video')
        @require_subscription()
        @check_usage_limit('video_processing')
        def process_video():
            ...
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            from flask import current_app
            
            user_id = request.args.get('user_id')
            
            # Try getting from form data (for multipart/form-data)
            if not user_id:
                user_id = request.form.get('user_id')
                
            # Try getting from JSON if not found yet (silent=True prevents 415 error)
            if not user_id:
                try:
                    json_data = request.get_json(silent=True)
                    if json_data:
                        user_id = json_data.get('user_id')
                except Exception:
                    pass
            
            if not user_id:
                return jsonify({
                    'success': False,
                    'error': 'user_id required'
                }), 401
            
            db = current_app.config.get('DETECTIFAI_DB')
            middleware = SubscriptionMiddleware(db)
            
            # Check limit
            limit_check = middleware.check_usage_limit(user_id, limit_type)
            
            if not limit_check['allowed']:
                return jsonify({
                    'success': False,
                    'error': 'Usage limit exceeded',
                    'message': limit_check['message'],
                    'usage': {
                        'current': limit_check['current'],
                        'limit': limit_check['limit'],
                        'remaining': limit_check['remaining']
                    },
                    'upgrade_url': '/pricing'
                }), 429  # Too Many Requests
            
            # Execute function
            result = f(*args, **kwargs)
            
            # Auto-increment if successful and enabled
            if auto_increment:
                # Check if response indicates success
                if isinstance(result, tuple):
                    response_data, status_code = result[0], result[1]
                else:
                    response_data = result
                    status_code = 200
                
                # Only increment on success
                if status_code < 400:
                    middleware.increment_usage(user_id, limit_type)
            
            return result
        
        return decorated_function
    return decorator
