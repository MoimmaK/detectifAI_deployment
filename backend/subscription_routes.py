"""
Subscription Routes for DetectifAI Payment Management

API endpoints for:
- Creating checkout sessions
- Managing subscriptions
- Accessing customer portal
- Retrieving subscription status
- Handling webhooks
"""

from flask import Blueprint, request, jsonify
from stripe_service import get_stripe_service
from pymongo import MongoClient
from datetime import datetime, timedelta
from uuid import uuid4
import os
import logging
import json
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Create Blueprint
subscription_bp = Blueprint('subscriptions', __name__, url_prefix='/api/subscriptions')

# MongoDB connection
MONGO_URI = os.getenv('MONGO_URI')
client = MongoClient(MONGO_URI)
db = client.get_default_database()

# Collections
subscription_plans = db.subscription_plans
user_subscriptions = db.user_subscriptions
subscription_events = db.subscription_events
payment_history = db.payment_history
users = db.users

# Initialize Stripe service
stripe_service = get_stripe_service()


@subscription_bp.route('/create-checkout-session', methods=['POST'])
def create_checkout_session():
    """
    Create a Stripe Checkout session for subscription purchase
    
    Request body:
    {
        "user_id": "user123",
        "user_email": "user@example.com",
        "plan_name": "basic",  # or "pro"
        "billing_period": "monthly"  # or "yearly"
    }
    """
    try:
        data = request.json
        user_id = data.get('user_id')
        user_email = data.get('user_email')
        plan_name = data.get('plan_name', 'basic').lower()
        billing_period = data.get('billing_period', 'monthly').lower()
        
        # Validate input
        if not user_id or not user_email:
            return jsonify({'error': 'user_id and user_email are required'}), 400
        
        if plan_name not in ['basic', 'pro']:
            return jsonify({'error': 'Invalid plan_name. Must be "basic" or "pro"'}), 400
        
        if billing_period not in ['monthly', 'yearly']:
            return jsonify({'error': 'Invalid billing_period. Must be "monthly" or "yearly"'}), 400
        
        # Get price ID
        price_id = stripe_service.get_price_id(plan_name, billing_period)
        if not price_id:
            return jsonify({'error': 'Price ID not found for selected plan'}), 400
        
        # Create checkout session
        session = stripe_service.create_checkout_session(
            user_id=user_id,
            user_email=user_email,
            price_id=price_id,
            plan_name=plan_name,
            billing_period=billing_period
        )
        
        logger.info(f"✅ Created checkout session for user {user_id}: {session['session_id']}")
        
        return jsonify({
            'success': True,
            'session_id': session['session_id'],
            'url': session['url']
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error creating checkout session: {str(e)}")
        return jsonify({'error': str(e)}), 500


@subscription_bp.route('/sync-from-stripe', methods=['POST'])
def sync_subscription_from_stripe():
    """
    Manually sync subscription from Stripe (for development when webhooks don't reach localhost)
    
    Request body:
    {
        "user_id": "user123",
        "user_email": "user@example.com" (required for finding customer)
    }
    """
    try:
        data = request.json
        user_id = data.get('user_id')
        user_email = data.get('user_email')
        
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
        
        if not user_email:
            return jsonify({'error': 'user_email is required'}), 400
        
        import stripe
        stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
        
        # Find customer by email
        customers = stripe.Customer.list(email=user_email, limit=1)
        if not customers.data:
            return jsonify({'error': f'No Stripe customer found with email: {user_email}'}), 404
        
        customer = customers.data[0]
        customer_id = customer.id
        
        logger.info(f"✅ Found customer: {customer_id} for email: {user_email}")
        
        # Get latest subscription for this customer
        subscriptions = stripe.Subscription.list(customer=customer_id, limit=1, status='active')
        if not subscriptions.data:
            # Try to get any subscription (including past_due, etc)
            subscriptions = stripe.Subscription.list(customer=customer_id, limit=1)
            if not subscriptions.data:
                return jsonify({'error': 'No subscriptions found for this customer'}), 404
        
        subscription_data = subscriptions.data[0]
        
        logger.info(f"✅ Found subscription: {subscription_data.id} with status: {subscription_data.status}")
        
        # Get plan details from subscription
        # Try multiple ways to get price info
        price_id = None
        try:
            # Method 1: Direct attribute access
            if hasattr(subscription_data, 'items') and subscription_data.items and len(subscription_data.items.data) > 0:
                price_id = subscription_data.items.data[0].price.id
                logger.info(f"✅ Got price_id via attribute: {price_id}")
        except Exception as e:
            logger.warning(f"⚠️ Attribute access failed: {e}")
        
        if not price_id:
            try:
                # Method 2: Dictionary access
                price_id = subscription_data['items']['data'][0]['price']['id']
                logger.info(f"✅ Got price_id via dict: {price_id}")
            except Exception as e:
                logger.warning(f"⚠️ Dict access failed: {e}")
        
        # Determine plan and billing period
        plan_name = 'basic'  # default
        billing_period = 'monthly'  # default
        
        if price_id:
            # Map price_id to plan
            price_ids = {
                os.getenv('STRIPE_BASIC_MONTHLY_PRICE_ID'): ('basic', 'monthly'),
                os.getenv('STRIPE_BASIC_YEARLY_PRICE_ID'): ('basic', 'yearly'),
                os.getenv('STRIPE_PRO_MONTHLY_PRICE_ID'): ('pro', 'monthly'),
                os.getenv('STRIPE_PRO_YEARLY_PRICE_ID'): ('pro', 'yearly'),
            }
            
            plan_info = price_ids.get(price_id)
            if plan_info:
                plan_name, billing_period = plan_info
        
        # Try metadata as fallback
        if hasattr(subscription_data, 'metadata'):
            plan_name = subscription_data.metadata.get('plan_name', plan_name)
            billing_period = subscription_data.metadata.get('billing_period', billing_period)
        
        logger.info(f"✅ Detected plan: {plan_name}, billing: {billing_period}")
        
        # Get plan from database
        plan = subscription_plans.find_one({
            'plan_id': f'detectifai_{plan_name}'
        })
        
        if not plan:
            logger.error(f"❌ Plan not found in database: detectifai_{plan_name}")
            return jsonify({'error': f'Plan not found: {plan_name}'}), 404
        
        # Check if subscription already exists
        existing = user_subscriptions.find_one({
            'stripe_subscription_id': subscription_data.id
        })
        
        if existing:
            # Update existing
            user_subscriptions.update_one(
                {'stripe_subscription_id': subscription_data.id},
                {
                    '$set': {
                        'user_id': user_id,  # Update user_id
                        'status': subscription_data.status,
                        'billing_period': billing_period,
                        'current_period_start': datetime.fromtimestamp(subscription_data.current_period_start) if hasattr(subscription_data, 'current_period_start') else datetime.utcnow(),
                        'current_period_end': datetime.fromtimestamp(subscription_data.current_period_end) if hasattr(subscription_data, 'current_period_end') else datetime.utcnow(),
                        'updated_at': datetime.utcnow()
                    }
                }
            )
            logger.info(f"✅ Updated existing subscription for user {user_id}")
            message = "Subscription updated successfully"
        else:
            # Delete any old subscriptions for this user first
            user_subscriptions.delete_many({'user_id': user_id})
            
            # Create new subscription
            subscription_id = str(uuid4())
            
            # Safely get timestamps
            created_timestamp = subscription_data.created if hasattr(subscription_data, 'created') else int(datetime.utcnow().timestamp())
            period_start = subscription_data.current_period_start if hasattr(subscription_data, 'current_period_start') else int(datetime.utcnow().timestamp())
            period_end = subscription_data.current_period_end if hasattr(subscription_data, 'current_period_end') else int((datetime.utcnow() + timedelta(days=30)).timestamp())
            
            user_subscriptions.insert_one({
                'subscription_id': subscription_id,
                'user_id': user_id,
                'plan_id': plan['plan_id'],
                'start_date': datetime.fromtimestamp(created_timestamp),
                'end_date': datetime.fromtimestamp(period_end),
                'stripe_customer_id': customer_id,
                'stripe_subscription_id': subscription_data.id,
                'billing_period': billing_period,
                'status': subscription_data.status,
                'current_period_start': datetime.fromtimestamp(period_start),
                'current_period_end': datetime.fromtimestamp(period_end),
                'cancel_at_period_end': subscription_data.cancel_at_period_end if hasattr(subscription_data, 'cancel_at_period_end') else False,
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            })
            logger.info(f"✅ Created subscription for user {user_id}")
            message = "Subscription synced successfully"
        
        return jsonify({
            'success': True,
            'message': message,
            'subscription': {
                'subscription_id': subscription_data.id,
                'status': subscription_data.status,
                'plan': plan['plan_name'],
                'billing_period': billing_period,
                'customer_email': user_email
            }
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error syncing subscription: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@subscription_bp.route('/create-portal-session', methods=['POST'])
def create_portal_session():
    """
    Create a Stripe Customer Portal session for subscription management
    
    Request body:
    {
        "user_id": "user123"
    }
    """
    try:
        data = request.json
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
        
        # Get user's subscription
        subscription = user_subscriptions.find_one({'user_id': user_id})
        if not subscription:
            return jsonify({'error': 'No subscription found for user'}), 404
        
        stripe_customer_id = subscription.get('stripe_customer_id')
        if not stripe_customer_id:
            return jsonify({'error': 'No Stripe customer ID found'}), 400
        
        # Create portal session
        portal_session = stripe_service.create_customer_portal_session(
            customer_id=stripe_customer_id
        )
        
        logger.info(f"✅ Created portal session for user {user_id}")
        
        return jsonify({
            'success': True,
            'url': portal_session['url']
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error creating portal session: {str(e)}")
        return jsonify({'error': str(e)}), 500


@subscription_bp.route('/status', methods=['GET'])
def get_subscription_status():
    """
    Get current subscription status for a user
    
    Query params:
    - user_id: User ID
    """
    try:
        user_id = request.args.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
        
        # Get user's subscription
        subscription = user_subscriptions.find_one({'user_id': user_id})
        
        if not subscription:
            return jsonify({
                'success': True,
                'has_subscription': False,
                'plan': None
            }), 200
        
        # Get plan details
        plan = subscription_plans.find_one({'plan_id': subscription['plan_id']})
        
        # Format subscription data
        # Handle features - check if it's already a list or needs splitting
        features = []
        if plan:
            plan_features = plan.get('features', '')
            if isinstance(plan_features, list):
                features = plan_features
            elif isinstance(plan_features, str):
                features = plan_features.split(',') if plan_features else []
        
        subscription_data = {
            'has_subscription': True,
            'subscription_id': subscription['subscription_id'],
            'plan_name': plan['plan_name'] if plan else 'Unknown',
            'plan_id': subscription['plan_id'],
            'billing_period': subscription.get('billing_period', 'monthly'),
            'status': subscription.get('status', 'active'),
            'current_period_start': subscription.get('current_period_start').isoformat() if subscription.get('current_period_start') else None,
            'current_period_end': subscription.get('current_period_end').isoformat() if subscription.get('current_period_end') else None,
            'cancel_at_period_end': subscription.get('cancel_at_period_end', False),
            'stripe_customer_id': subscription.get('stripe_customer_id'),
            'features': features
        }
        
        logger.info(f"✅ Retrieved subscription status for user {user_id}")
        
        return jsonify({
            'success': True,
            **subscription_data
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting subscription status: {str(e)}")
        return jsonify({'error': str(e)}), 500


@subscription_bp.route('/cancel', methods=['POST'])
def cancel_subscription():
    """
    Cancel a subscription
    
    Request body:
    {
        "user_id": "user123",
        "immediate": false  # If true, cancel immediately; otherwise at period end
    }
    """
    try:
        data = request.json
        user_id = data.get('user_id')
        immediate = data.get('immediate', False)
        
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
        
        # Get user's subscription
        subscription = user_subscriptions.find_one({'user_id': user_id})
        if not subscription:
            return jsonify({'error': 'No subscription found for user'}), 404
        
        stripe_subscription_id = subscription.get('stripe_subscription_id')
        if not stripe_subscription_id:
            return jsonify({'error': 'No Stripe subscription ID found'}), 400
        
        # Cancel subscription
        result = stripe_service.cancel_subscription(
            subscription_id=stripe_subscription_id,
            at_period_end=not immediate
        )
        
        # Update database
        user_subscriptions.update_one(
            {'user_id': user_id},
            {
                '$set': {
                    'status': 'canceled' if immediate else 'active',
                    'cancel_at_period_end': not immediate,
                    'updated_at': datetime.utcnow()
                }
            }
        )
        
        # Log event
        subscription_events.insert_one({
            'event_id': str(uuid4()),
            'subscription_id': subscription['subscription_id'],
            'event_type': 'subscription_canceled',
            'event_data': {
                'immediate': immediate,
                'canceled_at': result.get('canceled_at').isoformat() if result.get('canceled_at') else None
            },
            'created_at': datetime.utcnow()
        })
        
        logger.info(f"✅ Canceled subscription for user {user_id} (immediate: {immediate})")
        
        return jsonify({
            'success': True,
            'message': 'Subscription canceled successfully',
            'cancel_at_period_end': not immediate
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error canceling subscription: {str(e)}")
        return jsonify({'error': str(e)}), 500


@subscription_bp.route('/upgrade', methods=['POST'])
def upgrade_subscription():
    """
    Upgrade/downgrade subscription to a different plan
    
    Request body:
    {
        "user_id": "user123",
        "new_plan_name": "pro",  # or "basic"
        "new_billing_period": "monthly"  # or "yearly"
    }
    """
    try:
        data = request.json
        user_id = data.get('user_id')
        new_plan_name = data.get('new_plan_name', 'pro').lower()
        new_billing_period = data.get('new_billing_period', 'monthly').lower()
        
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
        
        # Get user's subscription
        subscription = user_subscriptions.find_one({'user_id': user_id})
        if not subscription:
            return jsonify({'error': 'No subscription found for user'}), 404
        
        stripe_subscription_id = subscription.get('stripe_subscription_id')
        if not stripe_subscription_id:
            return jsonify({'error': 'No Stripe subscription ID found'}), 400
        
        # Get new price ID
        new_price_id = stripe_service.get_price_id(new_plan_name, new_billing_period)
        if not new_price_id:
            return jsonify({'error': 'Price ID not found for new plan'}), 400
        
        # Get new plan from database
        new_plan = subscription_plans.find_one({
            'plan_name': f'DetectifAI {new_plan_name.capitalize()}'
        })
        if not new_plan:
            return jsonify({'error': 'Plan not found in database'}), 404
        
        # Update subscription
        result = stripe_service.update_subscription(
            subscription_id=stripe_subscription_id,
            new_price_id=new_price_id
        )
        
        # Update database
        user_subscriptions.update_one(
            {'user_id': user_id},
            {
                '$set': {
                    'plan_id': new_plan['plan_id'],
                    'billing_period': new_billing_period,
                    'updated_at': datetime.utcnow()
                }
            }
        )
        
        # Log event
        subscription_events.insert_one({
            'event_id': str(uuid4()),
            'subscription_id': subscription['subscription_id'],
            'event_type': 'subscription_updated',
            'event_data': {
                'old_plan': subscription.get('plan_id'),
                'new_plan': new_plan['plan_id'],
                'new_billing_period': new_billing_period
            },
            'created_at': datetime.utcnow()
        })
        
        logger.info(f"✅ Updated subscription for user {user_id} to {new_plan_name}")
        
        return jsonify({
            'success': True,
            'message': 'Subscription updated successfully',
            'new_plan': new_plan_name,
            'billing_period': new_billing_period
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error updating subscription: {str(e)}")
        return jsonify({'error': str(e)}), 500


@subscription_bp.route('/plans', methods=['GET'])
def get_subscription_plans():
    """
    Get all available subscription plans
    """
    try:
        plans = list(subscription_plans.find({'is_active': True}, {'_id': 0}))
        
        # Format plans
        formatted_plans = []
        for plan in plans:
            # Handle features field
            plan_features = plan.get('features', '')
            if isinstance(plan_features, list):
                features = plan_features
            elif isinstance(plan_features, str):
                features = plan_features.split(',') if plan_features else []
            else:
                features = []
            
            formatted_plans.append({
                'plan_id': plan['plan_id'],
                'plan_name': plan['plan_name'],
                'description': plan.get('description', ''),
                'price': float(plan['price']),
                'features': features,
                'billing_periods': plan.get('billing_periods', ['monthly']),
                'stripe_product_id': plan.get('stripe_product_id'),
                'stripe_price_ids': plan.get('stripe_price_ids', {})
            })
        
        logger.info(f"✅ Retrieved {len(formatted_plans)} subscription plans")
        
        return jsonify({
            'success': True,
            'plans': formatted_plans
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error retrieving plans: {str(e)}")
        return jsonify({'error': str(e)}), 500


@subscription_bp.route('/webhook', methods=['POST'])
def stripe_webhook():
    """
    Handle Stripe webhook events
    """
    try:
        payload = request.data
        sig_header = request.headers.get('Stripe-Signature')
        
        # Development mode: Allow webhooks without signature if webhook secret is placeholder
        webhook_secret = os.getenv('STRIPE_WEBHOOK_SECRET', '')
        dev_mode = webhook_secret == 'whsec_xxxxx' or not webhook_secret
        
        if dev_mode:
            # Development: Parse without verification
            logger.warning("⚠️ Development mode: Webhook signature verification DISABLED")
            event = json.loads(payload)
        else:
            # Production: Verify signature
            if not sig_header:
                return jsonify({'error': 'Missing Stripe signature'}), 400
            event = stripe_service.construct_webhook_event(payload, sig_header)
        
        # Handle different event types
        event_type = event['type']
        event_data = event['data']['object']
        
        logger.info(f"📬 Received webhook event: {event_type}")
        
        if event_type == 'customer.subscription.created':
            handle_subscription_created(event_data)
        elif event_type == 'customer.subscription.updated':
            handle_subscription_updated(event_data)
        elif event_type == 'customer.subscription.deleted':
            handle_subscription_deleted(event_data)
        elif event_type == 'invoice.payment_succeeded':
            handle_payment_succeeded(event_data)
        elif event_type == 'invoice.payment_failed':
            handle_payment_failed(event_data)
        else:
            logger.info(f"ℹ️ Unhandled webhook event type: {event_type}")
        
        return jsonify({'success': True}), 200
        
    except Exception as e:
        logger.error(f"❌ Error processing webhook: {str(e)}")
        return jsonify({'error': str(e)}), 400


def handle_subscription_created(subscription_data):
    """Handle subscription.created event"""
    try:
        user_id = subscription_data['metadata'].get('user_id')
        plan_name = subscription_data['metadata'].get('plan_name')
        billing_period = subscription_data['metadata'].get('billing_period')
        
        if not user_id:
            logger.warning("⚠️ No user_id in subscription metadata")
            return
        
        # Get plan from database
        plan = subscription_plans.find_one({
            'plan_name': f'DetectifAI {plan_name.capitalize()}'
        })
        
        if not plan:
            logger.error(f"❌ Plan not found: {plan_name}")
            return
        
        # Create subscription in database
        subscription_id = str(uuid4())
        user_subscriptions.insert_one({
            'subscription_id': subscription_id,
            'user_id': user_id,
            'plan_id': plan['plan_id'],
            'start_date': datetime.utcnow(),
            'end_date': datetime.fromtimestamp(subscription_data['current_period_end']),
            'stripe_customer_id': subscription_data['customer'],
            'stripe_subscription_id': subscription_data['id'],
            'billing_period': billing_period,
            'status': subscription_data['status'],
            'current_period_start': datetime.fromtimestamp(subscription_data['current_period_start']),
            'current_period_end': datetime.fromtimestamp(subscription_data['current_period_end']),
            'cancel_at_period_end': False,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        })
        
        # Log event
        subscription_events.insert_one({
            'event_id': str(uuid4()),
            'subscription_id': subscription_id,
            'event_type': 'subscription_created',
            'stripe_event_id': subscription_data['id'],
            'event_data': {'plan_name': plan_name, 'billing_period': billing_period},
            'created_at': datetime.utcnow()
        })
        
        logger.info(f"✅ Created subscription for user {user_id}")
        
    except Exception as e:
        logger.error(f"❌ Error handling subscription created: {str(e)}")


def handle_subscription_updated(subscription_data):
    """Handle subscription.updated event"""
    try:
        stripe_subscription_id = subscription_data['id']
        
        # Update subscription in database
        user_subscriptions.update_one(
            {'stripe_subscription_id': stripe_subscription_id},
            {
                '$set': {
                    'status': subscription_data['status'],
                    'current_period_start': datetime.fromtimestamp(subscription_data['current_period_start']),
                    'current_period_end': datetime.fromtimestamp(subscription_data['current_period_end']),
                    'cancel_at_period_end': subscription_data.get('cancel_at_period_end', False),
                    'updated_at': datetime.utcnow()
                }
            }
        )
        
        logger.info(f"✅ Updated subscription {stripe_subscription_id}")
        
    except Exception as e:
        logger.error(f"❌ Error handling subscription updated: {str(e)}")


def handle_subscription_deleted(subscription_data):
    """Handle subscription.deleted event"""
    try:
        stripe_subscription_id = subscription_data['id']
        
        # Update subscription status
        user_subscriptions.update_one(
            {'stripe_subscription_id': stripe_subscription_id},
            {
                '$set': {
                    'status': 'canceled',
                    'updated_at': datetime.utcnow()
                }
            }
        )
        
        logger.info(f"✅ Marked subscription as canceled: {stripe_subscription_id}")
        
    except Exception as e:
        logger.error(f"❌ Error handling subscription deleted: {str(e)}")


def handle_payment_succeeded(invoice_data):
    """Handle invoice.payment_succeeded event"""
    try:
        customer_id = invoice_data['customer']
        amount = invoice_data['amount_paid'] / 100  # Convert from cents
        
        # Get user subscription
        subscription = user_subscriptions.find_one({'stripe_customer_id': customer_id})
        if not subscription:
            logger.warning(f"⚠️ No subscription found for customer {customer_id}")
            return
        
        # Log payment
        payment_history.insert_one({
            'payment_id': str(uuid4()),
            'user_id': subscription['user_id'],
            'stripe_payment_intent_id': invoice_data.get('payment_intent'),
            'amount': amount,
            'currency': invoice_data['currency'].upper(),
            'status': 'succeeded',
            'payment_method': 'card',
            'created_at': datetime.utcnow()
        })
        
        logger.info(f"✅ Recorded payment of ${amount} for user {subscription['user_id']}")
        
    except Exception as e:
        logger.error(f"❌ Error handling payment succeeded: {str(e)}")


def handle_payment_failed(invoice_data):
    """Handle invoice.payment_failed event"""
    try:
        customer_id = invoice_data['customer']
        
        # Get user subscription
        subscription = user_subscriptions.find_one({'stripe_customer_id': customer_id})
        if not subscription:
            logger.warning(f"⚠️ No subscription found for customer {customer_id}")
            return
        
        # Update subscription status
        user_subscriptions.update_one(
            {'stripe_customer_id': customer_id},
            {
                '$set': {
                    'status': 'past_due',
                    'updated_at': datetime.utcnow()
                }
            }
        )
        
        logger.warning(f"⚠️ Payment failed for user {subscription['user_id']}")
        
    except Exception as e:
        logger.error(f"❌ Error handling payment failed: {str(e)}")
