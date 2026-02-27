"""
Stripe Service Module for DetectifAI Subscription Management

This module provides a wrapper around Stripe API for:
- Creating checkout sessions
- Managing customer subscriptions
- Handling subscription updates
- Processing webhooks
"""

import stripe
import os
from typing import Dict, Optional, List
from datetime import datetime
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

# Configure Stripe
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')

class StripeService:
    """Service class for Stripe payment and subscription management"""
    
    def __init__(self):
        self.stripe_secret_key = os.getenv('STRIPE_SECRET_KEY')
        self.webhook_secret = os.getenv('STRIPE_WEBHOOK_SECRET')
        self.frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')
        
        # Product IDs
        self.basic_product_id = os.getenv('STRIPE_BASIC_PRODUCT_ID')
        self.pro_product_id = os.getenv('STRIPE_PRO_PRODUCT_ID')
        
        # Price IDs
        self.price_ids = {
            'basic_monthly': os.getenv('STRIPE_BASIC_MONTHLY_PRICE_ID'),
            'basic_yearly': os.getenv('STRIPE_BASIC_YEARLY_PRICE_ID'),
            'pro_monthly': os.getenv('STRIPE_PRO_MONTHLY_PRICE_ID'),
            'pro_yearly': os.getenv('STRIPE_PRO_YEARLY_PRICE_ID'),
        }
        
        # Validate configuration
        if not self.stripe_secret_key:
            raise ValueError("STRIPE_SECRET_KEY not set in environment variables")
        
        logger.info("✅ Stripe service initialized successfully")
    
    def create_checkout_session(
        self,
        user_id: str,
        user_email: str,
        price_id: str,
        plan_name: str,
        billing_period: str
    ) -> Dict:
        """
        Create a Stripe Checkout session for subscription
        
        Args:
            user_id: DetectifAI user ID
            user_email: User's email address
            price_id: Stripe price ID
            plan_name: Plan name (Basic or Pro)
            billing_period: Billing period (monthly or yearly)
        
        Returns:
            Dictionary with checkout session details
        """
        try:
            # Create checkout session
            checkout_session = stripe.checkout.Session.create(
                customer_email=user_email,
                payment_method_types=['card'],
                line_items=[{
                    'price': price_id,
                    'quantity': 1,
                }],
                mode='subscription',
                success_url=f'{self.frontend_url}/dashboard/subscription?success=true&session_id={{CHECKOUT_SESSION_ID}}',
                cancel_url=f'{self.frontend_url}/pricing?canceled=true',
                metadata={
                    'user_id': user_id,
                    'plan_name': plan_name,
                    'billing_period': billing_period
                },
                subscription_data={
                    'metadata': {
                        'user_id': user_id,
                        'plan_name': plan_name,
                        'billing_period': billing_period
                    }
                }
            )
            
            logger.info(f"✅ Created checkout session for user {user_id}: {checkout_session.id}")
            
            return {
                'session_id': checkout_session.id,
                'url': checkout_session.url,
                'status': 'created'
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"❌ Stripe error creating checkout session: {str(e)}")
            raise Exception(f"Failed to create checkout session: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Error creating checkout session: {str(e)}")
            raise
    
    def create_customer_portal_session(
        self,
        customer_id: str,
        return_url: Optional[str] = None
    ) -> Dict:
        """
        Create a customer portal session for subscription management
        
        Args:
            customer_id: Stripe customer ID
            return_url: URL to return to after portal session
        
        Returns:
            Dictionary with portal session URL
        """
        try:
            if not return_url:
                return_url = f'{self.frontend_url}/dashboard/subscription'
            
            portal_session = stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url=return_url,
            )
            
            logger.info(f"✅ Created portal session for customer {customer_id}")
            
            return {
                'url': portal_session.url,
                'status': 'created'
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"❌ Stripe error creating portal session: {str(e)}")
            raise Exception(f"Failed to create portal session: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Error creating portal session: {str(e)}")
            raise
    
    def get_subscription(self, subscription_id: str) -> Optional[Dict]:
        """
        Retrieve subscription details from Stripe
        
        Args:
            subscription_id: Stripe subscription ID
        
        Returns:
            Subscription details or None if not found
        """
        try:
            subscription = stripe.Subscription.retrieve(subscription_id)
            
            return {
                'id': subscription.id,
                'customer': subscription.customer,
                'status': subscription.status,
                'current_period_start': datetime.fromtimestamp(subscription.current_period_start),
                'current_period_end': datetime.fromtimestamp(subscription.current_period_end),
                'cancel_at_period_end': subscription.cancel_at_period_end,
                'canceled_at': datetime.fromtimestamp(subscription.canceled_at) if subscription.canceled_at else None,
                'plan': {
                    'id': subscription.plan.id,
                    'amount': subscription.plan.amount / 100,  # Convert from cents
                    'currency': subscription.plan.currency,
                    'interval': subscription.plan.interval,
                }
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"❌ Stripe error retrieving subscription: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"❌ Error retrieving subscription: {str(e)}")
            return None
    
    def cancel_subscription(
        self,
        subscription_id: str,
        at_period_end: bool = True
    ) -> Dict:
        """
        Cancel a subscription
        
        Args:
            subscription_id: Stripe subscription ID
            at_period_end: If True, cancel at period end; if False, cancel immediately
        
        Returns:
            Updated subscription details
        """
        try:
            if at_period_end:
                # Cancel at period end
                subscription = stripe.Subscription.modify(
                    subscription_id,
                    cancel_at_period_end=True
                )
                logger.info(f"✅ Subscription {subscription_id} will cancel at period end")
            else:
                # Cancel immediately
                subscription = stripe.Subscription.delete(subscription_id)
                logger.info(f"✅ Subscription {subscription_id} canceled immediately")
            
            return {
                'id': subscription.id,
                'status': subscription.status,
                'cancel_at_period_end': subscription.cancel_at_period_end,
                'canceled_at': datetime.fromtimestamp(subscription.canceled_at) if subscription.canceled_at else None,
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"❌ Stripe error canceling subscription: {str(e)}")
            raise Exception(f"Failed to cancel subscription: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Error canceling subscription: {str(e)}")
            raise
    
    def update_subscription(
        self,
        subscription_id: str,
        new_price_id: str
    ) -> Dict:
        """
        Update subscription to a new plan/price
        
        Args:
            subscription_id: Stripe subscription ID
            new_price_id: New Stripe price ID
        
        Returns:
            Updated subscription details
        """
        try:
            # Get current subscription
            subscription = stripe.Subscription.retrieve(subscription_id)
            
            # Update subscription with new price
            updated_subscription = stripe.Subscription.modify(
                subscription_id,
                items=[{
                    'id': subscription['items']['data'][0].id,
                    'price': new_price_id,
                }],
                proration_behavior='create_prorations',  # Prorate the change
            )
            
            logger.info(f"✅ Updated subscription {subscription_id} to price {new_price_id}")
            
            return {
                'id': updated_subscription.id,
                'status': updated_subscription.status,
                'current_period_start': datetime.fromtimestamp(updated_subscription.current_period_start),
                'current_period_end': datetime.fromtimestamp(updated_subscription.current_period_end),
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"❌ Stripe error updating subscription: {str(e)}")
            raise Exception(f"Failed to update subscription: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Error updating subscription: {str(e)}")
            raise
    
    def get_customer_subscriptions(self, customer_id: str) -> List[Dict]:
        """
        Get all subscriptions for a customer
        
        Args:
            customer_id: Stripe customer ID
        
        Returns:
            List of subscription details
        """
        try:
            subscriptions = stripe.Subscription.list(
                customer=customer_id,
                limit=10
            )
            
            return [{
                'id': sub.id,
                'status': sub.status,
                'current_period_start': datetime.fromtimestamp(sub.current_period_start),
                'current_period_end': datetime.fromtimestamp(sub.current_period_end),
                'cancel_at_period_end': sub.cancel_at_period_end,
                'plan': {
                    'id': sub.plan.id,
                    'amount': sub.plan.amount / 100,
                    'currency': sub.plan.currency,
                    'interval': sub.plan.interval,
                }
            } for sub in subscriptions.data]
            
        except stripe.error.StripeError as e:
            logger.error(f"❌ Stripe error retrieving customer subscriptions: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"❌ Error retrieving customer subscriptions: {str(e)}")
            return []
    
    def construct_webhook_event(self, payload: bytes, signature: str):
        """
        Construct and verify webhook event from Stripe
        
        Args:
            payload: Raw request body
            signature: Stripe signature header
        
        Returns:
            Verified Stripe event object
        """
        try:
            event = stripe.Webhook.construct_event(
                payload, signature, self.webhook_secret
            )
            logger.info(f"✅ Verified webhook event: {event['type']}")
            return event
            
        except ValueError as e:
            logger.error(f"❌ Invalid webhook payload: {str(e)}")
            raise
        except stripe.error.SignatureVerificationError as e:
            logger.error(f"❌ Invalid webhook signature: {str(e)}")
            raise
    
    def get_price_id(self, plan_name: str, billing_period: str) -> Optional[str]:
        """
        Get Stripe price ID for a plan and billing period
        
        Args:
            plan_name: 'basic' or 'pro'
            billing_period: 'monthly' or 'yearly'
        
        Returns:
            Stripe price ID or None if not found
        """
        key = f"{plan_name.lower()}_{billing_period.lower()}"
        return self.price_ids.get(key)


# Singleton instance
_stripe_service = None

def get_stripe_service() -> StripeService:
    """Get or create Stripe service singleton"""
    global _stripe_service
    if _stripe_service is None:
        _stripe_service = StripeService()
    return _stripe_service
