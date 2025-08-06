"""
Security and Authentication Middleware for Professional RAG API
"""

import jwt
import time
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from functools import wraps

from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import redis

from config import config

class SecurityManager:
    """Enhanced security manager for API authentication and rate limiting"""
    
    def __init__(self):
        self.security = HTTPBearer()
        self.redis_client = None
        try:
            # Try to connect to Redis for session management
            if hasattr(config, 'REDIS_URL'):
                import redis
                self.redis_client = redis.from_url(config.REDIS_URL)
        except:
            self.redis_client = None
        
        self.secret_key = config.JWT_SECRET_KEY or secrets.token_urlsafe(32)
        self.api_keys = self._load_api_keys()
    
    def _load_api_keys(self) -> Dict[str, Dict]:
        """Load API keys from environment or config"""
        api_keys = {}
        
        # Default admin key for testing
        if hasattr(config, 'ADMIN_API_KEY') and config.ADMIN_API_KEY:
            api_keys[config.ADMIN_API_KEY] = {
                'name': 'admin',
                'permissions': ['read', 'write', 'admin'],
                'rate_limit': 1000  # requests per hour
            }
        
        # Load additional keys from config
        if hasattr(config, 'API_KEYS') and config.API_KEYS:
            for key_data in config.API_KEYS:
                api_keys[key_data['key']] = {
                    'name': key_data.get('name', 'user'),
                    'permissions': key_data.get('permissions', ['read']),
                    'rate_limit': key_data.get('rate_limit', 100)
                }
        
        return api_keys
    
    def generate_api_key(self, name: str, permissions: list = None) -> str:
        """Generate a new API key"""
        if permissions is None:
            permissions = ['read']
        
        api_key = f"rag_{secrets.token_urlsafe(32)}"
        self.api_keys[api_key] = {
            'name': name,
            'permissions': permissions,
            'rate_limit': 100,
            'created_at': datetime.now().isoformat()
        }
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[Dict]:
        """Validate API key and return user info"""
        return self.api_keys.get(api_key)
    
    def check_rate_limit(self, api_key: str, endpoint: str) -> bool:
        """Check if request is within rate limits"""
        if not self.redis_client:
            return True  # Skip rate limiting if Redis not available
        
        user_info = self.validate_api_key(api_key)
        if not user_info:
            return False
        
        rate_limit = user_info.get('rate_limit', 100)
        current_hour = int(time.time() / 3600)
        
        key = f"rate_limit:{api_key}:{current_hour}"
        
        try:
            current_count = self.redis_client.get(key)
            if current_count is None:
                self.redis_client.setex(key, 3600, 1)
                return True
            
            current_count = int(current_count)
            if current_count >= rate_limit:
                return False
            
            self.redis_client.incr(key)
            return True
        except:
            return True  # Allow if Redis fails
    
    def generate_jwt_token(self, user_data: Dict) -> str:
        """Generate JWT token for session management"""
        payload = {
            'user': user_data,
            'exp': datetime.utcnow() + timedelta(hours=24),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_jwt_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token and return user data"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload.get('user')
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    def __init__(self, app, security_manager: SecurityManager):
        super().__init__(app)
        self.security_manager = security_manager
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks and docs
        if request.url.path in ['/health', '/docs', '/redoc', '/openapi.json']:
            return await call_next(request)
        
        # Extract API key from header
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            api_key = auth_header.split(' ')[1]
            
            if not self.security_manager.check_rate_limit(api_key, request.url.path):
                return JSONResponse(
                    status_code=429,
                    content={'error': 'Rate limit exceeded'}
                )
        
        return await call_next(request)

class SecurityMiddleware(BaseHTTPMiddleware):
    """Security headers and general security middleware"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        response.headers['Content-Security-Policy'] = "default-src 'self'"
        
        return response

# Authentication dependency functions
security_manager = SecurityManager()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> Dict:
    """Get current authenticated user"""
    token = credentials.credentials
    
    # Try JWT token first
    user_data = security_manager.verify_jwt_token(token)
    if user_data:
        return user_data
    
    # Try API key
    user_info = security_manager.validate_api_key(token)
    if user_info:
        return user_info
    
    raise HTTPException(status_code=401, detail="Invalid authentication credentials")

async def require_permission(permission: str):
    """Dependency to require specific permission"""
    def permission_dependency(user: Dict = Depends(get_current_user)):
        if permission not in user.get('permissions', []):
            raise HTTPException(status_code=403, detail=f"Permission '{permission}' required")
        return user
    return permission_dependency

async def get_admin_user(user: Dict = Depends(get_current_user)) -> Dict:
    """Require admin permissions"""
    if 'admin' not in user.get('permissions', []):
        raise HTTPException(status_code=403, detail="Admin permission required")
    return user

def api_key_required(func):
    """Decorator for API key requirement"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract credentials from kwargs or use dependency
        user = kwargs.get('user') or await get_current_user()
        return await func(*args, **kwargs)
    return wrapper
