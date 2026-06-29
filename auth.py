"""
Authentication and authorization module for AI Tutor.

This module provides:
- Password hashing using bcrypt
- JWT token generation and validation
- Role-based access control (RBAC)
- User session management
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from functools import wraps

from config_db import db_config

logger = logging.getLogger(__name__)

# Try to import security dependencies
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    logger.warning("bcrypt not installed. Password hashing will use fallback.")

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logger.warning("PyJWT not installed. Token authentication unavailable.")


class AuthenticationError(Exception):
    """Raised when authentication fails."""
    pass


class AuthorizationError(Exception):
    """Raised when authorization fails."""
    pass


def _get_secret_key() -> str:
    """Get or generate a secret key for JWT signing."""
    # Use module-level cache to ensure same key within process lifetime
    if not hasattr(_get_secret_key, "_cached_secret"):
        secret = os.getenv("JWT_SECRET_KEY", "")
        if not secret:
            # Generate a default secret for development (not for production!)
            logger.warning(
                "JWT_SECRET_KEY not set. Using generated secret (development only)."
            )
            secret = secrets.token_hex(32)
        _get_secret_key._cached_secret = secret
    return _get_secret_key._cached_secret


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password string
    """
    if not BCRYPT_AVAILABLE:
        # Fallback to SHA-256 (not recommended for production)
        logger.warning("Using fallback password hashing (SHA-256)")
        return hashlib.sha256(password.encode()).hexdigest()
    
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        password: Plain text password to verify
        hashed: Stored password hash
        
    Returns:
        True if password matches, False otherwise
    """
    if not BCRYPT_AVAILABLE:
        # Fallback verification
        return hashlib.sha256(password.encode()).hexdigest() == hashed
    
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False


def generate_token(user_id: int, role: str = "user", expires_hours: int = 24) -> str:
    """
    Generate a JWT token for a user.
    
    Args:
        user_id: User's database ID
        role: User's role (default: "user")
        expires_hours: Token validity in hours
        
    Returns:
        JWT token string
        
    Raises:
        ImportError: If PyJWT is not installed
    """
    if not JWT_AVAILABLE:
        raise ImportError(
            "PyJWT is required for token generation. "
            "Install with: pip install PyJWT"
        )
    
    payload = {
        "user_id": user_id,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=expires_hours),
        "iat": datetime.utcnow(),
        "type": "access"
    }
    
    token = jwt.encode(payload, _get_secret_key(), algorithm="HS256")
    return token


def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token payload as dictionary
        
    Raises:
        AuthenticationError: If token is invalid or expired
        ImportError: If PyJWT is not installed
    """
    if not JWT_AVAILABLE:
        raise ImportError(
            "PyJWT is required for token verification. "
            "Install with: pip install PyJWT"
        )
    
    try:
        payload = jwt.decode(
            token, 
            _get_secret_key(), 
            algorithms=["HS256"]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.InvalidTokenError as e:
        raise AuthenticationError(f"Invalid token: {e}")


def require_auth(func):
    """
    Decorator to require authentication for a function.
    
    Usage:
        @require_auth
        def protected_function(user_id, **kwargs):
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract token from kwargs or args
        token = kwargs.get("token") or kwargs.get("auth_token")
        
        if not token:
            raise AuthenticationError("Authentication token required")
        
        payload = verify_token(token)
        kwargs["user_id"] = payload["user_id"]
        kwargs["user_role"] = payload.get("role", "user")
        
        return func(*args, **kwargs)
    
    return wrapper


def require_role(required_role: str):
    """
    Decorator to require a specific role for a function.
    
    Usage:
        @require_role("admin")
        def admin_function(user_id, **kwargs):
            ...
    """
    def decorator(func):
        @wraps(func)
        @require_auth
        def wrapper(*args, **kwargs):
            user_role = kwargs.get("user_role", "")
            
            if user_role != required_role and required_role != "user":
                # Check if user has admin role (admins can access everything)
                if user_role != "admin":
                    raise AuthorizationError(
                        f"Access denied. Required role: {required_role}"
                    )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


class UserManager:
    """
    User management operations.
    
    This class handles user CRUD operations and authentication.
    """
    
    def __init__(self):
        self.db_type = db_config.DB_TYPE
    
    def create_user(
        self,
        name: str,
        email: Optional[str] = None,
        password: Optional[str] = None,
        level: int = 1
    ) -> int:
        """
        Create a new user.
        
        Args:
            name: User's display name
            email: User's email (optional but recommended)
            password: User's password (will be hashed)
            level: Initial difficulty level (1-5)
            
        Returns:
            New user's ID
            
        Raises:
            ValueError: If required fields are missing
        """
        if not name or not name.strip():
            raise ValueError("User name is required")
        
        from db_manager import get_db_connection
        
        password_hash = hash_password(password) if password else None
        
        query = """
            INSERT INTO users (name, email, password_hash, level)
            VALUES (?, ?, ?, ?)
        """
        
        try:
            with get_db_connection() as conn:
                cursor = conn.execute(query, (name.strip(), email, password_hash, level))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            raise
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user information by ID."""
        from db_manager import execute_query
        
        query = """
            SELECT id, name, email, level, is_active, created_at, last_login
            FROM users
            WHERE id = ?
        """
        
        try:
            results = execute_query(query, (user_id,), fetch=True)
            if results:
                return dict(results[0])
            return None
        except Exception as e:
            logger.error(f"Failed to get user: {e}")
            return None
    
    def authenticate_user(self, identifier: str, password: str) -> Optional[str]:
        """
        Authenticate a user and return a JWT token.
        
        Args:
            identifier: Username or email
            password: User's password
            
        Returns:
            JWT token if authentication successful, None otherwise
        """
        from db_manager import execute_query
        
        query = """
            SELECT u.id, u.password_hash, r.name as role
            FROM users u
            LEFT JOIN user_roles ur ON u.id = ur.user_id
            LEFT JOIN roles r ON ur.role_id = r.id
            WHERE (u.name = ? OR u.email = ?) AND u.is_active = 1
        """
        
        try:
            results = execute_query(query, (identifier, identifier), fetch=True)
            
            if not results:
                logger.warning(f"Authentication failed: user not found '{identifier}'")
                return None
            
            user_data = results[0]
            
            if not user_data.get("password_hash"):
                logger.warning(f"User '{identifier}' has no password set")
                return None
            
            if verify_password(password, user_data["password_hash"]):
                # Update last login
                update_query = "UPDATE users SET last_login = datetime('now') WHERE id = ?"
                execute_query(update_query, (user_data["id"],))
                
                # Generate token
                role = user_data.get("role", "user")
                return generate_token(user_data["id"], role)
            
            logger.warning(f"Authentication failed: invalid password for '{identifier}'")
            return None
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
    
    def get_user_roles(self, user_id: int) -> List[str]:
        """Get all roles assigned to a user."""
        from db_manager import execute_query
        
        query = """
            SELECT r.name
            FROM roles r
            INNER JOIN user_roles ur ON r.id = ur.role_id
            WHERE ur.user_id = ?
        """
        
        try:
            results = execute_query(query, (user_id,), fetch=True)
            return [row["name"] for row in results] if results else []
        except Exception as e:
            logger.error(f"Failed to get user roles: {e}")
            return []
    
    def assign_role(self, user_id: int, role_name: str) -> bool:
        """Assign a role to a user."""
        from db_manager import execute_query
        
        # First get role ID
        role_query = "SELECT id FROM roles WHERE name = ?"
        role_results = execute_query(role_query, (role_name,), fetch=True)
        
        if not role_results:
            logger.error(f"Role '{role_name}' not found")
            return False
        
        role_id = role_results[0]["id"]
        
        insert_query = """
            INSERT OR IGNORE INTO user_roles (user_id, role_id)
            VALUES (?, ?)
        """
        
        try:
            execute_query(insert_query, (user_id, role_id))
            return True
        except Exception as e:
            logger.error(f"Failed to assign role: {e}")
            return False
    
    def deactivate_user(self, user_id: int) -> bool:
        """Deactivate a user account."""
        from db_manager import execute_query
        
        query = "UPDATE users SET is_active = 0 WHERE id = ?"
        
        try:
            execute_query(query, (user_id,))
            return True
        except Exception as e:
            logger.error(f"Failed to deactivate user: {e}")
            return False


# Global user manager instance
user_manager = UserManager()
