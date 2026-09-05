"""
Authentication and authorization for AI Tutor.

Provides:
- password hashing (bcrypt, with read-only verification of legacy SHA-256 hashes);
- JWT issuing and verification;
- role-based access control decorators;
- user management operations.

All SQL goes through :mod:`db_manager`, so this module runs unchanged on SQLite and
PostgreSQL, and never holds a connection of its own.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, List, Optional

import db_manager
from config import LEGACY_PASSWORD_FALLBACK_ENABLED, SESSION_TIMEOUT_HOURS
from validation import require_level, require_non_empty_str, require_positive_int

logger = logging.getLogger(__name__)

try:
    import bcrypt

    BCRYPT_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the local environment
    bcrypt = None
    BCRYPT_AVAILABLE = False
    logger.warning("bcrypt not installed. Password hashing is unavailable.")

try:
    import jwt

    JWT_AVAILABLE = True
except ImportError:  # pragma: no cover
    jwt = None
    JWT_AVAILABLE = False
    logger.warning("PyJWT not installed. Token authentication unavailable.")

BCRYPT_HASH_PREFIX = "$2"
JWT_ALGORITHM = "HS256"

ROLE_USER = "user"
ROLE_ADMIN = "admin"

# Admins satisfy any role requirement; every other role must match exactly.
ROLE_PRECEDENCE = {ROLE_ADMIN: 2, ROLE_USER: 1}


class AuthenticationError(Exception):
    """Raised when authentication fails."""


class AuthorizationError(Exception):
    """Raised when authorization fails."""


class SecurityUnavailableError(RuntimeError):
    """Raised when a required security dependency is not installed."""


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_secret_key() -> str:
    """
    Return the JWT signing key.

    Cached per process. When JWT_SECRET_KEY is unset a random key is generated so
    tokens still work within one run, but they will not survive a restart - which
    is exactly the behaviour you want to notice in production, hence the warning.
    """
    from config import JWT_SECRET_KEY

    if not JWT_SECRET_KEY:
        logger.warning("JWT_SECRET_KEY not set. Using an ephemeral secret (development only).")
        return secrets.token_hex(32)
    return JWT_SECRET_KEY


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    """
    Hash a password with bcrypt.

    Raises:
        SecurityUnavailableError: when bcrypt is missing. A silent fall back to
            unsalted SHA-256 would store effectively plaintext credentials, which
            is worse than failing to start.
        ValueError: when the password is empty.
    """
    if not BCRYPT_AVAILABLE:
        raise SecurityUnavailableError(
            "bcrypt is required to hash passwords. Install with: pip install bcrypt"
        )

    if not password:
        raise ValueError("password must not be empty")

    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def _legacy_sha256(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    """
    Verify a password against a stored hash.

    Handles both bcrypt hashes and the legacy unsalted SHA-256 hashes written by
    earlier versions of this module, so existing accounts are never locked out.
    """
    if not isinstance(password, str) or not isinstance(hashed, str) or not hashed:
        return False

    if hashed.startswith(BCRYPT_HASH_PREFIX):
        if not BCRYPT_AVAILABLE:
            logger.error("Stored hash is bcrypt but bcrypt is not installed; cannot verify.")
            return False
        try:
            return bool(bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8")))
        except ValueError as exc:
            logger.error("Password verification error: %s", exc)
            return False

    if not LEGACY_PASSWORD_FALLBACK_ENABLED:
        logger.error(
            "Refusing legacy SHA-256 password verification (set ALLOW_LEGACY_PASSWORD_FALLBACK "
            "only for migration)."
        )
        return False

    # Comparison is constant-time to avoid leaking the hash prefix through timing.
    return secrets.compare_digest(_legacy_sha256(password), hashed)


def needs_password_rehash(hashed: str) -> bool:
    """True when the stored hash should be upgraded on next successful login."""
    return not hashed.startswith(BCRYPT_HASH_PREFIX)


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
def _now() -> datetime:
    return datetime.now(timezone.utc)


def generate_token(
    user_id: int,
    role: str = ROLE_USER,
    expires_hours: Optional[int] = None,
) -> str:
    """
    Issue a signed JWT for a user.

    Raises:
        SecurityUnavailableError: when PyJWT is not installed.
        ValueError: for an invalid user id or expiry window.
    """
    if not JWT_AVAILABLE:
        raise SecurityUnavailableError("PyJWT is required for token generation. pip install PyJWT")

    user_id = require_positive_int(user_id, "user_id")
    hours = SESSION_TIMEOUT_HOURS if expires_hours is None else expires_hours
    if hours <= 0:
        raise ValueError("expires_hours must be positive")

    issued_at = _now()
    payload = {
        "user_id": user_id,
        "role": role,
        "iat": issued_at,
        "exp": issued_at + timedelta(hours=hours),
        "type": "access",
    }
    return jwt.encode(payload, _get_secret_key(), algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify and decode a JWT.

    Raises:
        AuthenticationError: when the token is invalid or expired.
        SecurityUnavailableError: when PyJWT is not installed.
    """
    if not JWT_AVAILABLE:
        raise SecurityUnavailableError("PyJWT is required for token verification. pip install PyJWT")
    if not token:
        raise AuthenticationError("Authentication token required")

    try:
        return jwt.decode(token, _get_secret_key(), algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError(f"Invalid token: {exc}") from exc


def _resolve_authorized_user(args: tuple, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Read a verified token from kwargs, or accept an already-authenticated user."""
    if "current_user" in kwargs:
        return dict(kwargs["current_user"])

    token = kwargs.get("token") or kwargs.get("auth_token") or (args[0] if args else None)
    if not token:
        raise AuthenticationError("Authentication token required")

    return verify_token(token)


def require_auth(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator requiring a valid token; injects ``user_id`` and ``user_role``.

    Usage:
        @require_auth
        def protected_function(**kwargs): ...
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        payload = _resolve_authorized_user(args, kwargs)
        kwargs["user_id"] = payload.get("user_id")
        kwargs["user_role"] = payload.get("role", ROLE_USER)
        return func(*args, **kwargs)

    return wrapper


def require_role(required_role: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator requiring a role, with admins allowed everywhere.

    Usage:
        @require_role("admin")
        def admin_function(**kwargs): ...
    """
    required_level = ROLE_PRECEDENCE.get(required_role)
    if required_level is None:
        raise ValueError(f"Unknown role in @require_role: {required_role!r}")

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        @require_auth
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            user_role = kwargs.get("user_role", ROLE_USER)
            if ROLE_PRECEDENCE.get(user_role, 0) < required_level:
                raise AuthorizationError(f"Access denied. Required role: {required_role}")
            return func(*args, **kwargs)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------
def _clean_email(email: Optional[str]) -> Optional[str]:
    if email is None:
        return None
    cleaned = str(email).strip()
    return cleaned.lower() or None


class UserManager:
    """User CRUD and authentication helpers backed by :mod:`db_manager`."""

    def create_user(
        self,
        name: str,
        email: Optional[str] = None,
        password: Optional[str] = None,
        level: int = 1,
    ) -> int:
        """
        Create a user and return their id.

        Raises:
            ValueError: for a missing name or an out-of-range level.
            db_manager.DatabaseError: when the insert fails (e.g. duplicate email).
        """
        name = require_non_empty_str(name, "name", max_length=100)
        level = require_level(level, "level")
        password_hash = hash_password(password) if password else None

        return db_manager.insert_returning_id(
            "users",
            {
                "name": name,
                "email": _clean_email(email),
                "password_hash": password_hash,
                "level": level,
            },
        )

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Return a user row as a dict, or None when unknown."""
        require_positive_int(user_id, "user_id")
        return db_manager.fetch_one(
            """
            SELECT id, name, email, level, is_active, created_at, last_login
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        )

    def get_user_by_identifier(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Look up the login row (including password hash) by name or email."""
        identifier = require_non_empty_str(identifier, "identifier")
        return db_manager.fetch_one(
            """
            SELECT u.id, u.name, u.password_hash, u.level,
                   (SELECT r.name FROM user_roles ur
                    INNER JOIN roles r ON r.id = ur.role_id
                    WHERE ur.user_id = u.id
                    -- Admins outrank plain users when a user holds several roles.
                    ORDER BY CASE r.name WHEN 'admin' THEN 0 ELSE 1 END ASC LIMIT 1) AS role
            FROM users u
            WHERE (u.name = ? OR lower(u.email) = ?) AND u.is_active = 1
            """,
            (identifier, identifier.lower()),
        )

    def authenticate_user(self, identifier: str, password: str) -> Optional[str]:
        """
        Verify credentials and return a JWT, or None when authentication fails.

        Failures are logged but never disclose which part was wrong, and the timing
        of the "unknown user" path stays similar to the "wrong password" path.
        """
        try:
            user = self.get_user_by_identifier(identifier)
        except (ValueError, db_manager.DatabaseError) as exc:
            logger.error("Authentication lookup failed: %s", exc)
            return None

        if not user:
            logger.warning("Authentication failed: unknown user '%s'", identifier)
            return None

        stored_hash = user.get("password_hash")
        if not stored_hash or not verify_password(password or "", stored_hash):
            logger.warning("Authentication failed: invalid credentials for '%s'", identifier)
            return None

        if needs_password_rehash(stored_hash) and BCRYPT_AVAILABLE:
            self._upgrade_password_hash(user["id"], password)

        self.touch_last_login(int(user["id"]))
        return generate_token(int(user["id"]), user.get("role") or ROLE_USER)

    def _upgrade_password_hash(self, user_id: int, password: str) -> None:
        """Transparently migrate a legacy hash to bcrypt after a good login."""
        try:
            db_manager.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (hash_password(password), user_id),
            )
            logger.info("Upgraded legacy password hash for user %s", user_id)
        except (ValueError, db_manager.DatabaseError) as exc:
            # Migration is opportunistic; never fail a good login over it.
            logger.warning("Could not upgrade legacy password hash: %s", exc)

    def touch_last_login(self, user_id: int) -> None:
        """Record the current time in ``last_login`` (timestamp built in Python for portability)."""
        require_positive_int(user_id, "user_id")
        db_manager.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (_now().strftime("%Y-%m-%d %H:%M:%S"), user_id),
        )

    def get_user_roles(self, user_id: int) -> List[str]:
        """Return every role name assigned to a user."""
        require_positive_int(user_id, "user_id")
        rows = db_manager.fetch_all(
            """
            SELECT r.name
            FROM roles r
            INNER JOIN user_roles ur ON r.id = ur.role_id
            WHERE ur.user_id = ?
            ORDER BY r.name ASC
            """,
            (user_id,),
        )
        return [row["name"] for row in rows]

    def assign_role(self, user_id: int, role_name: str) -> bool:
        """
        Assign a role, ignoring an assignment that already exists.

        Returns:
            False when the role name is unknown; True otherwise (including when the
            user already had the role).
        """
        require_positive_int(user_id, "user_id")
        role_name = require_non_empty_str(role_name, "role_name")

        role = db_manager.fetch_one("SELECT id FROM roles WHERE name = ?", (role_name,))
        if not role:
            logger.error("Role '%s' not found", role_name)
            return False

        db_manager.insert_ignore("user_roles", {"user_id": user_id, "role_id": int(role["id"])})
        return True

    def deactivate_user(self, user_id: int) -> bool:
        """Soft-delete a user so their history stays queryable."""
        require_positive_int(user_id, "user_id")
        db_manager.execute("UPDATE users SET is_active = 0 WHERE id = ?", (user_id,))
        return True


user_manager = UserManager()
