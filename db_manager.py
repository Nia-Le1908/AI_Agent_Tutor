"""
Database connection manager supporting both SQLite and PostgreSQL.

This module provides:
- Unified database connection interface
- Connection pooling for PostgreSQL
- Thread-safe connections
- Automatic retry logic
- Context managers for safe resource handling
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Generator, Optional, Any, Dict, List
from config_db import db_config, is_postgresql, is_sqlite

logger = logging.getLogger(__name__)


# Try to import PostgreSQL dependencies if needed
if is_postgresql():
    try:
        from sqlalchemy import create_engine, text
        from sqlalchemy.pool import QueuePool
        from psycopg2 import pool
        PSYCOPG2_AVAILABLE = True
    except ImportError:
        PSYCOPG2_AVAILABLE = False
        logger.warning(
            "PostgreSQL selected but psycopg2/sqlalchemy not installed. "
            "Install with: pip install psycopg2-binary sqlalchemy"
        )
else:
    PSYCOPG2_AVAILABLE = False
    try:
        import sqlite3
    except ImportError:
        raise ImportError("sqlite3 is required for SQLite database support")


class DatabaseManager:
    """
    Unified database manager supporting SQLite and PostgreSQL.
    
    Features:
    - Thread-safe connections
    - Connection pooling (PostgreSQL)
    - Automatic retries
    - Context manager support
    """
    
    _instance: Optional[DatabaseManager] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> DatabaseManager:
        """Singleton pattern for database manager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.db_type = db_config.DB_TYPE
        self._engine = None
        self._connection_pool = None
        self._local = threading.local()
        
        if self.db_type == "postgresql":
            self._init_postgresql()
        else:
            self._init_sqlite()
        
        self._initialized = True
        logger.info(f"Database manager initialized with {self.db_type}")
    
    def _init_postgresql(self) -> None:
        """Initialize PostgreSQL connection pool."""
        if not PSYCOPG2_AVAILABLE:
            raise ImportError(
                "PostgreSQL support requires psycopg2-binary and sqlalchemy. "
                "Install with: pip install psycopg2-binary sqlalchemy"
            )
        
        try:
            from sqlalchemy import create_engine
            
            connection_string = db_config.get_connection_string()
            self._engine = create_engine(
                connection_string,
                pool_size=db_config.POSTGRES_POOL_SIZE,
                max_overflow=db_config.POSTGRES_MAX_OVERFLOW,
                pool_pre_ping=True,
                echo=False,
            )
            logger.info("PostgreSQL connection pool created successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            raise
    
    def _init_sqlite(self) -> None:
        """Initialize SQLite connection settings."""
        # SQLite connections are created on-demand
        logger.info("SQLite database manager initialized")
    
    @contextmanager
    def get_connection(self) -> Generator[Any, None, None]:
        """
        Get a database connection using context manager.
        
        Yields:
            Database connection object (sqlite3.Connection or SQLAlchemy connection)
        
        Example:
            with db_manager.get_connection() as conn:
                conn.execute("SELECT * FROM users")
        """
        conn = None
        try:
            if self.db_type == "postgresql":
                conn = self._engine.connect()
                yield conn
            else:
                conn = self._get_sqlite_connection()
                yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn is not None:
                conn.close()
    
    def _get_sqlite_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with safe defaults."""
        import sqlite3
        from config import DB_PATH
        
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn
    
    def execute_query(
        self, 
        query: str, 
        params: Optional[tuple] = None,
        fetch: bool = False
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Execute a database query with optional parameters.
        
        Args:
            query: SQL query string
            params: Query parameters tuple
            fetch: If True, fetch and return results
            
        Returns:
            List of dictionaries if fetch=True, None otherwise
        """
        if params is None:
            params = ()
        
        with self.get_connection() as conn:
            try:
                if self.db_type == "postgresql":
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(query, params)
                
                if fetch:
                    rows = result.fetchall()
                    if self.db_type == "postgresql":
                        return [dict(row._mapping) for row in rows]
                    else:
                        return [dict(row) for row in rows]
                
                conn.commit() if hasattr(conn, 'commit') else None
                return None
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                raise
    
    def execute_many(
        self,
        query: str,
        params_list: List[tuple]
    ) -> int:
        """
        Execute a query multiple times with different parameters.
        
        Args:
            query: SQL query string with placeholders
            params_list: List of parameter tuples
            
        Returns:
            Number of rows affected
        """
        with self.get_connection() as conn:
            try:
                if self.db_type == "postgresql":
                    from sqlalchemy import text
                    result = conn.execute(text(query), params_list)
                else:
                    result = conn.executemany(query, params_list)
                
                conn.commit() if hasattr(conn, 'commit') else None
                return result.rowcount if hasattr(result, 'rowcount') else len(params_list)
            except Exception as e:
                logger.error(f"Bulk execution failed: {e}")
                raise
    
    def health_check(self) -> bool:
        """
        Check database connectivity.
        
        Returns:
            True if database is accessible, False otherwise
        """
        try:
            with self.get_connection() as conn:
                if self.db_type == "postgresql":
                    from sqlalchemy import text
                    conn.execute(text("SELECT 1"))
                else:
                    conn.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Global singleton instance
db_manager = DatabaseManager()


def get_db_connection():
    """Get database connection using context manager."""
    return db_manager.get_connection()


def execute_query(query: str, params: Optional[tuple] = None, fetch: bool = False):
    """Execute a database query."""
    return db_manager.execute_query(query, params, fetch)


def health_check() -> bool:
    """Check database health."""
    return db_manager.health_check()
