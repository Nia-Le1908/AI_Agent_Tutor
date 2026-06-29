"""
Database configuration supporting both SQLite (development) and PostgreSQL (production).

This module provides database connection management with support for:
- SQLite for local development and testing
- PostgreSQL for production deployments with multiple users
- Connection pooling for better performance
- Automatic migration support
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Literal

from dotenv import load_dotenv

load_dotenv()


class DatabaseConfig:
    """Centralized database configuration."""
    
    def __init__(self):
        # Database type selector: 'sqlite' or 'postgresql'
        self.DB_TYPE: str = os.getenv("DB_TYPE", "sqlite").strip().lower()
        
        # SQLite settings
        self.SQLITE_DB_PATH: str = os.getenv(
            "SQLITE_DB_PATH", 
            "data/ai_tutor_v5.db"
        )
        
        # PostgreSQL settings
        self.POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
        self.POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
        self.POSTGRES_DB: str = os.getenv("POSTGRES_DB", "ai_tutor")
        self.POSTGRES_USER: str = os.getenv("POSTGRES_USER", "ai_tutor_user")
        self.POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "")
        self.POSTGRES_POOL_SIZE: int = int(os.getenv("POSTGRES_POOL_SIZE", "10"))
        self.POSTGRES_MAX_OVERFLOW: int = int(os.getenv("POSTGRES_MAX_OVERFLOW", "20"))
        
        # Validate configuration based on DB_TYPE
        if self.DB_TYPE == "postgresql":
            self._validate_postgres_config()
    
    def _validate_postgres_config(self) -> None:
        """Validate PostgreSQL configuration when selected."""
        if not self.POSTGRES_PASSWORD:
            raise ValueError(
                "POSTGRES_PASSWORD is required when DB_TYPE=postgresql. "
                "Set it in .env file."
            )
    
    def get_connection_string(self) -> str:
        """Get database connection string for SQLAlchemy or direct connections."""
        if self.DB_TYPE == "postgresql":
            return (
                f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
                f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
            )
        else:
            sqlite_path = Path(self.SQLITE_DB_PATH).resolve()
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{sqlite_path}"
    
    def get_driver_name(self) -> Literal["sqlite", "postgresql"]:
        """Get the database driver name."""
        return self.DB_TYPE if self.DB_TYPE in ("sqlite", "postgresql") else "sqlite"


# Global configuration instance
db_config = DatabaseConfig()

# Backward compatibility constants for existing code
DB_TYPE = db_config.DB_TYPE
DB_PATH = db_config.SQLITE_DB_PATH if db_config.DB_TYPE == "sqlite" else ""
CONNECTION_STRING = db_config.get_connection_string()


def get_db_type() -> str:
    """Get current database type."""
    return db_config.DB_TYPE


def is_postgresql() -> bool:
    """Check if PostgreSQL is configured."""
    return db_config.DB_TYPE == "postgresql"


def is_sqlite() -> bool:
    """Check if SQLite is configured."""
    return db_config.DB_TYPE == "sqlite"
