"""
Database configuration and connection management for the Nurse Conversation Processing API.
Handles async SQLAlchemy setup, sessions, and database initialization.
"""
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.pool import QueuePool
from loguru import logger

from config import get_settings
from database_models import Base


class DatabaseManager:
    """Database connection and session manager."""
    
    def __init__(self):
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self._settings = get_settings()
    
    def create_engine(self) -> AsyncEngine:
        """Create and configure the async database engine."""
        if self._engine is not None:
            return self._engine
        
        db_settings = self._settings.database_settings
        
        # Engine configuration
        engine_kwargs = {
            "echo": db_settings.echo,
            "future": True,
            "poolclass": QueuePool,
            "pool_size": db_settings.pool_size,
            "max_overflow": db_settings.max_overflow,
            "pool_pre_ping": True,  # Validate connections
            "pool_recycle": 3600,   # Recycle connections every hour
        }
        
        # Handle SQLite vs PostgreSQL
        if db_settings.url.startswith("sqlite"):
            # SQLite-specific configuration
            engine_kwargs["connect_args"] = {
                "check_same_thread": False,
                "timeout": 30
            }
            # Remove PostgreSQL-specific options for SQLite
            engine_kwargs.pop("poolclass", None)
            engine_kwargs.pop("pool_size", None)
            engine_kwargs.pop("max_overflow", None)
        
        self._engine = create_async_engine(
            db_settings.url,
            **engine_kwargs
        )
        
        logger.info(f"Database engine created for: {db_settings.url.split('@')[-1] if '@' in db_settings.url else 'SQLite'}")
        return self._engine
    
    def create_session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Create the async session factory."""
        if self._session_factory is not None:
            return self._session_factory
        
        engine = self.create_engine()
        self._session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False
        )
        
        logger.info("Database session factory created")
        return self._session_factory
    
    async def create_tables(self):
        """Create all database tables."""
        engine = self.create_engine()
        async with engine.begin() as conn:
            logger.info("Creating database tables...")
            # Serialize concurrent creates across workers using advisory lock
            try:
                await conn.exec_driver_sql("SELECT pg_advisory_lock(1183372841)")
                await conn.run_sync(Base.metadata.create_all)
                logger.info("Database tables created successfully")
            finally:
                try:
                    await conn.exec_driver_sql("SELECT pg_advisory_unlock(1183372841)")
                except Exception:
                    pass
    
    async def drop_tables(self):
        """Drop all database tables (use with caution!)."""
        engine = self.create_engine()
        async with engine.begin() as conn:
            logger.warning("Dropping all database tables...")
            await conn.run_sync(Base.metadata.drop_all)
            logger.warning("All database tables dropped")
    
    async def check_connection(self) -> bool:
        """Check if database connection is working."""
        try:
            engine = self.create_engine()
            async with engine.begin() as conn:
                await conn.run_sync(lambda sync_conn: sync_conn.exec_driver_sql("SELECT 1"))
            logger.info("Database connection check successful")
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an async database session."""
        session_factory = self.create_session_factory()
        session = session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()
    
    async def close(self):
        """Close the database engine and all connections."""
        if self._engine:
            logger.info("Closing database connections...")
            await self._engine.dispose()
            logger.info("Database connections closed")
    
    @property
    def engine(self) -> AsyncEngine:
        """Get the database engine."""
        return self.create_engine()
    
    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get the session factory."""
        return self.create_session_factory()


# Global database manager instance
db_manager = DatabaseManager()


async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency function for FastAPI to get database sessions.
    Use this in your route dependencies.
    """
    async with db_manager.get_session() as session:
        yield session


async def init_database():
    """Initialize the database (create tables, etc.)."""
    logger.info("Initializing database...")
    await db_manager.create_tables()
    
    # Verify connection
    if await db_manager.check_connection():
        logger.info("Database initialization completed successfully")
    else:
        logger.error("Database initialization failed - connection check failed")
        raise Exception("Database connection failed")


async def cleanup_database():
    """Cleanup database connections."""
    logger.info("Cleaning up database connections...")
    await db_manager.close()


# Health check function
async def check_database_health() -> dict:
    """Check database health status."""
    try:
        is_connected = await db_manager.check_connection()
        return {
            "status": "healthy" if is_connected else "unhealthy",
            "connected": is_connected,
            "engine": str(db_manager._engine) if db_manager._engine else None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "connected": False,
            "error": str(e)
        }


# Transaction utilities
@asynccontextmanager
async def database_transaction() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database transactions.
    Automatically handles commit/rollback.
    """
    async with db_manager.get_session() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def execute_in_transaction(func, *args, **kwargs):
    """Execute a function within a database transaction."""
    async with database_transaction() as session:
        return await func(session, *args, **kwargs)


# Repository base class
class BaseRepository:
    """Base repository class with common database operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def add(self, obj):
        """Add an object to the session."""
        self.session.add(obj)
        await self.session.flush()
        return obj
    
    async def delete(self, obj):
        """Delete an object from the session."""
        await self.session.delete(obj)
        await self.session.flush()
    
    async def commit(self):
        """Commit the current transaction."""
        await self.session.commit()
    
    async def rollback(self):
        """Rollback the current transaction."""
        await self.session.rollback()
    
    async def refresh(self, obj):
        """Refresh an object from the database."""
        await self.session.refresh(obj)
        return obj


# Database utilities
async def run_database_migrations():
    """Run database migrations (placeholder for Alembic integration)."""
    logger.info("Database migrations would run here (implement with Alembic)")
    pass


async def create_test_data():
    """Create test data for development/testing."""
    logger.info("Creating test data...")
    
    # This would create sample data for testing
    # Implementation depends on your specific needs
    pass


def get_database_url() -> str:
    """Get the configured database URL."""
    return get_settings().database_url


# Connection pool monitoring
async def get_pool_status() -> dict:
    """Get connection pool status information."""
    engine = db_manager.engine
    pool = engine.pool
    
    return {
        "size": pool.size() if hasattr(pool, 'size') else None,
        "checked_in": pool.checkedin() if hasattr(pool, 'checkedin') else None,
        "checked_out": pool.checkedout() if hasattr(pool, 'checkedout') else None,
        "overflow": pool.overflow() if hasattr(pool, 'overflow') else None,
        "invalid": pool.invalid() if hasattr(pool, 'invalid') else None,
    }