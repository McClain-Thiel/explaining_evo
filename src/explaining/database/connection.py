import os
from typing import Optional, Dict, Any, Generator
from sqlmodel import SQLModel, Session, create_engine
from sqlalchemy.engine import Engine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default database connection string for development
DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/explainable_evo"


def get_database_url() -> str:
    """
    Get database URL from environment variable or use default.
    
    Returns:
        Database connection URL string
    """
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def create_db_engine(database_url: Optional[str] = None, echo: bool = False) -> Engine:
    """
    Create SQLAlchemy engine for database connection.
    
    Args:
        database_url: Optional database URL (will use get_database_url() if not provided)
        echo: Whether to echo SQL statements
        
    Returns:
        SQLAlchemy engine instance
    """
    if database_url is None:
        database_url = get_database_url()
        
    logger.info(f"Creating database engine with URL: {database_url}")
    return create_engine(database_url, echo=echo)


def get_session(engine: Optional[Engine] = None) -> Generator[Session, None, None]:
    """
    Get a database session.
    
    Args:
        engine: Optional SQLAlchemy engine (will create one if not provided)
        
    Yields:
        SQLModel session
    """
    if engine is None:
        engine = create_db_engine()
        
    with Session(engine) as session:
        yield session


def init_db(engine: Optional[Engine] = None, drop_all: bool = False) -> None:
    """
    Initialize database schema.
    
    Args:
        engine: Optional SQLAlchemy engine (will create one if not provided)
        drop_all: Whether to drop all tables before creating them
    """
    from src.explaining.database.models import Sequence, FeatureSet, Feature, SequenceFeature
    
    if engine is None:
        engine = create_db_engine()
    
    logger.info("Initializing database schema...")
    
    if drop_all:
        logger.warning("Dropping all tables from database!")
        SQLModel.metadata.drop_all(engine)
    
    SQLModel.metadata.create_all(engine)
    logger.info("Database schema created successfully.") 