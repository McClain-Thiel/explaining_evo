#!/usr/bin/env python3
"""
Script to initialize the database schema for Evo2 feature analysis.
"""

import argparse
import os
import sys
import logging

from src.explaining.database.connection import get_database_url, create_db_engine, init_db


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Initialize database schema for Evo2 feature analysis"
    )
    
    parser.add_argument(
        "--database_url", 
        type=str, 
        default=None,
        help="Database connection URL (overrides env variable)"
    )
    parser.add_argument(
        "--drop_tables", 
        action="store_true",
        help="Drop existing tables before creating new ones"
    )
    parser.add_argument(
        "--echo", 
        action="store_true",
        help="Echo SQL statements"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main function to initialize the database."""
    args = parse_args()
    
    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url
    
    database_url = get_database_url()
    logger.info(f"Connecting to database: {database_url}")
    
    engine = create_db_engine(database_url, echo=args.echo)
    
    if args.drop_tables:
        logger.warning("Dropping all existing tables!")
        confirm = input("Are you sure you want to drop all tables? This cannot be undone! (y/n): ")
        if confirm.lower() != "y":
            logger.info("Operation cancelled.")
            return
    
    logger.info("Initializing database schema...")
    init_db(engine, drop_all=args.drop_tables)
    logger.info("Database schema initialized successfully.")


if __name__ == "__main__":
    main() 