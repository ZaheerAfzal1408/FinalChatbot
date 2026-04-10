import os
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

POSTGRES_URI = os.getenv("POSTGRES_URI", "")

def get_db_connection():
    """
    Establishes a connection to the PostgreSQL database using the URI from environment variables.
    Returns:
        psycopg2.connection or None: Reusable DB connection object.
    """
    if not POSTGRES_URI:
        logger.error("POSTGRES_URI is not set in the environment variables.")
        return None
    try:
        conn = psycopg2.connect(POSTGRES_URI, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to DB: {e}")
        return None

def execute_query(query, params=None):
    """
    Helper to execute a query and return results.
    """
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            if cur.description: # If it's a SELECT query
                return cur.fetchall()
            conn.commit()
            return []
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        return []
    finally:
        conn.close()
