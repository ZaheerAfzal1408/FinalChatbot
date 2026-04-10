import pandas as pd
import logging
import re
from .db_utils import execute_query

logger = logging.getLogger(__name__)

def fetch_tank_data(days=7):
    """
    Fetches oil level data for specific Tank asset IDs.
    Uses SQL-side JSON extraction as requested.
    """
    query = f"""
        SELECT 
            a.name AS tank_id,
            (d.value->>'TankOilLevelInFeet001')::float AS level_feet,
            (d.value->>'Timestamp')::timestamp AS sensor_timestamp,
            d.timestamp AS db_timestamp
        FROM asset_datapoint d
        JOIN asset a ON d.entity_id = a.id
        WHERE d.entity_id IN (
            '2TV1OLqZSv5JwIEHomeRR1',
            '6UXHYoFE9as4nc25HA72AN',
            '5UwgGOBQ8VqdVZO37E1ci8',
            '2H7Gj65gRcLS1V6OTq72Hk',
            '7BKD81c3z8gKpGZt5bIc87',
            '2pd9BV69qYHsnoyiAAqJmX',
            '50Y5udYfIWJWLEbduASYwT',
            '6tUKaWyipyL8MfjaTntrSH',
            '6UrAIw3WfeBmCn7yYgHGKR',
            '64G6XlehtAyuhFP6U5V9iB',
            '7gmxORynTjM8Lmp4DDtZSP',
            '4la3XSG9sUisBdczxxjYLv'
        )
        AND d.timestamp >= NOW() - INTERVAL '{days} days'
        AND d.attribute_name = 'data'
        ORDER BY a.name, d.timestamp ASC
    """
    
    try:
        results = execute_query(query)
        if not results:
            logger.warning(f"No Tank data found in the last {days} days for the specified IDs.")
            return None
        
        df = pd.DataFrame(results)
        
        # Consistent normalization: Physical07 instead of Physical7
        def normalize_tank_name(name):
            if not name: return name
            return re.sub(r'Physical(\d)$', r'Physical0\1', name)
            
        df['tank_id'] = df['tank_id'].apply(normalize_tank_name)
        df['sensor_timestamp'] = pd.to_datetime(df['sensor_timestamp'])
        
        return df
    except Exception as e:
        logger.error(f"Failed to fetch Tank data: {e}")
        return None
