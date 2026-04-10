import pandas as pd
import logging
from .db_utils import execute_query

logger = logging.getLogger(__name__)

def fetch_coldroom_data(days=7):
    """
    Fetches temperature and humidity for specific Coldroom asset IDs.
    Uses SQL-side JSON extraction for efficiency.
    """
    query = f"""
        SELECT 
            a.name AS coldroom_name,
            (d.value->>'Temperature')::float AS temperature,
            (d.value->>'Humidity')::float AS humidity,
            (d.value->>'Timestamp')::timestamp AS sensor_timestamp,
            d.timestamp AS db_timestamp
        FROM asset_datapoint d
        JOIN asset a ON d.entity_id = a.id
        WHERE d.entity_id IN (
            '1zT8DEEVLGvl8nUxaLL0W8',
            '5uhvZaKF94qAupKL1H8Vm1',
            '5rsuXBEAO4FRILtQKEPUNV',
            '4wpAc9iS6UsEiX6oJuvpDC',
            '4SA19NMdneRLOc0p9UUXfT',
            '5yeZ8X3xlpurMQ53WA9MOV',
            '7VQugAzZEx3Rdj1Dm07ofQ',
            '3IC18zO8okOcVDWcQ3UAjj',
            '2K03aJIUXF62nR5lZ73N2J',
            '4AGgJSPtDhEvAwZBFDO977',
            '4XqB7d4bMAmvXFhbj03UGd'
        )
        AND d.timestamp >= NOW() - INTERVAL '{days} days'
        AND d.attribute_name = 'data'
        ORDER BY a.name, d.timestamp ASC
    """
    
    try:
        results = execute_query(query)
        if not results:
            logger.warning(f"No Coldroom data found in the last {days} days for the specified IDs.")
            return None
        
        df = pd.DataFrame(results)
        df['sensor_timestamp'] = pd.to_datetime(df['sensor_timestamp'])
        return df
    except Exception as e:
        logger.error(f"Failed to fetch Coldroom data: {e}")
        return None
