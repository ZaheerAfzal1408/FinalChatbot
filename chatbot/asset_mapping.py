import os
import sys
import pandas as pd
import logging

# Mappings of Hash IDs to Asset Names
COLDROOM_MAPPINGS = {
    "4SA19NMdneRLOc0p9UUXfT": "coldroom1",
    "5rsuXBEAO4FRILtQKEPUNV": "coldroom2",
    "2K03aJIUXF62nR5lZ73N2J": "coldroom3",
    "3IC18zO8okOcVDWcQ3UAjj": "coldroom4",
    "4AGgJSPtDhEvAwZBFDO977": "coldroom5",
    "5uhvZaKF94qAupKL1H8Vm1": "coldroom6",
    "4wpAc9iS6UsEiX6oJuvpDC": "coldroom7",
    "1zT8DEEVLGvl8nUxaLL0W8": "coldroom8",
    "7VQugAzZEx3Rdj1Dm07ofQ": "coldroom9",
    "4XqB7d4bMAmvXFhbj03UGd": "coldroom10"
}

TANK_MAPPINGS = {
    "2TV1OLqZSv5JwIEHomeRR1": "tank1",
    "6UXHYoFE9as4nc25HA72AN": "tank2",
    "5UwgGOBQ8VqdVZO37E1ci8": "tank3",
    "2H7Gj65gRcLS1V6OTq72Hk": "tank4",
    "7BKD81c3z8gKpGZt5bIc87": "tank5",
    "2pd9BV69qYHsnoyiAAqJmX": "tank6",
    "50Y5udYfIWJWLEbduASYwT": "tank7",
    "6tUKaWyipyL8MfjaTntrSH": "tank8",
    "6UrAIw3WfeBmCn7yYgHGKR": "tank9",
    "64G6XlehtAyuhFP6U5V9iB": "tank11",
    "7gmxORynTjM8Lmp4DDtZSP": "tank12",
    "4la3XSG9sUisBdczxxjYLv": "tank13"
}

def get_asset_name(asset_id):
    """Returns the user-friendly name for a given hash ID."""
    name = COLDROOM_MAPPINGS.get(asset_id)
    if name: return name
    return TANK_MAPPINGS.get(asset_id, asset_id)

def get_asset_id(name):
    """Returns the hash ID for a given name (e.g. 'coldroom1' or 'tank1')."""
    search_name = name.lower().replace(" ", "")
    # Check ColdRooms
    for aid, n in COLDROOM_MAPPINGS.items():
        if n.lower() == search_name:
            return aid
    # Check Tanks
    for aid, n in TANK_MAPPINGS.items():
        if n.lower() == search_name:
            return aid
    return None

# --- Industrial Data Fetching Layer ---

# Robust path resolution for backend components
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
backend_path = os.path.join(project_root, 'backend')
if backend_path not in sys.path:
    sys.path.append(backend_path)
    sys.path.append(os.path.join(backend_path, 'database'))

try:
    import db_utils 
    execute_query = db_utils.execute_query
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to dynamically import legacy DB utils: {e}")
    def execute_query(*args, **kwargs): return []

logger = logging.getLogger(__name__)

def fetch_coldroom_data(asset_id: str, days: int = 7):
    """
    Dedicated Coldroom Fetcher: Retrieves temperature/humidity metrics.
    """
    if asset_id not in COLDROOM_MAPPINGS:
        logger.warning(f"Industrial Mapping: ID {asset_id} not found in COLDROOM_MAPPINGS.")
    
    query = f"""
        SELECT 
            (d.value->>'Temperature')::float AS temperature, 
            (d.value->>'Humidity')::float AS humidity,
            (d.value->>'Timestamp')::timestamp AS sensor_timestamp
        FROM asset_datapoint d
        WHERE d.entity_id = %s
        AND d.timestamp >= NOW() - INTERVAL '{days} days'
        AND d.attribute_name = 'data'
        ORDER BY d.timestamp ASC
    """
    try:
        results = execute_query(query, (asset_id,))
        if not results: return None
        
        df = pd.DataFrame(results)
        df['sensor_timestamp'] = pd.to_datetime(df['sensor_timestamp'])
        logger.info(f"Industrial Fetch: Successfully retrieved {len(df)} coldroom data rows for {asset_id}.")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch coldroom data for {asset_id}: {e}")
        return None

def fetch_tank_data(asset_id: str, hours: int = 3):
    """
    Dedicated Tank Fetcher: Retrieves level metrics for refinery assets.
    """
    if asset_id not in TANK_MAPPINGS:
        logger.warning(f"Industrial Mapping: ID {asset_id} not found in TANK_MAPPINGS.")

    query = f"""
        SELECT 
            (d.value->>'TankOilLevelInFeet001')::float AS level_feet,
            (d.value->>'Timestamp')::timestamp AS sensor_timestamp
        FROM asset_datapoint d
        WHERE d.entity_id = %s
        AND d.timestamp >= NOW() - INTERVAL '{hours} hours'
        AND d.attribute_name = 'data'
        ORDER BY d.timestamp ASC
    """
    try:
        results = execute_query(query, (asset_id,))
        if not results: return None
        
        df = pd.DataFrame(results)
        df['sensor_timestamp'] = pd.to_datetime(df['sensor_timestamp'])
        logger.info(f"Industrial Fetch: Successfully retrieved {len(df)} tank data rows for {asset_id}.")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch tank data for {asset_id}: {e}")
        return None
