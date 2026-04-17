import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
import pickle
import time
from tensorflow.keras.models import load_model

# Import local consolidated identification and fetchers from asset_mapping
from core import asset_mapping as am
from core.shared_utils import slugify, engineer_coldroom_features, engineer_tank_features

# Force CPU optimization
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

# Logger for industrial status
logger = logging.getLogger(__name__)

# Constants
TIME_STEPS = 30
MAX_FT_DEFAULT = 25.0
DATA_BASE_DIR = "data"
MODEL_BASE_DIR = "models"

TANK_MAX_LEVELS = {
    'tank7': 15.0,
    'tank11': 50.0,
    'tank12': 50.0,
    'tank13': 50.0
}

# Native imports from unified backend structure
try:
    from train.train_utils import load_artifact
    from train.coldroom.train_coldroom import train_coldroom
    from train.tanks_1_6.train_tanks_1_6 import train_tank_1_6
    from train.tanks_7_13.train_tanks_7_13 import train_tank_7_13
    logger.info("Industrial Data Hub: Logic components from backend successfully linked.")
except Exception as e:
    logger.error(f"Industrial Data Hub Error: Local logic unavailable! Reason: {e}")
    def _unavailable_stub(*args, **kwargs):
        raise RuntimeError("Backend logic components are missing from the current directory.")
    train_coldroom = train_tank_1_6 = train_tank_7_13 = _unavailable_stub
    load_artifact = _unavailable_stub


def engineer_tank_features(df, max_ft):
    """ Implements Tank feature engineering parity with CORRECTED training logic. """
    df = df.copy()
    df['sensor_timestamp'] = pd.to_datetime(df['sensor_timestamp'])
    df = df.sort_values('sensor_timestamp')
    df['actual_level'] = max_ft - df['level_feet'] 
    df['fill_pct'] = (df['actual_level'] / max_ft) * 100
    df['roc'] = df['actual_level'].diff().fillna(0)
    df['roc_abs'] = df['roc'].abs()
    df['accel'] = df['roc'].diff().fillna(0)
    df['roll_mean'] = df['actual_level'].rolling(60, min_periods=1).mean().fillna(df['actual_level'].mean())
    df['roll_std'] = df['actual_level'].rolling(60, min_periods=1).std().fillna(0)
    df['roll_range'] = df['actual_level'].rolling(60, min_periods=1).max() - df['actual_level'].rolling(60, min_periods=1).min()
    df['dev_from_mean'] = df['actual_level'] - df['roll_mean']
    mu = df['actual_level'].mean()
    sig = df['actual_level'].std() + 1e-9
    df['z_score'] = (df['actual_level'] - mu) / sig
    df['hour'] = df['sensor_timestamp'].dt.hour
    df['minute'] = df['sensor_timestamp'].dt.minute
    df['day_of_week'] = df['sensor_timestamp'].dt.dayofweek
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    return df

def prepare_data_for_training(df, asset_name, relative_path):
    """ Saves industrial data to a hierarchical root 'data' folder. """
    full_path = os.path.join(DATA_BASE_DIR, relative_path)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    data_file = os.path.join(full_path, "data.csv")
    df.to_csv(data_file, index=False)
    logger.info(f"Industrial Fetch: Data snapshot stored in {data_file}")
    return data_file

def analyze_coldroom(asset_name: str, force_retrain: bool = False, fetch_hours: int = None):
    """ Industrial pipeline for ColdRoom Temperature/Humidity anomaly detection. """
    start_time = time.time()
    asset_id = am.get_asset_id(asset_name)
    if not asset_id or asset_id not in am.COLDROOM_MAPPINGS:
        return f"Error: {asset_name} is not a recognized Coldroom asset."
    
    slug = slugify(asset_name)
    model_dir = os.path.join(MODEL_BASE_DIR, 'coldroom', slug)
    model_path = os.path.join(model_dir, 'model.h5')
    config_path = os.path.join(model_dir, 'config.pkl')
    
    if force_retrain and os.path.exists(model_dir):
        logger.info(f"Forcing retraining for {asset_name}. Deleting legacy artifacts...")
        import shutil
        shutil.rmtree(model_dir)

    is_cached = os.path.exists(model_path) and os.path.exists(config_path)
    if fetch_hours is None:
        fetch_hours = 24 if is_cached else 168 
    
    df = am.fetch_coldroom_data(asset_id, days=max(1, fetch_hours//24)) # fetch_coldroom_data uses days
    # If fetch_hours < 24 but we need bits of it, we still fetch 1 day but we'll slice it if needed.
    # Actually, am.fetch_coldroom_data might actually use days. 
    # Let's check am.fetch_coldroom_data signature.
    if df is None or len(df) < TIME_STEPS:
        count = len(df) if df is not None else 0
        return f"Error: Insufficient telemetry for {asset_name} (Need {TIME_STEPS} rows, got {count})."

    data_path = prepare_data_for_training(df, asset_name, os.path.join('coldroom', slug))
    
    if not is_cached:
        if not os.path.exists(model_dir): os.makedirs(model_dir)
        logger.info(f"Industrial Hub: Initial Training for {asset_name}...")
        train_coldroom(data_path, model_dir)

    try:
        model = load_model(model_path, compile=False)
        scalar = load_artifact(os.path.join(model_dir, 'scalar.pkl'))
        config = load_artifact(config_path)
        threshold = config['threshold']

        df = engineer_coldroom_features(df)
        df = df.dropna(subset=['temperature', 'humidity'])
        features = config.get('features', ["temperature", "humidity", "temp_diff", "rolling_mean", "rolling_std"])
        X_scaled = scalar.transform(df[features].values)
        X_room = np.array([X_scaled[-TIME_STEPS:]]) 
        
        X_pred = model.predict(X_room, verbose=0)
        mse = float(np.mean(np.power(X_room - X_pred, 2)))
        is_anomaly = mse > threshold
        latest_row = df.iloc[-1]
        canonical_name = am.get_asset_name(asset_id).capitalize()
        
        return {
            "name": canonical_name,
            "status": "Anomaly" if is_anomaly else "Normal",
            "latest_temp": float(latest_row["temperature"]),
            "humidity": float(latest_row["humidity"]),
            "anomaly": 1 if is_anomaly else 0,
            "threshold": float(threshold),
            "latest_mse": mse,
            "timestamp": str(latest_row["sensor_timestamp"])
        }
    except Exception as e:
        return f"Error during analysis of {asset_name}: {e}"

def analyze_tank(asset_name: str, force_retrain: bool = False, fetch_hours: int = None):
    """ Industrial pipeline for Refinery Tank oil level anomaly detection. """
    start_time = time.time()
    asset_id = am.get_asset_id(asset_name)
    if not asset_id or asset_id not in am.TANK_MAPPINGS:
        return f"Error: {asset_name} is not a recognized Refinery Tank."
    
    import re
    tank_num_match = re.search(r'\d+', asset_name)
    tank_num = int(tank_num_match.group()) if tank_num_match else 0
    group_name = "tanks_1_to_6" if 1 <= tank_num <= 6 else "tanks_7_to_13"
    
    model_dir = os.path.join(MODEL_BASE_DIR, group_name, f"tank{tank_num}")
    model_path = os.path.join(model_dir, 'model.h5')
    config_path = os.path.join(model_dir, 'config.pkl')
    
    if force_retrain and os.path.exists(model_dir):
        logger.info(f"Forcing retraining for {asset_name}. Deleting legacy artifacts...")
        import shutil
        shutil.rmtree(model_dir)

    is_cached = os.path.exists(model_path) and os.path.exists(config_path)
    if fetch_hours is None:
        fetch_hours = 24 if is_cached else 168 
    
    df = am.fetch_tank_data(asset_id, hours=fetch_hours)
    if df is None or len(df) < TIME_STEPS:
        count = len(df) if df is not None else 0
        return f"Error: Insufficient telemetry for {asset_name} (Got {count})."

    relative_path = os.path.join('tanks', group_name, f"tank{tank_num}")
    data_path = prepare_data_for_training(df, asset_name, relative_path)
    
    if not is_cached:
        if not os.path.exists(model_dir): os.makedirs(model_dir)
        logger.info(f"Industrial Hub: Retraining {asset_name} model...")
        if 1 <= tank_num <= 6: train_tank_1_6(data_path, model_dir)
        else: train_tank_7_13(data_path, model_dir)

    try:
        model = load_model(model_path, compile=False)
        scalar = load_artifact(os.path.join(model_dir, 'scalar.pkl'))
        config = load_artifact(config_path)
        threshold = config["threshold"]
        max_ft = TANK_MAX_LEVELS.get(asset_name.lower().replace(" ", ""), MAX_FT_DEFAULT)
        
        df = engineer_tank_features(df, max_ft)
        df = df.dropna(subset=['actual_level']).fillna(0)
        features = config["features"]
        X_scaled = scalar.transform(df[features].values)
        X_seq = np.array([X_scaled[-TIME_STEPS:]])
        X_pred = model.predict(X_seq, verbose=0)
        mse = float(np.mean(np.power(X_seq - X_pred, 2)))
        
        is_anomaly = mse > threshold
        latest_row = df.iloc[-1]
        canonical_name = am.get_asset_name(asset_id)
        
        return {
            "name": canonical_name,
            "status": "Anomaly" if is_anomaly else "Normal",
            "latest_level": float(latest_row["actual_level"]),
            "max_capacity": f"{max_ft} ft",
            "anomaly": 1 if is_anomaly else 0,
            "threshold": float(threshold),
            "latest_mse": mse,
            "timestamp": str(latest_row["sensor_timestamp"])
        }
    except Exception as e:
        return f"Error during Tank analysis: {e}"

def scan_all_coldrooms(fetch_hours: int = None):
    """Industrial Tool: Scan all discovered coldrooms for anomalies."""
    if not am.COLDROOM_MAPPINGS: am.load_dynamic_mappings()
    logger.info(f"Industrial Hub: Starting global scan for {len(am.COLDROOM_MAPPINGS)} coldrooms...")
    reports = []
    anomalies = []
    passed = 0
    failed = 0
    
    for asset_id, asset_name in am.COLDROOM_MAPPINGS.items():
        try:
            logger.debug(f"Scanning coldroom: {asset_name} ({asset_id})")
            result = analyze_coldroom(asset_name, fetch_hours=fetch_hours)
            if isinstance(result, dict):
                reports.append(result)
                passed += 1
                if result.get("anomaly"):
                    logger.info(f"Anomaly detected in {asset_name}!")
                    anomalies.append(result)
            else:
                logger.warning(f"Analysis for {asset_name} returned non-dict result: {result}")
                failed += 1
        except Exception as e:
            logger.error(f"Exception during analysis of {asset_name}: {e}")
            failed += 1
    
    logger.info(f"Global coldroom scan complete. Passed: {passed}, Failed: {failed}, Anomalies: {len(anomalies)}")
    return {
        "summary": f"{'🔴' if anomalies else '🟢'} Scanned {len(am.COLDROOM_MAPPINGS)} assets. Found {len(anomalies)} anomalies. {passed} checked successfuly.",
        "incidents": anomalies, 
        "all_reports": reports,
        "scanned_count": len(am.COLDROOM_MAPPINGS),
        "operational_count": passed - len(anomalies),
        "failed_count": failed
    }

def scan_all_tanks(fetch_hours: int = None):
    """Industrial Tool: Scan all discovered tanks for anomalies."""
    if not am.TANK_MAPPINGS: am.load_dynamic_mappings()
    logger.info(f"Industrial Hub: Starting global scan for {len(am.TANK_MAPPINGS)} refinery tanks...")
    reports = []
    anomalies = []
    passed = 0
    failed = 0

    for asset_id, asset_name in am.TANK_MAPPINGS.items():
        try:
            logger.debug(f"Scanning tank: {asset_name} ({asset_id})")
            result = analyze_tank(asset_name, fetch_hours=fetch_hours)
            if isinstance(result, dict):
                reports.append(result)
                passed += 1
                if result.get("anomaly"):
                    logger.info(f"Anomaly detected in {asset_name}!")
                    anomalies.append(result)
            else:
                logger.warning(f"Analysis for {asset_name} returned non-dict result: {result}")
                failed += 1
        except Exception as e:
            logger.error(f"Exception during analysis of {asset_name}: {e}")
            failed += 1
            
    logger.info(f"Global tank scan complete. Passed: {passed}, Failed: {failed}, Anomalies: {len(anomalies)}")
    return {
        "summary": f"{'🔴' if anomalies else '🟢'} Scanned {len(am.TANK_MAPPINGS)} refinery tanks. Found {len(anomalies)} anomalies. {passed} checked successfuly.",
        "incidents": anomalies, 
        "all_reports": reports,
        "scanned_count": len(am.TANK_MAPPINGS),
        "operational_count": passed - len(anomalies),
        "failed_count": failed
    }
