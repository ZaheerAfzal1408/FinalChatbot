import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
import pickle
import time
from tensorflow.keras.models import load_model

# Import consolidated identification and fetchers from asset_mapping
from asset_mapping import get_asset_id, TANK_MAPPINGS, fetch_coldroom_data, fetch_tank_data

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

# Global variable to store import error if any
_backend_import_error = None

# Add backend components for logic reuse
try:
    # Robust path resolution for backend components
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    backend_path = os.path.join(project_root, 'backend')
    
    # Ensure backend path is at the start of sys.path to avoid conflicts
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    
    # Verify the path exists
    if not os.path.exists(backend_path):
        raise ImportError(f"Backend directory not found at {backend_path}")

    # Now attempt imports
    from app import slugify
    from train.train_utils import load_artifact
    from train.coldroom.train_coldroom import train_coldroom
    from train.tanks_1_6.train_tanks_1_6 import train_tank_1_6
    from train.tanks_7_13.train_tanks_7_13 import train_tank_7_13
    
    logger.info("Industrial Data Hub: Logic components from backend successfully linked.")

except Exception as e:
    _backend_import_error = str(e)
    logger.error(f"Industrial Data Hub Error: Backend logic unavailable! Reason: {e}")
    
    # Define descriptive failure functions instead of None stubs
    def _unavailable_stub(*args, **kwargs):
        raise RuntimeError(
            f"Backend logic is unavailable. This is likely due to a failed import or missing dependency. "
            f"Original Error: {_backend_import_error}"
        )
    
    # Replace None with functions that give clear errors
    train_coldroom = train_tank_1_6 = train_tank_7_13 = _unavailable_stub
    load_artifact = _unavailable_stub
    
    # Stub slugify as a simple regex if app.py import failed
    def slugify(text):
        import re
        return re.sub(r'[^a-zA-Z0-9]', '', str(text)).lower()

def engineer_coldroom_features(df):
    """
    Implements Coldroom feature engineering parity with training logic.
    """
    df = df.copy()
    df['sensor_timestamp'] = pd.to_datetime(df['sensor_timestamp'])
    df = df.sort_values('sensor_timestamp')
    df['temp_diff'] = df['temperature'].diff().fillna(0)
    df['rolling_mean'] = df['temperature'].rolling(window=12).mean().fillna(df['temperature'].mean())
    df['rolling_std'] = df['temperature'].rolling(window=12).std().fillna(0)
    return df

def engineer_tank_features(df, max_ft):
    """
    Implements Tank feature engineering parity with CORRECTED training logic.
    Logic: actual_level = MaxFt - raw_sensor_value
    """
    df = df.copy()
    df['sensor_timestamp'] = pd.to_datetime(df['sensor_timestamp'])
    df = df.sort_values('sensor_timestamp')
    
    # ─── CORRECTED LEVEL LOGIC ───
    df['actual_level'] = max_ft - df['level_feet'] # level_feet is the raw value from DB
    
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
    """
    Saves industrial data to a hierarchical root 'data' folder.
    """
    full_path = os.path.join(DATA_BASE_DIR, relative_path)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    
    data_file = os.path.join(full_path, "data.csv")
    df.to_csv(data_file, index=False)
    logger.info(f"Industrial Fetch: Data snapshot stored in {data_file}")
    return data_file

def analyze_coldroom(asset_name: str):
    """
    Specialist: Fetches -> Stores (data/) -> Trains (if needed) -> Saves (model/) -> Analyzes.
    """
    asset_id = get_asset_id(asset_name)
    if not asset_id:
        return f"Error: {asset_name} is not a recognized ColdRoom asset."
    
    # ─── PHASE 2 OPTIMIZATION: CACHE CHECK BEFORE FETCH ───
    slug = slugify(asset_name)
    model_dir = os.path.join(MODEL_BASE_DIR, 'coldroom', slug)
    model_path = os.path.join(model_dir, 'model.h5')
    config_path = os.path.join(model_dir, 'config.pkl')
    
    is_cached = os.path.exists(model_path) and os.path.exists(config_path)
    fetch_days = 1 if is_cached else 7
    
    # 1. FETCH (Optimized window)
    df = fetch_coldroom_data(asset_id, days=fetch_days)
    if df is None or len(df) < TIME_STEPS:
        return f"Error: Insufficient data for {asset_name} ({len(df) if df is not None else 0} rows fetched)."

    # 2. STORE
    data_path = prepare_data_for_training(df, asset_name, os.path.join('coldroom', slug))
    
    # 3. TRAIN (Conditional)
    if is_cached:
        logger.info(f"Industrial Hub: Found cached model for {asset_name}. Skipping training.")
    else:
        if not os.path.exists(model_dir): os.makedirs(model_dir)
        logger.info(f"Industrial Hub: Initial Training for {asset_name} (7 days)...")
        try:
            train_results = train_coldroom(data_path, model_dir)
            if not train_results:
                return f"Error: Training failed for {asset_name}."
        except Exception as e:
            return f"Error triggering training: {e}"

    # 4. PREDICT
    try:
        start_time = time.time()
        model = load_artifact(model_path)
        scalar = load_artifact(os.path.join(model_dir, 'scalar.pkl')) or \
                 load_artifact(os.path.join(model_dir, 'scaler.pkl')) or \
                 load_artifact(os.path.join(model_dir, 'scaler.joblib'))
        config = load_artifact(config_path)
        
        threshold = config.get('threshold') if config else None
        if not threshold:
            th_val = load_artifact(os.path.join(model_dir, 'threshold.pkl')) or \
                     load_artifact(os.path.join(model_dir, 'threshold.joblib'))
            threshold = th_val['threshold'] if isinstance(th_val, dict) else th_val

        if model is None or scalar is None or threshold is None:
            return f"Error: Missing artifacts for {asset_name}."

        df = engineer_coldroom_features(df)
        df = df.dropna(subset=['temperature', 'humidity'])
        
        features = config.get('features', ["temperature", "humidity", "temp_diff", "rolling_mean", "rolling_std"])
        X_scaled = scalar.transform(df[features].values)
        X_room = np.array([X_scaled[-TIME_STEPS:]]) # Last sequence
        
        X_pred = model.predict(X_room, verbose=0)
        mse = float(np.mean(np.power(X_room - X_pred, 2)))
        
        is_anomaly = mse > threshold
        latest_row = df.iloc[-1]
        
        logger.info(f"Industrial Result: {asset_name} -> {'ANOMALY' if is_anomaly else 'NORMAL'} (MSE: {mse:.4f}, Inference Time: {time.time()-start_time:.2f}s)")
        
        return {
            "name": asset_name,
            "status": "Anomaly" if is_anomaly else "Normal",
            "latest_temp": float(latest_row["temperature"]),
            "humidity": float(latest_row["humidity"]),
            "anomaly": 1 if is_anomaly else 0,
            "threshold": float(threshold),
            "latest_mse": mse,
            "timestamp": str(latest_row["sensor_timestamp"])
        }
    except Exception as e:
        return f"Error during industrial analysis of {asset_name}: {e}"

def analyze_tank(asset_name: str):
    """
    Specialist: Fetches -> Stores (data/) -> Trains (if needed) -> Saves (model/) -> Analyzes.
    """
    asset_id = get_asset_id(asset_name)
    if not asset_id or asset_id not in TANK_MAPPINGS:
        return f"Error: {asset_name} is not a recognized Refinery Tank."
    
    # ─── PHASE 2 OPTIMIZATION: CACHE CHECK BEFORE FETCH ───
    import re
    tank_num_match = re.search(r'\d+', asset_name)
    tank_num = int(tank_num_match.group()) if tank_num_match else 0
    group_name = "tanks_1_to_6" if 1 <= tank_num <= 6 else "tanks_7_to_13"
    
    model_dir = os.path.join(MODEL_BASE_DIR, group_name, f"tank{tank_num}")
    model_path = os.path.join(model_dir, 'model.h5')
    config_path = os.path.join(model_dir, 'config.pkl')
    
    is_cached = os.path.exists(model_path) and os.path.exists(config_path)
    fetch_hours = 24 if is_cached else 168 # Parity with 1 day/7 days
    
    # 1. FETCH
    df = fetch_tank_data(asset_id, hours=fetch_hours)
    if df is None or len(df) < TIME_STEPS:
         return f"Error: Insufficient data for {asset_name} (Need {TIME_STEPS}, got {len(df) if df is not None else 0})."

    # 2. STORE
    relative_path = os.path.join('tanks', group_name, f"tank{tank_num}")
    data_path = prepare_data_for_training(df, asset_name, relative_path)
    
    # 3. TRAIN (Conditional)
    if is_cached:
        logger.info(f"Industrial Hub: Found cached model for {asset_name}. Skipping training.")
    else:
        if not os.path.exists(model_dir): os.makedirs(model_dir)
        logger.info(f"Industrial Hub: Retraining {asset_name} model (7 days)...")
        try:
            if 1 <= tank_num <= 6:
                train_results = train_tank_1_6(data_path, model_dir)
            else:
                train_results = train_tank_7_13(data_path, model_dir)
            if not train_results:
                 return f"Error: Tank Training failed for {asset_name}."
        except Exception as e:
            return f"Error triggering Tank Training: {e}"

    # 4. PREDICT
    try:
        start_time = time.time()
        model = load_artifact(model_path)
        scalar = load_artifact(os.path.join(model_dir, 'scalar.pkl')) or \
                 load_artifact(os.path.join(model_dir, 'scaler.pkl')) or \
                 load_artifact(os.path.join(model_dir, 'scaler.joblib'))
        config = load_artifact(config_path)
        
        if model is None or scalar is None or config is None:
             return f"Error: Missing artifacts for {asset_name}."

        threshold = config["threshold"]
        
        # ─── STABILIZED HEIGHT LOGIC ───
        # Use simple asset_name (e.g. 'tank7') for mapping to avoid hash ID confusion.
        max_ft = TANK_MAX_LEVELS.get(asset_name.lower().replace(" ", ""), MAX_FT_DEFAULT)
        
        df = engineer_tank_features(df, max_ft)
        df = df.dropna(subset=['actual_level'])
        df = df.fillna(0)
        
        features = config["features"]
        X_scaled = scalar.transform(df[features].values)
        X_seq = np.array([X_scaled[-TIME_STEPS:]])
        
        X_pred = model.predict(X_seq, verbose=0)
        mse = float(np.mean(np.power(X_seq - X_pred, 2)))
        
        is_anomaly = mse > threshold
        latest_row = df.iloc[-1]
        
        logger.info(f"Industrial Result: {asset_name} -> {'ANOMALY' if is_anomaly else 'NORMAL'} (MSE: {mse:.4f}, Inference Time: {time.time()-start_time:.2f}s)")
        
        return {
            "name": asset_name,
            "status": "Anomaly" if is_anomaly else "Normal",
            "latest_level": float(latest_row["actual_level"]),
            "max_capacity": f"{max_ft} ft",
            "operational_range": f"0 to {max_ft} ft",
            "anomaly": 1 if is_anomaly else 0,
            "threshold": float(threshold),
            "latest_mse": mse,
            "timestamp": str(latest_row["sensor_timestamp"])
        }
    except Exception as e:
        logger.error(f"Tank analysis failure: {e}")
        return f"Error during Tank analysis: {e}"
