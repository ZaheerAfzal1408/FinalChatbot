import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
import time
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, LSTM, Input, RepeatVector, TimeDistributed

# Unified Backend Imports
from core import asset_mapping as am
from core.shared_utils import slugify, engineer_smoke_features
from core.status_evaluator import evaluate_smoke_status

# Force CPU optimization
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

logger = logging.getLogger(__name__)

# Constants
SMOKE_TIME_STEPS = 12
SMOKE_MODEL_DIR = "models/smoke_alarms"
SMOKE_DATA_DIR = "data/smoke_alarms"

def prepare_smoke_data(df, asset_name, relative_path):
    """Saves smoke telemetry locally for training context."""
    base = os.path.join(SMOKE_DATA_DIR, relative_path)
    if not os.path.exists(base):
        os.makedirs(base, exist_ok=True)
    df_sorted = df.sort_values(by=['sensor_node', 'sensor_timestamp'])
    csv_path = os.path.join(base, "data.csv")
    df_sorted.to_csv(csv_path, index=False)
    return csv_path

def train_smoke_model(df, model_dir, scaler_path, model_path, config_path):
    """Trains a one-time LSTM Autoencoder for a clean smoke baseline."""
    from sklearn.preprocessing import MinMaxScaler
    
    # Filter for 'ok' status to build a healthy baseline
    train_df = df[df['warn'].isin(['ok', 'mute', 'ok-vol-test'])]
    if len(train_df) < SMOKE_TIME_STEPS * 2:
        train_df = df # Fallback to all data if filter is too strict
        
    threshold = 0.05
    scaler = MinMaxScaler()
    features = ['temp', 'humi', 'bat_voltage', 'bat_percent']
    scaler.fit(train_df[features].fillna(0))
    joblib.dump(scaler, scaler_path)
    joblib.dump({'threshold': threshold, 'features': features}, config_path)
    
    # LSTM Autoencoder Architecture
    inputs = Input(shape=(SMOKE_TIME_STEPS, 4))
    x = LSTM(16, activation='relu')(inputs)
    x = RepeatVector(SMOKE_TIME_STEPS)(x)
    x = LSTM(16, activation='relu', return_sequences=True)(x)
    outputs = TimeDistributed(Dense(4))(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    
    X_train = []
    for _, group in train_df.groupby('sensor_node'):
        if len(group) < SMOKE_TIME_STEPS: continue
        scaled = scaler.transform(group[features].fillna(0))
        for i in range(len(scaled) - SMOKE_TIME_STEPS + 1):
            X_train.append(scaled[i : i + SMOKE_TIME_STEPS])
            
    X_train = np.array(X_train)
    if len(X_train) > 0:
        model.fit(X_train, X_train, epochs=25, batch_size=8, verbose=0)
    model.save(model_path)
    return model, threshold

def analyze_smoke_incident(asset_name: str, force_retrain: bool = False, fetch_hours: int = None):
    """Hierarchical Analyst for Smoke Alarm Network."""
    asset_id = am.get_asset_id(asset_name)
    if not asset_id: return f"Error: {asset_name} unknown."

    zone_name = am.get_asset_zone(asset_id)
    slug = slugify(asset_name)
    model_dir = os.path.join(SMOKE_MODEL_DIR, zone_name, slug)
    model_path = os.path.join(model_dir, 'model.h5')
    scaler_path = os.path.join(model_dir, 'scalar.pkl')
    config_path = os.path.join(model_dir, 'config.pkl')

    if force_retrain and os.path.exists(model_dir):
        import shutil
        shutil.rmtree(model_dir)

    is_cached = os.path.exists(model_path) and os.path.exists(scaler_path)
    
    if fetch_hours is None:
        fetch_hours = 24 * 30 # Default 30 days
    
    df_raw = am.fetch_smoke_data(asset_id, days=max(1, fetch_hours // 24))
    if df_raw is None or len(df_raw) < SMOKE_TIME_STEPS:
        return f"Error: Insufficient data for {asset_name}."

    prepare_smoke_data(df_raw, asset_name, os.path.join(zone_name, slug))

    if not is_cached:
        if not os.path.exists(model_dir): os.makedirs(model_dir)
        logger.info(f"Training Smoke Baseline: {zone_name}/{asset_name}")
        train_smoke_model(df_raw, model_dir, scaler_path, model_path, config_path)

    try:
        model = load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)
        config = joblib.load(config_path)
        threshold = config.get('threshold', 0.05)
        
        df = engineer_smoke_features(df_raw)
        features = ['temp', 'humi', 'bat_voltage', 'bat_percent']
        sensor_reports = []
        
        for s_node, group in df.groupby('sensor_node'):
            if len(group) < SMOKE_TIME_STEPS: continue
            
            scaled_vals = scaler.transform(group[features].values)
            X_seq = np.array([scaled_vals[-SMOKE_TIME_STEPS:]])
            X_pred = model.predict(X_seq, verbose=0)
            mse = float(np.mean(np.square(X_seq - X_pred)))
            
            # Status Evaluation
            warn_str = str(group['warn'].iloc[-1]).lower()
            latest = group.iloc[-1]
            intensity, status = evaluate_smoke_status(
                incident_detected=(mse > threshold),
                warn_string=warn_str,
                temp=float(latest["temp"]),
                humi=float(latest["humi"]),
                bat_v=float(latest["bat_voltage"]),
                mse=mse,
                threshold=threshold
            )
            
            sensor_reports.append({
                "name": f"{asset_name} ({s_node})",
                "zone": zone_name,
                "status": status,
                "intensity": intensity,
                "anomaly": 1 if status != "Normal" else 0,
                "latest_temp": float(latest["temp"]),
                "latest_humi": float(latest["humi"]),
                "bat_v": float(latest["bat_voltage"]),
                "mse": mse,
                "timestamp": str(latest["sensor_timestamp"])
            })

        return sensor_reports if sensor_reports else f"Buffer: {asset_name} telemetry is sparse."
    except Exception as e:
        logger.error(f"Smoke Analysis Exception: {e}")
        return f"Error: {e}"

def scan_all_smoke_alarms(fetch_hours: int = None):
    """Global Safety Scanner for all Smoke Alarm assets."""
    if not am.SMOKE_MAPPINGS: am.load_dynamic_mappings()
    all_reports = []
    incidents = []
    for aid, name in am.SMOKE_MAPPINGS.items():
        res = analyze_smoke_incident(name, fetch_hours=fetch_hours)
        if isinstance(res, list):
            all_reports.extend(res)
            incidents.extend([r for r in res if r['anomaly']])
    return {"summary": f"Found {len(incidents)} fire safety incidents.", "incidents": incidents, "all_reports": all_reports}
