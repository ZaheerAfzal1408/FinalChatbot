import pandas as pd
import numpy as np
import os
import logging
from ..train_utils import create_sequences, build_lstm_autoencoder, save_artifact
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

def train_tank_1_6(data_path, model_dir):
    """
    Trains/Retrains Tank LSTM Autoencoder for groups 1-6.
    """
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    df['sensor_timestamp'] = pd.to_datetime(df['sensor_timestamp'])
    df = df.sort_values('sensor_timestamp')
    df = df.dropna(subset=['level_feet'])
    if df.empty:
        logger.warning(f"No valid data (NaNs dropped) for {data_path}")
        return None
    
    MAX_FT = 25.0
    
    # Feature Engineering (Matching User's Specified Logic)
    group = df.copy()
    group['headspace'] = MAX_FT - group['level_feet']
    group['fill_pct'] = (group['level_feet'] / MAX_FT) * 100
    group['roc'] = group['level_feet'].diff()
    group['roc_abs'] = group['roc'].abs()
    group['accel'] = group['roc'].diff()
    group['roll_mean'] = group['level_feet'].rolling(60, min_periods=1).mean()
    group['roll_std'] = group['level_feet'].rolling(60, min_periods=1).std().fillna(0)
    group['roll_range'] = group['level_feet'].rolling(60, min_periods=1).max() - group['level_feet'].rolling(60, min_periods=1).min()
    group['dev_from_mean'] = group['level_feet'] - group['roll_mean']
    
    mu = group['level_feet'].mean()
    sig = group['level_feet'].std() + 1e-9
    group['z_score'] = (group['level_feet'] - mu) / sig
    
    # Time Features
    group['hour'] = group['sensor_timestamp'].dt.hour
    group['minute'] = group['sensor_timestamp'].dt.minute
    group['day_of_week'] = group['sensor_timestamp'].dt.dayofweek
    group['is_night'] = ((group['hour'] >= 22) | (group['hour'] <= 5)).astype(int)
    
    group = group.fillna(0)
    
    features = ["headspace", "fill_pct", "roc", "roc_abs", "accel", "roll_mean", "roll_std", "roll_range", "dev_from_mean", "z_score", "hour", "minute", "day_of_week", "is_night"]
    X = group[features].values
    
    # Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Sequencing
    TIME_STEPS = 30
    X_seq = create_sequences(X_scaled, TIME_STEPS)
    
    if len(X_seq) == 0:
        logger.warning(f"Not enough data for sequence: {data_path}")
        return None
    
    # Build & Train
    model = build_lstm_autoencoder((TIME_STEPS, X_scaled.shape[1]))
    model.fit(X_seq, X_seq, epochs=10, batch_size=64, validation_split=0.1, verbose=0)
    
    # Evaluation
    reconstructions = model.predict(X_seq)
    mse = np.mean(np.power(X_seq - reconstructions, 2), axis=(1, 2))
    threshold = np.mean(mse) + 3 * np.std(mse) 

    # Save Artifacts
    save_artifact(model, os.path.join(model_dir, 'model.h5'))
    save_artifact(scaler, os.path.join(model_dir, 'scaler.joblib'))
    
    config = {
        "seq_len": TIME_STEPS,
        "features": features,
        "threshold": float(threshold)
    }
    save_artifact(config, os.path.join(model_dir, 'config.joblib'))
    
    return {
        "mse": float(np.mean(mse)),
        "threshold": float(threshold),
        "last_mse": float(mse[-1])
    }
