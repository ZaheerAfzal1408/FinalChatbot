import re
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def slugify(text):
    """
    Common slugification for asset names to be used in paths.
    """
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
    # actual_level is the real oil height (Max - AirGap)
    df['actual_level'] = max_ft - df['level_feet'] 
    
    df['fill_pct'] = (df['actual_level'] / max_ft) * 100
    df['roc'] = df['actual_level'].diff().fillna(0)
    df['roc_abs'] = df['roc'].abs()
    df['accel'] = df['roc'].diff().fillna(1e-9) # Avoid exact zero diffs for stability
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

def engineer_smoke_features(df):
    """
    Prepares smoke alarm features: temp, humi, bat_voltage, bat_percent.
    """
    df = df.copy()
    df['sensor_timestamp'] = pd.to_datetime(df['sensor_timestamp'])
    df = df.sort_values('sensor_timestamp')
    
    # Ensure numeric types for AI features
    features = ['temp', 'humi', 'bat_voltage', 'bat_percent']
    for feat in features:
        df[feat] = pd.to_numeric(df[feat], errors='coerce')
    
    return df.ffill().fillna(0)
