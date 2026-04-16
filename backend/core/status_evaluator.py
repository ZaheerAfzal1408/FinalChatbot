import numpy as np
from datetime import datetime

def evaluate_coldroom_status(anomaly_detected, temp, is_night, is_weekend, is_peak, reconstruction_error, threshold):
    """
    Implements the intensity math from your JavaScript snippet for Coldrooms.
    """
    intense = 0
    if anomaly_detected:
        intense += 2
        if is_night:
            intense += 1
        if is_weekend:
            intense += 1
            
    if not is_peak and temp > 35:
        intense += 2
        
    if reconstruction_error > threshold * 0.8:
        intense += 2
        
    level = "Normal"
    if intense >= 4:
        level = "Critical"
    elif intense >= 2:
        level = "Warning"
        
    return intense, level

def evaluate_tank_status(anomaly_detected, anomalies, is_night):
    """
    Implements the intensity math from your JavaScript snippet for Tanks.
    """
    intense = 0
    if anomaly_detected:
        intense += 2
        
    # Check for specific dangerous types
    has_impossible = any(a.get('type') == 'impossible_value' for a in anomalies)
    has_spike = any(a.get('type') == 'sudden_spike' for a in anomalies)
    has_fault = any(a.get('type') == 'sensor_fault' for a in anomalies)
    
    if has_impossible:
        intense += 2
    if has_spike:
        intense += 1
    if has_fault:
        intense += 1
        
    if anomaly_detected and is_night:
        intense += 1
        
    # Mapping to Level
    level = "Normal"
    if intense >= 4:
        level = "Critical"
    elif intense >= 2:
        level = "Warning"
        
    return intense, level

def evaluate_smoke_status(incident_detected, warn_string, temp, humi, bat_v, mse, threshold):
    """
    Implements the intensity math for Smoke Alarms.
    Factors: AI Deviations, Hardware Warnings, and Environmental Spikes.
    """
    intense = 0
    
    # 1. AI Deviation Logic
    if incident_detected:
        intense += 2
    elif mse > (threshold * 0.8):
        intense += 1 # Pre-incident warning
        
    # 2. Hardware Warnings (String-based)
    if warn_string in ['warn', 'fault', 'remove']:
        intense += 3
    elif warn_string == 'low-vol' or bat_v < 2.8:
        intense += 1
        
    # 3. Environmental Spikes
    if temp > 35:
        intense += 2
        
    # Mapping to Level
    level = "Normal"
    if intense >= 4:
        level = "Critical"
    elif intense >= 2:
        level = "Warning"
        
    return intense, level
