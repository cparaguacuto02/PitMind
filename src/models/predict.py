import os
import joblib
import pandas as pd
import numpy as np

def calculate_stress_index(live_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates live telemetry to calculate cognitive overload.
    """
    model_path = "src/models/saved_weights/physical_hr_baseline.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Trained model not found. Run train.py first!")
        
    # 1. Load the trained brain
    model = joblib.load(model_path)
    
    # 2. Extract the features it needs to make a prediction
    physical_features = ['speed_kmh', 'throttle_pct', 'brake_active', 'g_force_longitudinal', 'g_force_lat']
    X_live = live_df[physical_features]
    
    # 3. Predict the Baseline (What the HR *should* be physically)
    predicted_physical_hr = model.predict(X_live)
    
    output_df = live_df.copy()
    output_df['predicted_base_hr'] = predicted_physical_hr
    
    # 4. Calculate the Delta (The Unexplained Stress)
    # We use absolute difference. If HR is drastically higher OR lower than expected, it indicates stress/shock.
    hr_residual = np.abs(output_df['heart_rate'] - output_df['predicted_base_hr'])
    
    # 5. Build the Final PitMind Stress Index (0 to 100 Scale)
    # We combine the physiological anomaly (hr_residual) with the NLP vocal stress flag
    
    # Base stress from the heart rate deviation
    raw_stress = hr_residual * 50 
    
    # Apply a heavy multiplier if Whisper caught them yelling or using urgent keywords on the radio
    if 'audio_urgency_flag' in output_df.columns:
        nlp_multiplier = 1.0 + (output_df['audio_urgency_flag'] * 0.5) # +50% penalty for urgent radio calls
        raw_stress = raw_stress * nlp_multiplier
        
    # Normalize the final score to a clean 0-100 gauge for the Streamlit UI
    output_df['pitmind_stress_index'] = np.clip(raw_stress, 0, 100).round(1)
    
    return output_df