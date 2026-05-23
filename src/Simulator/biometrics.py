import numpy as np
import pandas as pd

def inject_synthetic_biometrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates human physiological data (Heart Rate, HRV) 
    based on the physical load of the car (G-Force, Speed) and audio stress.
    """
    if df.empty:
        print("⚠️ Dataframe is empty. Skipping biometrics simulation.")
        return df
        
    syn_df = df.copy()
    
    # 1. Ensure we have G-Force data
    # FastF1 extraction handles longitudinal (accel/decel). We simulate lateral (cornering)
    # G-forces here to ensure our downstream scaler has all the columns it expects.
    if 'g_force_longitudinal' not in syn_df.columns:
        syn_df['g_force_longitudinal'] = 0.0
        
    if 'g_force_lat' not in syn_df.columns:
        # Simulate lateral Gs: higher variance during high-speed cornering
        syn_df['g_force_lat'] = np.random.normal(0, 1.5, len(syn_df))
        
    print("🧬 Calculating physiological load models...")

    # 2. Calculate Total Physical Load (Vector magnitude of G-Forces)
    # We use a rolling average because the human body takes a few seconds to react to stress
    total_g = np.sqrt(syn_df['g_force_longitudinal']**2 + syn_df['g_force_lat']**2)
    rolling_g = total_g.rolling(window=10, min_periods=1).mean()
    
    # 3. Simulate Heart Rate (bpm)
    # Baseline F1 driver HR is around 130-140 bpm. Spikes to 180+ under heavy load.
    base_hr = 135
    hr_g_response = rolling_g * 12 # E.g., 3Gs of load = +36 bpm
    hr_noise = np.random.normal(0, 2, len(syn_df)) # Natural heartbeat variance
    
    syn_df['heart_rate'] = np.clip(base_hr + hr_g_response + hr_noise, 80, 200).round(1)
    
    # 4. Simulate Heart Rate Variability (ms)
    # HRV drops when the sympathetic nervous system (fight or flight) activates.
    # High stress = Low HRV.
    base_hrv = 55
    hrv_stress_drop = rolling_g * 8 
    hrv_noise = np.random.normal(0, 3, len(syn_df))
    
    syn_df['hrv'] = np.clip(base_hrv - hrv_stress_drop + hrv_noise, 15, 80).round(1)
    
    # 5. Connect the Brain to the Radio (Cognitive Load)
    # If the whisper script flagged the driver as stressed on the radio, spike the heart rate!
    if 'audio_urgency_flag' in syn_df.columns:
        print("🗣️ Mapping vocal urgency to adrenaline spikes...")
        # A frantic radio message causes an immediate adrenaline spike (+15 bpm)
        audio_spike = syn_df['audio_urgency_flag'].fillna(0) * 15
        syn_df['heart_rate'] = np.clip(syn_df['heart_rate'] + audio_spike, 80, 200)
        syn_df['hrv'] = np.clip(syn_df['hrv'] - (audio_spike * 0.5), 10, 80)

    print("✅ Synthetic biometrics injected successfully.")
    return syn_df

# --- TESTING HOOK ---
if __name__ == "__main__":
    # Create a mock telemetry row with extreme braking (High G-force) and a stressed radio call
    mock_telemetry = pd.DataFrame({
        'date': ['2026-05-24T14:00:00Z', '2026-05-24T14:00:01Z'],
        'speed_kmh': [320, 150],
        'g_force_longitudinal': [-0.5, -4.5], # Heavy braking
        'audio_urgency_flag': [0, 1]          # Yelling on the radio
    })
    
    bio_df = inject_synthetic_biometrics(mock_telemetry)
    print("\n📝 Simulated Physiological Response:")
    print(bio_df[['g_force_longitudinal', 'audio_urgency_flag', 'heart_rate', 'hrv']])