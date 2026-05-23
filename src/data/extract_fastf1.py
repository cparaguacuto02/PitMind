import os
import fastf1
import pandas as pd

def extract_driver_lap_telemetry(
    year: int, 
    gp_name: str, 
    session_type: str, 
    driver_code: str, 
    cache_dir: str = "data/f1_cache"
) -> pd.DataFrame:
 
    # 1. Initialize local disk caching
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    fastf1.Cache.enable_cache(cache_dir)
    
    print(f"🏎️ Loading {year} {gp_name} Grand Prix ({session_type})...")
    
    try:
        # 2. Fetch the session and load core telemetry arrays
        session = fastf1.get_session(year, gp_name, session_type)
        session.load(telemetry=True, laps=True, weather=False)
        
        # 3. Filter down to your chosen driver
        driver_laps = session.laps.pick_driver(driver_code)
        if driver_laps.empty:
            raise ValueError(f"Driver '{driver_code}' not found in this session.")
            
        # 4. Get the fastest single lap baseline
        fastest_lap = driver_laps.pick_fastest()
        lap_number = fastest_lap['LapNumber']
        print(f"⏱️ Found fastest lap for {driver_code}: Lap #{int(lap_number)} (Time: {fastest_lap['LapTime']})")
        
        # 5. Extract the full time-series telemetry array
        # .add_distance() is essential for spatial track modeling mapping later
        telemetry = fastest_lap.get_telemetry().add_distance()
        
        # 6. Isolate and rename the essential columns needed for PitMind
        # We drop heavy geometric asset channels to keep the footprint lightweight
        essential_columns = {
            'Date': 'date',                  # Crucial Master UTC Timestamp
            'SessionTime': 'session_time',    # Time relative to session start
            'Distance': 'track_distance_m',   # Meters completed in lap
            'Speed': 'speed_kmh',             # Car speed
            'Throttle': 'throttle_pct',       # 0 - 100 pedal scale
            'Brake': 'brake_active',          # True/False or boolean pressure indicator
            'RPM': 'engine_rpm',
            'nGear': 'gear'
        }
        
        processed_df = telemetry[list(essential_columns.keys())].copy()
        processed_df.rename(columns=essential_columns, inplace=True)
        
        # 7. Synthesize approximate linear G-forces from velocity changes
        # Helps our rule engine capture heavy deceleration/acceleration load states
        time_delta_sec = processed_df['date'].diff().dt.total_seconds()
        speed_mps = processed_df['speed_kmh'] / 3.6
        delta_v = speed_mps.diff()
        
        # Accel/Decel G-Force = delta_v / (time * 9.81 m/s^2)
        processed_df['g_force_longitudinal'] = (delta_v / (time_delta_sec * 9.81)).fillna(0).round(2)
        
        print(f"🎉 Successfully extracted {len(processed_df)} telemetry rows for {driver_code}!")
        return processed_df
        
    except Exception as e:
        print(f"❌ Failed to extract FastF1 Data: {e}")
        return pd.DataFrame()

# --- STANDALONE TESTING RUN ---
if __name__ == "__main__":
    # Create target save path directories if testing locally
    os.makedirs("data/raw", exist_ok=True)
    
    # Extract Hamilton's Qualifying Telemetry from Monaco 2025
    df = extract_driver_lap_telemetry(
        year=2025, 
        gp_name="Monaco", 
        session_type="Q", 
        driver_code="HAM"
    )
    
    if not df.empty:
        # Save asset to local disk
        save_path = "data/raw/fastf1_telemetry_ham_monaco.csv"
        df.to_csv(save_path, index=False)
        print(f"💾 Saved raw telemetry mapping vector directly to: {save_path}")
        print(df.head())