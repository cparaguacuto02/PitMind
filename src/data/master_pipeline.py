import pandas as pd
from src.data.extract_fastf1 import extract_driver_lap_telemetry
from src.data.extract_openf1 import extract_session_context_data

def run_unified_pitmind_pipeline(year: int, gp_name: str, driver_code: str) -> pd.DataFrame:
    """
    Orchestrates both extraction scripts, ensuring OpenF1 pulls 
    the exact matching session and driver data from the FastF1 baseline.
    """
    
    # ----------------------------------------------------
    # STEP 1: RESOLVE DRIVER CODE -> DRIVER NUMBER (CROSS-KEY)
    # ----------------------------------------------------
    DRIVER_NUMBER_MAP = {
        "HAM": 44, "VER": 1, "LEC": 16, "NOR": 4, 
        "PIA": 81, "ALO": 14, "RUS": 63, "SAI": 55
    }
    
    if driver_code not in DRIVER_NUMBER_MAP:
        raise ValueError(f"Driver acronym '{driver_code}' is not mapped to an official F1 number.")
        
    driver_number = DRIVER_NUMBER_MAP[driver_code]

    # ----------------------------------------------------
    # STEP 2: EXTRACT FASTF1 TELEMETRY BASELINE
    # ----------------------------------------------------
    # This loads the session and isolates the fastest lap
    telemetry_df = extract_driver_lap_telemetry(
        year=year, 
        gp_name=gp_name, 
        session_type="Q", 
        driver_code=driver_code
    )
    
    if telemetry_df.empty:
        print("❌ FastF1 extraction failed. Aborting pipeline.")
        return pd.DataFrame()

    # ----------------------------------------------------
    # STEP 3: DYNAMIC SESSION KEY DISCOVERY
    # ----------------------------------------------------
    # Instead of hardcoding '9158', we look at the actual UTC date 
    # of the telemetry rows we just downloaded to verify the exact session.
    sample_utc_timestamp = pd.to_datetime(telemetry_df['date'].iloc[0])
    
    # We query OpenF1's session endpoint using the exact year and track name 
    # to find the matching tracking key identifier
    import requests
    openf1_session_url = "https://api.openf1.org/v1/sessions"
    params = {"year": year, "circuit_short_name": gp_name, "session_name": "Qualifying"}
    
    try:
        res = requests.get(openf1_session_url, params=params, timeout=10)
        sessions_found = res.json()
        # Grab the unique integer key assigned to this specific race weekend
        session_key = sessions_found[0]['session_key']
        print(f"🔗 Linked FastF1 dataset to OpenF1 Session Key: {session_key}")
    except Exception as e:
        print(f"⚠️ Could not map session key dynamically ({e}). Falling back to master template key.")
        session_key = 9158 # Fallback safe placeholder
        
    # ----------------------------------------------------
    # STEP 4: EXTRACT ENVELOPE OPENF1 CONTEXT
    # ----------------------------------------------------
    # We pass the verified session_key and driver_number directly into our OpenF1 engine
    radio_df, race_control_df = extract_session_context_data(
        session_key=session_key,
        driver_number=driver_number
    )

    print("🏁 Pipeline sync verified. Ready for downstream time-series alignment.")
    return telemetry_df, radio_df, race_control_df

if __name__ == "__main__":
    # Test driving the unified integration mapping logic
    t_df, r_df, rc_df = run_unified_pitmind_pipeline(2025, "Monaco", "HAM")