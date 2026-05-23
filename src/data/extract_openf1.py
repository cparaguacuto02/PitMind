import os
import json
import requests
import pandas as pd

def fetch_openf1_data(endpoint: str, params: dict, cache_path: str) -> list:
    """
    Helper function to query OpenF1 endpoints with a local JSON disk cache fallback.
    """
    # If the file has already been downloaded, load it instantly from disk
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            print(f"📦 Loading cached OpenF1 data from: {cache_path}")
            return json.load(f)
            
    # Otherwise, execute a live HTTP request to the OpenF1 network servers
    url = f"https://api.openf1.org/v1/{endpoint}"
    print(f"🌐 Fetching fresh live data from OpenF1: {url}...")
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # Save the result to our local data cache folder
        with open(cache_path, 'w') as f:
            json.dump(data, f)
        return data
    except requests.exceptions.RequestException as e:
        print(f"❌ OpenF1 API connection error: {e}")
        return []

def extract_session_context_data(
    session_key: int, 
    driver_number: int, 
    cache_dir: str = "data/openf1_cache"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts team radio events and global race control messages 
    for a given F1 session.
    
    Returns:
    --------
    tuple: (radio_df, race_control_df)
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    # --- 1. EXTRACT TEAM RADIO LOGS ---
    radio_params = {"session_key": session_key, "driver_number": driver_number}
    radio_cache_file = os.path.join(cache_dir, f"radio_s{session_key}_d{driver_number}.json")
    
    raw_radio = fetch_openf1_data("team_radio", radio_params, radio_cache_file)
    
    if raw_radio:
        radio_df = pd.DataFrame(raw_radio)
        # Standardize the timezone-aware string into a clean Pandas datetime object
        radio_df['date'] = pd.to_datetime(radio_df['date'])
        # Isolate the core columns relevant to the PitMind project metrics
        radio_df = radio_df[['date', 'session_key', 'driver_number', 'recording_url']].copy()
    else:
        radio_df = pd.DataFrame(columns=['date', 'session_key', 'driver_number', 'recording_url'])

    # --- 2. EXTRACT RACE CONTROL TIMELINES ---
    # Race control impacts all drivers globally, so we do not filter by driver_number
    rc_params = {"session_key": session_key}
    rc_cache_file = os.path.join(cache_dir, f"race_control_s{session_key}.json")
    
    raw_rc = fetch_openf1_data("race_control", rc_params, rc_cache_file)
    
    if raw_rc:
        rc_df = pd.DataFrame(raw_rc)
        rc_df['date'] = pd.to_datetime(rc_df['date'])
        # Focus on the categorical indicators: flag, sector, and literal message text
        rc_columns = ['date', 'session_key', 'category', 'flag', 'sector', 'message']
        # Ensure fallback safety keys if specific attributes don't populate
        for col in rc_columns:
            if col not in rc_df.columns:
                rc_df[col] = None
        rc_df = rc_df[rc_columns].copy()
    else:
        rc_df = pd.DataFrame(columns=['date', 'session_key', 'category', 'flag', 'sector', 'message'])

    print(f"🎉 OpenF1 Extraction Complete: Mapped {len(radio_df)} radio clips and {len(rc_df)} track events.")
    return radio_df, rc_df

# --- TESTING HOOK ---
if __name__ == "__main__":
    # Test using a valid historical F1 session key
    # Session 9158 corresponds to a real Grand Prix weekend event profile
    test_session = 9158
    test_driver = 44 # Lewis Hamilton
    
    radio_logs, track_incidents = extract_session_context_data(
        session_key=test_session, 
        driver_number=test_driver
    )
    
    print("\n📝 Sample Team Radio Logs:")
    print(radio_logs.head(2))
    
    print("\n🚩 Sample Race Control Messages:")
    print(track_incidents.head(2))