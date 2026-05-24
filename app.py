import streamlit as st
import pandas as pd
import numpy as np

# Import PitMind Modules
from src.data.master_pipeline import run_unified_pitmind_pipeline
from src.features.transcriber import transcribe_and_analyze_radio
from src.Simulator.biometrics import inject_synthetic_biometrics
from src.features.scaler import normalize_pitmind_data
from src.models.predict import calculate_stress_index
from src.models.nl_insights import generate_engineer_debrief

# UI Configuration
st.set_page_config(page_title="PitMind | AI Race Engineer", page_icon="🏎️", layout="wide")

st.title("🏎️ PitMind: AI Race Engineer Dashboard")
st.markdown("Real-time cognitive load tracking and Watsonx AI debriefs.")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Race Parameters")
    year = st.selectbox("Season Year", [2025, 2024, 2023], index=0)
    track = st.selectbox("Grand Prix", ["Monaco", "Silverstone", "Monza", "Spa", "Suzuka"])
    driver = st.selectbox("Driver Code", ["HAM", "VER", "LEC", "NOR", "PIA", "SAI", "RUS", "ALO"])
    
    run_pipeline = st.button("🚀 Run Live Telemetry Analysis", type="primary", use_container_width=True)

# --- MAIN EXECUTION PIPELINE ---
if run_pipeline:
    try:
        # Step 1: Data Extraction
        with st.status("📡 Downloading F1 Telemetry & Radio...", expanded=True) as status:
            st.write(f"Connecting to FastF1 and OpenF1 for {driver} at {track}...")
            telemetry_df, radio_df, race_control_df = run_unified_pitmind_pipeline(year, track, driver)
            
            if telemetry_df.empty:
                st.error("Failed to retrieve telemetry. Try a different session or track.")
                st.stop()

            # Step 2: NLP Transcription
            st.write("🎙️ Transcribing Team Radio via Whisper...")
            radio_df = transcribe_and_analyze_radio(radio_df)
            
            # Extract combined transcripts for the LLM
            radio_summary = " ".join(radio_df['transcript'].dropna().tolist()) if not radio_df.empty else "No radio communication detected."

            # Step 3: Biometric Simulation
            st.write("🧬 Injecting Synthetic Biometrics...")
            # If radio had an urgency flag, add a mock one to telemetry for the simulator to react to
            max_urgency = radio_df['audio_urgency_flag'].max() if not radio_df.empty and 'audio_urgency_flag' in radio_df.columns else 0
            telemetry_df['audio_urgency_flag'] = max_urgency 
            
            bio_df = inject_synthetic_biometrics(telemetry_df)

            # Step 4: ML Pre-processing
            st.write("📊 Normalizing features for model inference...")
            scaled_df = normalize_pitmind_data(bio_df, is_training=False)

            # Step 5: Stress Prediction
            st.write("🧠 Calculating Cognitive Stress Index...")
            final_df = calculate_stress_index(scaled_df)
            
            status.update(label="✅ Pipeline Execution Complete!", state="complete", expanded=False)

        # --- DASHBOARD VISUALIZATION ---
        
        # 1. Top Level Metrics
        st.subheader("Driver Physiological State")
        col1, col2, col3 = st.columns(3)
        
        overall_stress = final_df['pitmind_stress_index'].mean()
        max_stress = final_df['pitmind_stress_index'].max()
        avg_hr = final_df['heart_rate'].mean()
        
        col1.metric("Overall Lap Stress", f"{overall_stress:.1f} / 100")
        col2.metric("Peak Stress Index", f"{max_stress:.1f} / 100")
        col3.metric("Avg Heart Rate", f"{avg_hr:.0f} bpm")

        st.divider()

        # 2. Watsonx AI Race Engineer Debrief
        st.subheader("🤖 Watsonx AI Race Engineer Insights")
        with st.spinner("Granite-3 is analyzing the telemetry..."):
            ai_debrief = generate_engineer_debrief(
                driver=driver,
                track=track,
                stress_score=overall_stress,
                radio_summary=radio_summary
            )
            st.info(ai_debrief)

        st.divider()

        # 3. Telemetry Charting
        st.subheader("📈 Time-Series Telemetry")
        st.write("Driver Cognitive Load vs. Car Speed")
        
        # Plotting Speed and Stress on the same chart
        chart_data = final_df[['session_time', 'speed_kmh', 'pitmind_stress_index']].copy()
        chart_data = chart_data.set_index('session_time')
        st.line_chart(chart_data)

    except Exception as e:
        st.error(f"Pipeline Error: {str(e)}")
        st.write("Did you make sure to run `train.py` first to generate your physical_hr_baseline.pkl?")