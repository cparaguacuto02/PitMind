import os
import requests
import whisper
import pandas as pd

def extract_nlp_features(transcript: str) -> dict:
    """Converts raw text into lightweight numerical features for ML."""
    text = transcript.lower()
    
    # 1. Explicit Urgency (0 or 1)
    urgency_keywords = ["box", "stop", "now", "emergency", "crash", "puncture"]
    is_urgent = 1 if any(w in text for w in urgency_keywords) else 0
    
    # 2. Cognitive Overhead (Word count)
    word_count = len(text.split()) if text else 0
    
    # 3. Topic Category (0=Strategy/General, 1=Tires, 2=Mechanical)
    category = 0 
    if any(w in text for w in ["tire", "tyre", "grip", "slide", "soft", "medium"]):
        category = 1 
    elif any(w in text for w in ["engine", "power", "battery", "harvest", "brake"]):
        category = 2 
        
    return {
        "audio_urgency_flag": is_urgent,
        "audio_word_count": word_count,
        "audio_topic_category": category
    }

def transcribe_and_analyze_radio(radio_df: pd.DataFrame, model_size: str = "tiny") -> pd.DataFrame:
    """Downloads audio, transcribes it, and engineers NLP features."""
    if radio_df.empty:
        return radio_df
        
    print(f"🧠 Loading local Whisper '{model_size}' model...")
    model = whisper.load_model(model_size)
    
    # Create empty lists to hold our new columns
    transcripts = []
    urgency_flags = []
    word_counts = []
    topic_categories = []
    
    temp_dir = "data/audio_temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    for idx, row in radio_df.iterrows():
        url = row.get('recording_url')
        
        # Default empty values if no audio exists
        if pd.isna(url) or not str(url).endswith('.mp3'):
            transcripts.append("")
            urgency_flags.append(0)
            word_counts.append(0)
            topic_categories.append(0)
            continue
            
        local_path = os.path.join(temp_dir, url.split("/")[-1])
        
        try:
            # Download and transcribe
            response = requests.get(url, timeout=10)
            with open(local_path, "wb") as f:
                f.write(response.content)
                
            result = model.transcribe(local_path, language="en")
            text = result["text"].strip()
            
            # Extract numerical features instantly
            features = extract_nlp_features(text)
            
            transcripts.append(text)
            urgency_flags.append(features["audio_urgency_flag"])
            word_counts.append(features["audio_word_count"])
            topic_categories.append(features["audio_topic_category"])
            
        except Exception as e:
            print(f"❌ Failed to process audio: {e}")
            transcripts.append("ERROR")
            urgency_flags.append(0)
            word_counts.append(0)
            topic_categories.append(0)
            
        finally:
            if os.path.exists(local_path):
                os.remove(local_path)
                
    # Bind everything directly to your existing dataframe
    enhanced_df = radio_df.copy()
    enhanced_df['transcript'] = transcripts
    enhanced_df['audio_urgency_flag'] = urgency_flags
    enhanced_df['audio_word_count'] = word_counts
    enhanced_df['audio_topic_category'] = topic_categories
    
    print("✅ Whisper transcription and ML feature extraction complete.")
    return enhanced_df