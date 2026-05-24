import os
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

def generate_engineer_debrief(
    driver: str, 
    track: str, 
    stress_score: float, 
    radio_summary: str
) -> str:
    """
    Passes machine learning telemetry metrics into an IBM Granite foundation model
    to generate a natural language post-race debrief.
    """
    
    # 1. Authenticate with IBM Cloud
    # Best practice: Store these in a .env file, but for now we look for them in the OS
    api_key = os.environ.get("IBM_CLOUD_API_KEY", "your_ibm_api_key_here")
    project_id = os.environ.get("WATSONX_PROJECT_ID", "your_watsonx_project_id_here")
    
    credentials = Credentials(
        url="https://us-south.ml.cloud.ibm.com", # Change region if needed
        api_key=api_key
    )
    
    # 2. Set the guardrails and token parameters for the LLM
    parameters = {
        GenParams.DECODING_METHOD: "greedy", # We want analytical consistency, not creativity
        GenParams.MAX_NEW_TOKENS: 250,
        GenParams.REPETITION_PENALTY: 1.1
    }
    
    # 3. Load the Granite Model (IBM's highly efficient business model)
    model = ModelInference(
        model_id="ibm/granite-3-8b-instruct",
        params=parameters,
        credentials=credentials,
        project_id=project_id
    )
    
    # 4. Construct the Zero-Shot Prompt
    prompt = f"""You are a Principal Race Engineer for a Formula 1 team. 
Analyze the following telemetry and biometric ML predictions for {driver} at {track} and provide a concise, professional, 3-sentence summary of their cognitive state and driving quality.

DATA METRICS:
- Overall Stress Score: {stress_score}/100
- Radio Transcript Highlights: "{radio_summary}"

RACE ENGINEER SUMMARY:"""

    print("🧠 Requesting natural language insight from watsonx.ai...")
    
    try:
        response = model.generate_text(prompt=prompt)
        return response.strip()
    except Exception as e:
        return f"Error generating watsonx insight: {e}"

# --- TESTING HOOK ---
if __name__ == "__main__":
    # Ensure you set your API keys in your terminal first!
    # export IBM_CLOUD_API_KEY="xxx"
    # export WATSONX_PROJECT_ID="xxx"
    
    debrief = generate_engineer_debrief(
        driver="HAM",
        track="Monaco",
        stress_score=88.5,
        radio_summary="I have zero grip, the rears are gone."
    )
    print("\n🏎️ PITMIND ENGINEER DEBRIEF:")
    print(debrief)