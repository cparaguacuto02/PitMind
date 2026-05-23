import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def normalize_pitmind_data(
    df: pd.DataFrame, 
    output_dir: str = "data/processed",
    is_training: bool = True
) -> pd.DataFrame:
    """
    Scales and normalizes multi-modal telemetry and biometric data 
    to prepare it for machine learning architectures.
    """
    if df.empty:
        print("⚠️ Dataframe is empty. Skipping normalization.")
        return df

    processed_df = df.copy()
    os.makedirs(output_dir, exist_ok=True)

    # 1. Classify features by their distribution type
    minmax_features = ['Throttle', 'Brake'] # Strict bounds [0, 100]
    standard_features = ['Speed', 'g_force_long', 'g_force_lat', 'heart_rate', 'hrv'] # Normal curves

    # 2. Handle Scaling Paths (Fit during training, Load during live inference)
    minmax_scaler_path = os.path.join(output_dir, "minmax_scaler.pkl")
    standard_scaler_path = os.path.join(output_dir, "standard_scaler.pkl")

    if is_training:
        print("🏋️ Training Mode: Fitting new scalers on data patterns...")
        mm_scaler = MinMaxScaler(feature_range=(0, 1))
        st_scaler = StandardScaler()

        # Fit the scalers to learn the min/max and mean/variance
        mm_scaler.fit(processed_df[minmax_features])
        st_scaler.fit(processed_df[standard_features])

        # Save them to disk so your app.py can reuse them for live data
        joblib.dump(mm_scaler, minmax_scaler_path)
        joblib.dump(st_scaler, standard_scaler_path)
    else:
        print("🚀 Inference Mode: Loading saved scaling parameters from disk...")
        if not os.path.exists(minmax_scaler_path) or not os.path.exists(standard_scaler_path):
            raise FileNotFoundError("Scalers not found. You must run training mode first!")
        mm_scaler = joblib.load(minmax_scaler_path)
        st_scaler = joblib.load(standard_scaler_path)

    # 3. Transform the columns
    print("📊 Executing mathematical transformations...")
    processed_df[minmax_features] = mm_scaler.transform(processed_df[minmax_features])
    processed_df[standard_features] = st_scaler.transform(processed_df[standard_features])

    # 4. Save the final normalized telemetry table
    final_csv_path = os.path.join(output_dir, "normalized_telemetry.csv")
    processed_df.to_csv(final_csv_path, index=False)
    print(f"✨ Success! Normalized dataset cached at: {final_csv_path}")

    return processed_df

if __name__ == "__main__":
    # Mock row to demonstrate transformation values
    mock_data = pd.DataFrame({
        'date': ['2026-05-24T14:00:00Z'],
        'Throttle': [85.0],
        'Brake': [0.0],
        'Speed': [280.0],
        'g_force_long': [1.2],
        'g_force_lat': [3.5],
        'heart_rate': [145.0],
        'hrv': [42.0]
    })
    
    transformed_df = normalize_pitmind_data(mock_data, is_training=True)
    print("\n📝 Transformed Feature Preview:")
    print(transformed_df[ ['Throttle', 'Speed', 'heart_rate'] ])