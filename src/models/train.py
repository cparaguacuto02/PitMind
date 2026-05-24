import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_baseline_model(historical_df: pd.DataFrame, output_dir: str = "src/models/saved_weights"):
    """
    Trains a Random Forest model to learn a driver's baseline heart rate 
    based on the physical load (speed, braking, g-forces).
    """
    if historical_df.empty:
        raise ValueError("Cannot train on an empty dataframe.")
        
    print("🧠 Initializing Random Forest Training Sequence...")
    
    # 1. Define the features (Inputs) and target (Output)
    features = ['speed_kmh', 'throttle_pct', 'brake_active', 'g_force_longitudinal', 'g_force_lat']
    target = 'heart_rate'
    
    # Ensure all required columns exist
    for col in features + [target]:
        if col not in historical_df.columns:
            raise KeyError(f"Missing required training column: {col}")
            
    X = historical_df[features]
    y = historical_df[target]
    
    # 2. Split into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Build and Train the Random Forest Regressor
    # We use a regressor because Heart Rate is a continuous number, not a category
    rf_model = RandomForestRegressor(
        n_estimators=100,      # Number of "trees" in the forest
        max_depth=10,          # How complex each tree is allowed to get
        random_state=42,
        n_jobs=-1              # Use all CPU cores for faster training
    )
    
    print("🌲 Fitting Random Forest to historical telemetry...")
    rf_model.fit(X_train, y_train)
    
    # 4. Evaluate the model's accuracy
    predictions = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    print(f"✅ Training Complete! Model Error (RMSE): +/- {rmse:.2f} bpm")
    
    # 5. Save the trained model to the pickle file
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "physical_hr_baseline.pkl")
    joblib.dump(rf_model, save_path)
    print(f"💾 Random Forest weights successfully saved to: {save_path}")

# --- STANDALONE RUN ---
if __name__ == "__main__":
    import numpy as np
    print("🧪 Generating synthetic telemetry data for baseline training...")
    
    # Create 200 rows of realistic fake telemetry to train the Random Forest
    np.random.seed(42)
    mock_data = pd.DataFrame({
        'speed_kmh': np.random.uniform(80, 340, 200),
        'throttle_pct': np.random.uniform(0, 100, 200),
        'brake_active': np.random.choice([0, 1], 200),
        'g_force_longitudinal': np.random.normal(0, 3, 200),
        'g_force_lat': np.random.normal(0, 4, 200)
    })
    
    # Synthesize what a human heart rate SHOULD look like under these physical loads
    base_hr = 110
    g_load = mock_data['g_force_longitudinal'].abs() + mock_data['g_force_lat'].abs()
    mock_data['heart_rate'] = base_hr + (mock_data['speed_kmh'] * 0.05) + (g_load * 4)
    
    # Run the training pipeline!
    train_baseline_model(mock_data)