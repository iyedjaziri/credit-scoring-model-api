import pickle
import pandas as pd
import numpy as np
import sys

def verify_model():
    model_path = 'model/model.pkl'
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {model_path}")
        print(f"Type: {type(model)}")
        
        # Attempt to inspect features
        if hasattr(model, 'feature_name_'):
            print("\nExpected Features:")
            print(model.feature_name_)
        elif hasattr(model, 'feature_names_in_'):
             print("\nExpected Features:")
             print(model.feature_names_in_)
        
        # Create dummy data (1 row)
        # We need to know the features to create valid dummy data. 
        # For now, we'll try to create a random dataframe if we can get feature names, 
        # otherwise we might fail on prediction which is expected for this first run.
        
        if hasattr(model, 'n_features_in_'):
            n_features = model.n_features_in_
            print(f"\nNumber of features expected: {n_features}")
            
            # Load real data sample
            data_path = 'data/processed/train_prepared.csv'
            try:
                print(f"Loading sample data from {data_path}...")
                df = pd.read_csv(data_path, nrows=100) # Load more rows to ensure some categories are present if needed, though for 1 row prediction it might be tricky with OHE. 
                # Actually, for OHE to work similarly, we might need to know the columns. 
                # But since we align afterwards, it's fine.
                
                # Preprocessing
                import re
                
                # 1. Drop identifiers and target
                cols_to_drop = [c for c in ['TARGET', 'SK_ID_CURR'] if c in df.columns]
                df = df.drop(columns=cols_to_drop)
                
                # 2. One-Hot Encoding
                df = pd.get_dummies(df)
                
                # 3. Sanitize column names (LightGBM requirement)
                df = df.rename(columns = lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
                
                # 4. Align with model features
                if hasattr(model, 'feature_name_'):
                    expected_cols = model.feature_name_
                    
                    # Add missing columns with 0
                    for col in expected_cols:
                        if col not in df.columns:
                            df[col] = 0
                            
                    # Drop extra columns
                    df = df[expected_cols]
                    
                    # Select first row for prediction
                    dummy_data = df.iloc[[0]]
                else:
                    # Fallback if no feature names (unlikely for LGBM)
                    dummy_data = df.iloc[[0], :n_features]
                
                print("\nInput Data Sample (Processed):")
                print(dummy_data)
                
            except Exception as e:
                print(f"Failed to load/process real data: {e}")
                print("Falling back to random data for shape check...")
                dummy_data = pd.DataFrame(np.random.rand(1, n_features))
            
            try:
                prediction = model.predict(dummy_data)
                print(f"\nPrediction successful: {prediction}")
                print(f"Prediction Type: {type(prediction)}")
            except Exception as e:
                print(f"\nPrediction failed: {e}")
                print("Tip: Check data types vs model expectations.")

    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_model()
