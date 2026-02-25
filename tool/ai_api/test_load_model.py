import joblib
import sys
import os
import time

# Import necessary functions and classes from Ensemble_Methods
try:
    from Ensemble_Methods import preprocess_dataframe, DataFrameWrapper, has_content, ensure_dir
except ImportError:
    print("Could not import Ensemble_Methods")
    sys.exit(1)

model_path = "results/model_stats/2025-11-27_03-15-52_LinkedIn people profiles datasets - Clean_Original_label_data_StackingClassifier.pkl"
print(f"Loading model from {model_path}...")
start = time.time()
try:
    model = joblib.load(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")
end = time.time()
print(f"Loading took {end - start} seconds")
