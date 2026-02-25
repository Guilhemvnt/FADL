import sys
import json
import pandas as pd
import joblib
import os

# Import necessary functions and classes from Ensemble_Methods
try:
    from Ensemble_Methods import preprocess_dataframe, DataFrameWrapper, has_content, ensure_dir
except ImportError:
    # If strictly necessary, we can duplicate the class definition here if pickling has issues
    # with module paths, but usually it works if in the same directory.
    print(json.dumps({"error": "Could not import Ensemble_Methods"}))
    sys.exit(1)

def main():
    # Load the model
    model_path = "results/model_stats/2025-11-27_03-15-52_LinkedIn people profiles datasets - Clean_Original_label_data_StackingClassifier.pkl"
    
    if not os.path.exists(model_path):
        print(json.dumps({"error": f"Model file not found at {model_path}"}))
        sys.exit(1)

    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load model: {str(e)}"}))
        sys.exit(1)

    # Read input from stdin
    
    try:
        input_str = "/home/fasma/perso/FADL/dummy_input.json"
        if len(sys.argv) > 1:
            # Maybe passed as argument
            input_str = sys.argv[1]
        else:
            # Read from stdin
            exit(1)
        
        if not input_str.strip():
             print(json.dumps({"error": "Empty input"}))
             sys.exit(1)

        print(input_str)
        data = json.loads(input_str)
        
        # Ensure it's a list or wrap dict in list
        if isinstance(data, dict):
            data = [data]
        
        df = pd.DataFrame(data)
        
        # Helper for missing columns expected by preprocess_dataframe
        # define expected columns? preprocess_dataframe handles most.
        pass

    except Exception as e:
         print(json.dumps({"error": f"Failed to parse input: {str(e)}"}))
         sys.exit(1)

    # Preprocess
    try:
        processed_df = preprocess_dataframe(df)
        
        # Predict
        # Check if model expects specific features. 
        # The pipeline usually selects them if DataFrameWrapper is used.
        
        # StackingClassifier -> it might be the pipeline or just the classifier.
        # If it's a pipeline, 'predict' handles transformations.
        
        predictions = model.predict(processed_df)
        
        # Try to get probabilities
        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(processed_df)
            # Assuming binary classification, we want probability of class 1 (Genuine)
            # if the classes are [0, 1]
            if probabilities.shape[1] == 2:
                 probabilities = probabilities[:, 1]
        
        results = []
        for i, pred in enumerate(predictions):
            res = {
                "prediction": int(pred),
                "is_genuine": bool(pred == 1)
            }
            if probabilities is not None:
                res["probability"] = float(probabilities[i])
            results.append(res)

        # Output results
        print(json.dumps(results))

    except Exception as e:
        print(json.dumps({"error": f"Prediction failed: {str(e)}"}))
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
