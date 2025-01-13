import os
import yaml
import pandas as pd
from flask import Flask, request, jsonify
from train_model import train_model_with_gs
import mlflow

app = Flask(__name__)

params_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "params.yaml")
if not os.path.exists(params_path):
    raise FileNotFoundError(f"params.yaml not found at {params_path}")

# Load parameters from params.yaml
params = yaml.safe_load(open(params_path))["train"]

loaded_model_from_mlflow = None

# Extract necessary parameters from params.yaml
base_dir = os.path.dirname(os.path.abspath(__file__))  # Current directory (src)
data_dir = os.path.join(base_dir, "../data")  # Directory for 'data'
data_path = os.path.normpath(os.path.join(data_dir, params["data"]))  # Normalize the path
model_path = os.path.normpath(os.path.join(base_dir, "../models", params["model"]))  # Normalize the model path

random_state = params.get("random_state", None)
n_estimators = params.get("n_estimators", [100, 200])
max_depth = params.get("max_depth", [5, 10, None])


# Default Hyperparameter grid
default_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

@app.route('/mlops/retrain_on_demand', methods=['POST'])
def on_demand_retrain():
    """
    Endpoint to retrain the model on demand using the provided hyperparameter grid.

    :return: JSON response with best parameters and evaluation scores.
    """
    global data, default_param_grid

    # Get param_grid from request body, fall back to default if not provided
    param_grid = request.json.get('param_grid', default_param_grid)

    print(f"Data Path: {data_path}, Model Path: {model_path}, Param Grid: {param_grid}")

    try:
        # Train the model
        best_model, best_params, evaluation_scores, cv_scores = train_model_with_gs(
            data_path, model_path, param_grid)
        # Prepare response
        response = {
            "best_params": best_params,
            "evaluation_scores": evaluation_scores,
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": f"Model training failed: {str(e)}"}), 500

@app.route('/mlops/predict_diabetes_best_model_mlflow', methods=['POST'])
def predict_diabetes_best_model_mlflow():
    global loaded_model_from_mlflow

    try:
        # Parse input JSON
        data = request.get_json()
        features = data.get("features", None)

        if not features:
            return jsonify({"error": "No features provided"}), 400

        # Convert input features to DataFrame
        input_df = pd.DataFrame([features])

        # Set MLflow tracking URI and experiment
        mlflow.set_tracking_uri('http://localhost:5000')
        mlflow.set_experiment("Prima_Indians_Diabetes_Analysis")

        # Get the latest run for the experiment
        experiment = mlflow.get_experiment_by_name("Prima_Indians_Diabetes_Analysis")
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        latest_run = runs.sort_values(by="start_time", ascending=False).iloc[0]
        latest_run_id = latest_run['run_id']  # Extract the run ID of the latest run
        model_uri = f"runs:/{latest_run_id}/model"

        # Load the latest model if not already loaded
        if loaded_model_from_mlflow is None:
            loaded_model_from_mlflow = mlflow.sklearn.load_model(model_uri)

        # Make predictions
        predictions = loaded_model_from_mlflow.predict(input_df)

        # Prepare response
        response = {
            "predicted_outcome": translate_to_english(predictions[0])  # Convert numpy.int64 to int for JSON
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def translate_to_english(predicted_class):
    if predicted_class == 0:
        predicted_class_string = "Diabetes Negative"
    elif predicted_class == 1:
        predicted_class_string = "Diabetes Positive"
    else:
        predicted_class_string = "Diabetes Negative"
    return predicted_class_string

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
