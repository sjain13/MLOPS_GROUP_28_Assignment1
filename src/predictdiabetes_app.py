import os
import yaml
import pandas as pd
from flask import Flask, request, jsonify
from train_model import train_model_with_gs

app = Flask(__name__)

params_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "params.yaml")
if not os.path.exists(params_path):
    raise FileNotFoundError(f"params.yaml not found at {params_path}")

# Load parameters from params.yaml
params = yaml.safe_load(open(params_path))["train"]



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


if __name__ == '__main__':
    app.run(debug=True, port=5001)
