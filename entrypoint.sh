#!/bin/bash

# Start the MLflow UI in the background
mlflow ui --host 0.0.0.0 --port 5000 &

# Start the Flask app
python src/predictdiabetes_app.py
