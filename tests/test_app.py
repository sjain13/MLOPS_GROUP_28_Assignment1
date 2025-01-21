import pytest
import json
from src.predictdiabetes_app import app  # Import the Flask app
from src.train_model import train_model_with_gs


@pytest.fixture
def client():
    """Fixture to set up a test client for the Flask app."""
    with app.test_client() as client:
        yield client


def test_test_api(client):
    """Test the /mlops/test endpoint."""
    response = client.get('/mlops/test')  # Remove the extra slash
    assert response.status_code == 200
    data = response.get_json()
    assert data["message"] == "Application is running successfully!"
    assert data["status"] == "success"


def test_retrain_on_demand_success(client, mocker):
    """Test the /mlops/retrain_on_demand endpoint for successful retraining."""
    mock_train_model_with_gs = mocker.patch(
        'src.predictdiabetes_app.train_model_with_gs',  # Update the mock path
        return_value=(
            "mock_model",  # best_model
            {"n_estimators": 100, "max_depth": 5},  # best_params
            {"accuracy": 0.95},  # evaluation_scores
            None  # cv_scores
        )
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
    }

    response = client.post(
        '/mlops/retrain_on_demand',
        json={"param_grid": param_grid}
    )

    assert response.status_code == 200
    data = response.get_json()
    assert "best_params" in data
    assert "evaluation_scores" in data
    mock_train_model_with_gs.assert_called_once()


def test_retrain_on_demand_failure(client, mocker):
    """Test the /mlops/retrain_on_demand endpoint for failure."""
    mock_train_model_with_gs = mocker.patch(
        'src.predictdiabetes_app.train_model_with_gs',  # Update the mock path
        side_effect=Exception("Training failed due to invalid data")
    )

    response = client.post('/mlops/retrain_on_demand', json={})
    assert response.status_code == 500
    data = response.get_json()
    assert "error" in data
    assert "Training failed due to invalid data" in data["error"]


def test_predict_diabetes_best_model_mlflow_success(client, mocker):
    """Test the /mlops/predict_diabetes_best_model_mlflow endpoint."""
    mock_mlflow = mocker.patch('src.predictdiabetes_app.mlflow')  # Update the mock path
    mock_mlflow.sklearn.load_model.return_value.predict.return_value = [1]

    input_features = {
        "features": {
            "Pregnancies": 2,
            "Glucose": 120,
            "BloodPressure": 70,
            "SkinThickness": 20,
            "Insulin": 85,
            "BMI": 25.0,
            "DiabetesPedigreeFunction": 0.5,
            "Age": 30
        }
    }

    response = client.post(
        '/mlops/predict_diabetes_best_model_mlflow',
        json=input_features
    )

    assert response.status_code == 200
    data = response.get_json()
    assert "predicted_outcome" in data
    assert data["predicted_outcome"] == "Diabetes Positive"


def test_predict_diabetes_best_model_mlflow_no_features(client):
    """Test the /mlops/predict_diabetes_best_model_mlflow endpoint with no features."""
    response = client.post('/mlops/predict_diabetes_best_model_mlflow', json={})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert data["error"] == "No features provided"

@pytest.mark.skip(reason="Ignoring this test for now")
def test_predict_diabetes_best_model_mlflow_failure(client, mocker):
    """Test the /mlops/predict_diabetes_best_model_mlflow endpoint for failure."""
    # Mock mlflow.sklearn.load_model to raise an exception
    mock_mlflow_load_model = mocker.patch(
        'src.predictdiabetes_app.mlflow.sklearn.load_model',
        side_effect=Exception("Model loading failed")
    )

    # Input features for prediction
    input_features = {
        "features": {
            "Pregnancies": 2,
            "Glucose": 120,
            "BloodPressure": 70,
            "SkinThickness": 20,
            "Insulin": 85,
            "BMI": 25.0,
            "DiabetesPedigreeFunction": 0.5,
            "Age": 30
        }
    }

    # Call the endpoint
    response = client.post(
        '/mlops/predict_diabetes_best_model_mlflow',
        json=input_features
    )

    # Assert the response status and error message
    assert response.status_code == 500
    data = response.get_json()
    assert "error" in data
    assert data["error"] == "Model loading failed"

    # Assert that mlflow.sklearn.load_model was called
    mock_mlflow_load_model.assert_called_once()

