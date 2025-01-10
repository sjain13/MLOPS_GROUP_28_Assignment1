import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from mlflow.models import infer_signature
import os

from sklearn.model_selection import train_test_split,GridSearchCV
from urllib.parse import urlparse

import mlflow

def hyperparameter_tuning(X_train,y_train,param_grid):
    rf=RandomForestClassifier()
    grid_search=GridSearchCV(estimator=rf,param_grid=param_grid,cv=3,n_jobs=-1,verbose=2)
    grid_search.fit(X_train,y_train)
    return grid_search

def train_model_with_gs(data_path,model_path,param_grid):
    data=pd.read_csv(data_path)
    X=data.drop(columns=["Outcome"])
    y=data['Outcome']

    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")


    ## start the MLFLOW run
    with mlflow.start_run():
        # #split the dataset into training and test sets
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)
        signature=infer_signature(X_train,y_train)

        ## Define hyperparameter grid

        # param_grid = {
        #     'n_estimators': [100, 200],
        #     'max_depth': [5, 10, None],
        #     'min_samples_split': [2, 5],
        #     'min_samples_leaf': [1, 2]
        # }
        
        # Perform hyperparameter tuning
        grid_search=hyperparameter_tuning(X_train,y_train,param_grid)

        ## get the best model
        best_model=grid_search.best_estimator_

        ## predict and evaluate the model

        y_pred=best_model.predict(X_test)
        accuracy=accuracy_score(y_test,y_pred)
        print(f"Accuracy:{accuracy}")

         # Get classification report (for precision, recall, f1_score)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Prepare model scores dictionary
        model_scores = {
            "accuracy": accuracy,
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1_score": report["1"]["f1-score"]
        }

        # Log metrics to MLFlow
        mlflow.log_metric('accuracy', model_scores["accuracy"])
        mlflow.log_metric('precision', model_scores["precision"])
        mlflow.log_metric('recall', model_scores["recall"])
        mlflow.log_metric('f1_score', model_scores["f1_score"])

        # Log additional metrics
        mlflow.log_param("best_n_estimators", grid_search.best_params_['n_estimators'])
        mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
        mlflow.log_param("best_sample_split", grid_search.best_params_['min_samples_split'])
        mlflow.log_param("best_samples_leaf", grid_search.best_params_['min_samples_leaf'])

        # Log the confusion matrix and classification report as text
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(str(cr), "classification_report.txt")


        tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store!='file':
            mlflow.sklearn.log_model(best_model,"model",registered_model_name="Best Model for Prima Indians Diabetes")
        else:
            mlflow.sklearn.log_model(best_model, "model",signature=signature)

        ## create the directory to save the model
        os.makedirs(os.path.dirname(model_path),exist_ok=True)

        filename=model_path
        pickle.dump(best_model,open(filename,'wb'))

        print(f"Model saved to {model_path}")

        return best_model, grid_search.best_params_, model_scores, grid_search.cv_results_
