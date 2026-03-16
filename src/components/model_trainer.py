import os
import sys
import pickle
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.exception import CustomException
from src.logger import logging


class ModelTrainerConfig:

    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):

        try:

            logging.info("Splitting training and test data")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # Ensure MLflow logs inside project
            mlflow.set_tracking_uri("mlruns")
            mlflow.set_experiment("Stock_Buy_Sell_Prediction")

            models = {
                "LogisticRegression": LogisticRegression(max_iter=1000),
                "RandomForest": RandomForestClassifier(),
                "XGBoost": XGBClassifier(eval_metric="logloss")
            }

            model_report = {}
            trained_models = {}

            logging.info("Starting model training")

            for model_name, model in models.items():

                with mlflow.start_run(run_name=model_name):

                    logging.info(f"Training {model_name}")

                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)

                    # Evaluation metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)

                    # Log parameters
                    mlflow.log_param("model_name", model_name)

                    # Log metrics
                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_metric("precision", precision)
                    mlflow.log_metric("recall", recall)
                    mlflow.log_metric("f1_score", f1)

                    # Log model
                    mlflow.sklearn.log_model(model, "model")

                    model_report[model_name] = accuracy
                    trained_models[model_name] = model

                    logging.info(f"{model_name} Accuracy: {accuracy}")

                    # Feature importance for tree models
                    if hasattr(model, "feature_importances_"):

                        importance = model.feature_importances_

                        feature_names = [f"feature_{i}" for i in range(len(importance))]

                        feature_importance = pd.DataFrame({
                            "Feature": feature_names,
                            "Importance": importance
                        }).sort_values(by="Importance", ascending=False)

                        os.makedirs("artifacts", exist_ok=True)

                        feature_path = os.path.join("artifacts", "feature_importance.csv")

                        feature_importance.to_csv(feature_path, index=False)

                        mlflow.log_artifact(feature_path)

            # Select best model
            best_model_name = max(model_report, key=model_report.get)

            best_model = trained_models[best_model_name]
            best_score = model_report[best_model_name]

            logging.info(f"Best model selected: {best_model_name}")

            # Register best model in MLflow
            with mlflow.start_run(run_name="Best_Model"):

                mlflow.log_param("best_model", best_model_name)
                mlflow.log_metric("best_accuracy", best_score)

                mlflow.sklearn.log_model(
                    best_model,
                    name="stock_prediction_model",
                    registered_model_name="StockPredictionModel"
                )

            # Save best model locally
            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_file_path),
                exist_ok=True
            )

            with open(self.model_trainer_config.trained_model_file_path, "wb") as f:
                pickle.dump(best_model, f)

            logging.info("Best model saved successfully")

            return best_model_name, best_score


        except Exception as e:
            raise CustomException(e, sys)