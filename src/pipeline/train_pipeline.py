import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__ == "__main__":

    try:

        logging.info("Training pipeline started")

        # Step 1 Data Ingestion
        data_ingestion = DataIngestion()

        train_data, test_data = data_ingestion.initiate_data_ingestion()

        print("Data Ingestion Completed")


        # Step 2 Data Transformation
        data_transformation = DataTransformation()

        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_data,
            test_data
        )

        print("Data Transformation Completed")


        # Step 3 Model Training
        model_trainer = ModelTrainer()

        best_model, score = model_trainer.initiate_model_trainer(
            train_arr,
            test_arr
        )

        print("Best Model:", best_model)
        print("Accuracy:", score)

        logging.info("Training pipeline completed")


    except Exception as e:

        raise CustomException(e, sys)