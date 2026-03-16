import os
import sys
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging


class DataTransformationConfig:

    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:

    def __init__(self):

        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformer_object(self):

        try:
            numerical_columns = [
"Open","High","Low","Close","Volume",
"Return","Momentum_10","SMA_10","SMA_50","Volatility",
"RSI","MACD","MACD_signal","BB_high","BB_low","ATR","VWAP"
]

           

            logging.info("Creating preprocessing pipeline")

            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )

            return num_pipeline, numerical_columns

        except Exception as e:

            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded")

            preprocessing_obj, numerical_columns = self.get_data_transformer_object()

            target_column = "Target"

            input_feature_train_df = train_df[numerical_columns]
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df[numerical_columns]
            target_feature_test_df = test_df[target_column]

            logging.info("Applying preprocessing")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save preprocessor object
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)

            with open(self.data_transformation_config.preprocessor_obj_file_path, "wb") as file_obj:
                pickle.dump(preprocessing_obj, file_obj)

            logging.info("Preprocessor saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:

            raise CustomException(e, sys)


if __name__ == "__main__":

    obj = DataTransformation()

    train_arr, test_arr, preprocessor_path = obj.initiate_data_transformation(
        train_path="artifacts/train.csv",
        test_path="artifacts/test.csv"
    )

    print("Train shape:", train_arr.shape)
    print("Test shape:", test_arr.shape)