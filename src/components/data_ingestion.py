#datainegestion is reposible for :load dataset , train test split , save processed data into artifacts(artifacts folder stores intermediate datasets)
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
class DataIngestionConfig:
    train_data_path=os.path.join("artifacts","train.csv")
    test_data_path=os.path.join("artifacts","test.csv")
    raw_data_path=os.path.join("artifacts","data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("Enterd Data Ingestion method")
        try:
            df=pd.read_csv(r"C:\Users\Bhavani\OneDrive\Documents\Ml-project\ML_flow_Project\notebook\data\stock_data.csv")
            logging.info("Dataset Loaded")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split started")

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:

            raise CustomException(e,sys)
        
if __name__ == "__main__":

    obj = DataIngestion()

    train_data,test_data = obj.initiate_data_ingestion()

    print(train_data,test_data)


#python -m src.components.data_ingestion
