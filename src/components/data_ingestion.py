import os, sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig():
    raw_path:str = os.path.join('artifacts','raw.csv')
    train_path:str = os.path.join('artifacts','train.csv')
    test_path:str = os.path.join('artifacts','test.csv')

class DataIngestion():
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info('Started Data Ingestion.')
            df = pd.read_csv('notebooks\PCOS_data_without_infertility - Full_new.csv')
            logging.info('Read the dataset')
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_path,index=False,header=True)

            logging.info('Train Test Split initiated.')
            train_set, test_set = train_test_split(df,test_size = 0.3, random_state = 0)
            train_set.to_csv(self.data_ingestion_config.train_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_path,index=False,header=True)

            logging.info('Ingestion of data completed.')
            return(
                self.data_ingestion_config.train_path,
                self.data_ingestion_config.test_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    obj = DataIngestion()
    train_path , test_path = obj.initiate_data_ingestion()
    transform_obj = DataTransformation()
    train_arr, test_arr = transform_obj.initiate_data_transformation(trainset=train_path,testset=test_path)
    model_trainer_obj = ModelTrainer()
    accuracy, best_model_name = model_trainer_obj.initiate_model_trainer(train_array=train_arr,test_array=test_arr)
    print(f"Best Model for the data is {best_model_name} with accuracy {accuracy}")