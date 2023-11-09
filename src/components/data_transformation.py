import os,sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass


@dataclass
class DataTransformationConfig():
    preprocessor_file_path = os.path.join('src/models','preprocessor.pkl')
    cat_preprocessor_file_path = os.path.join('src/models','cat_preprocessor.pkl')

class DataTransformation():
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self,trainset_path,testset_path):
        try:
            train_df = pd.read_csv(trainset_path)
            test_df = pd.read_csv(testset_path)

            logging.info("Reading the train and test data completed.")
            train_df=train_df.dropna(axis=1)
            test_df = test_df.dropna(axis=1)
            logging.info("Converting categorical data to numeric")

            for column in train_df:
                train_df[column] = pd.to_numeric(train_df[column],errors="coerce")
            
            for column in test_df:
                test_df[column] = pd.to_numeric(test_df[column],errors="coerce")
            target_column_name = "PCOS (Y/N)"

            logging.info("Splitting the features and target column.")

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            train_arr = np.c_[
                np.array(input_feature_train_df), np.array(target_feature_train_df)
            ]
            test_arr = np.c_[np.array(input_feature_test_df), np.array(target_feature_test_df)]


            logging.info("Data Transformation complete")
            return(
                train_arr,test_arr
            )
        except Exception as e:
            raise CustomException(e,sys)