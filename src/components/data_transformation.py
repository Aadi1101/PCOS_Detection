"""
Module for data transformation. This module includes the DataTransformationConfig 
class for configuration and the DataTransformation class for handling the 
transformation process of training and testing datasets.
"""
import os
import sys
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """Configuration class for data transformation."""
    preprocessor_obj_file_path = os.path.join('.','preprocessor.pkl')

class DataTransformation():
    """
    Class for transforming data. This class handles loading, preprocessing,
    and scaling of training and testing datasets.
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def initiate_data_transformation(self,trainset,testset):
        """
    Scales the input features using StandardScaler.

    Args:
        input_feature_train_df (pd.DataFrame): The training features.
        input_feature_test_df (pd.DataFrame): The testing features.

    Returns:
        tuple: Scaled training and testing feature arrays.
    """
        try:
            logging.info('Data Transformation Started')
            train_df = pd.read_csv(trainset)
            test_df = pd.read_csv(testset)
            logging.info('Reading of training and testing data completed.')
            target_column_name = 'PCOS (Y/N)'

            logging.info('Handling missing values')
            train_df = train_df.dropna()
            test_df = test_df.dropna()
            train_df = train_df.dropna(axis=1)
            test_df  = test_df.dropna(axis=1)

            logging.info('Converting all the columns present in data to numeric format.')
            for column in train_df:
                train_df[column] = pd.to_numeric(train_df[column],errors='coerce')
            for column in test_df:
                test_df[column] = pd.to_numeric(test_df[column],errors='coerce')

            logging.info('Splitting the dataset into features and target')
            input_feature_train_df = train_df.drop([target_column_name],axis=1)
            input_feature_test_df = test_df.drop([target_column_name],axis=1)

            target_feature_train_df = train_df[target_column_name]
            target_feature_test_df = test_df[target_column_name]

            logging.info('Preprocessing the features by StandardScaler')
            sc = StandardScaler()
            input_feature_train_arr = sc.fit_transform(input_feature_train_df)
            input_feature_test_arr = sc.transform(input_feature_test_df)

            logging.info("Saving the preprocessor model")
            save_object(self.data_transformation_config.preprocessor_obj_file_path,obj=sc)

            logging.info('Converting dataframes to numpy arrays')
            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info('Data Transformation Complete.')
            return(
                train_arr,test_arr
            )
        except Exception as e:
            raise CustomException(e,sys) from e

    
    def selected_scaling(self,trainset,testset,selected_features):
        try:
            logging.info('Data Transformation Started for selected features')
            train_df = pd.read_csv(trainset,usecols=selected_features)
            test_df = pd.read_csv(testset,usecols=selected_features)
            logging.info('Reading of training and testing data completed for selected features.')

            logging.info('Handling missing values')
            train_df = train_df.dropna()
            test_df = test_df.dropna()
            train_df = train_df.dropna(axis=1)
            test_df  = test_df.dropna(axis=1)

            logging.info('Converting all the columns present in data to numeric format.')
            for column in train_df:
                train_df[column] = pd.to_numeric(train_df[column],errors='coerce')
            for column in test_df:
                test_df[column] = pd.to_numeric(test_df[column],errors='coerce')

            logging.info('Preprocessing the features by StandardScaler')
            sc = StandardScaler()
            input_feature_train_arr = sc.fit_transform(train_df)
            input_feature_test_arr = sc.transform(test_df)

            save_object(self.data_transformation_config.preprocessor_obj_file_path,obj=sc)
            logging.info("Saved the updated preprocessor model for selected features")

        except Exception as e:
            raise CustomException(e,sys)