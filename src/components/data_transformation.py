import os,sys
from src.logger import logging
from src.exception import CustomException
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class DataTransformation():
    def initiate_data_transformation(self,trainset,testset):
        try:
            logging.info('Data Transformation Started')
            train_df = pd.read_csv(trainset)
            test_df = pd.read_csv(testset)
            # print(f"Processing data : {train_df.columns[0]}")
            logging.info('Reading of training and testing data completed.')
            target_column_name = 'PCOS (Y/N)'

            logging.info('Handling missing values')
            # imputer = SimpleImputer(strategy="most_frequent")
            # train_df = imputer.fit_transform(train_df)
            # test_df = imputer.transform(test_df)
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
            # print("Data : ",input_feature_train_df)
            sc = StandardScaler()
            input_feature_train_arr = sc.fit_transform(input_feature_train_df)
            input_feature_test_arr = sc.transform(input_feature_test_df)

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
            raise CustomException(e,sys)