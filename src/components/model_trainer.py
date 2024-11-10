"""
Module for model training. This module includes the ModelTrainerConfig 
class for configuration and the ModelTrainer class for handling the 
training process of multiple machine learning models.
"""
import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_models,save_object, save_json_object

@dataclass
class ModelTrainerConfig():
    """Configuration class for model training paths."""
    model_path = os.path.join('.','model.pkl')
    model_report_path = os.path.join('.','model_report.json')

class ModelTrainer():
    """
    Class for training machine learning models. This class handles the 
    training process, evaluates various models, and saves the best model.
    """
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        """
        Initiates the model training process.

        Args:
            train_array (np.ndarray): The training data as a 2D array, 
                                      where the last column is the target variable.
            test_array (np.ndarray): The testing data as a 2D array, 
                                     where the last column is the target variable.

        Returns:
            tuple: A tuple containing the accuracy of the best model and its name.

        Raises:
            CustomException: If no suitable model is found or any other exception \
                occurs during the process.
        """
        try:
            logging.info('Model Training Started')
            x_train,y_train,x_test,y_test = train_array[:,:-1], train_array[:,-1],\
                test_array[:,:-1],test_array[:,-1]
            models = {
                "Logistic Regression":LogisticRegression(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Adaboost Classifier":AdaBoostClassifier(),
                "Random Forest Classifier": RandomForestClassifier(verbose=1),
                "Support Vector Machine": SVC(verbose=True,probability=True),
                "Naive Bayes":GaussianNB(),
                "K Nearest Neighbour":KNeighborsClassifier(),
                "XGBoost Classifier": XGBClassifier(),
            }

            params = {
                "Logistic Regression" : {
                    'penalty':['l2']
                },
                "Decision Tree": {
                    "max_depth":[20,30,40,None],
                    "min_samples_split":[2,5,10,20],
                    "min_samples_leaf":[1,5,10,None]
                },
                "Adaboost Classifier": {
                    "n_estimators":[200,300,400],
                    "learning_rate":[0.001,0.01,0.05]
                },
                "Random Forest Classifier":{
                    'n_estimators':[500,700],
                    'max_features':['sqrt','log2'],
                    'max_depth':[300,400],
                    'min_samples_split':[2,5],
                    'min_samples_leaf':[1,10],
                    'criterion':['gini','entropy']
                },
                "Support Vector Machine":{
                    'kernel':['linear','poly','sigmoid','rbf'],
                    'gamma':['scale','auto',0.01,0.1]
                },
                "K Nearest Neighbour":{
                    'n_neighbors':[5,10,20],
                    'metric':['euclidean','manhatten']
                },
                "Naive Bayes":{},
                "XGBoost Classifier":{
                    'n_estimators':[200,300,400],
                    'max_depth':[10,20,30],
                    'learning_rate':[0.001,0.01,0.05]
                }
            }
            
            model_report:dict = evaluate_models(x_train,y_train,x_test,y_test,models,params,epsilon=0.5,alpha=0.01,gamma=0.99)
            best_model_name = max(model_report,key=lambda name: model_report[name]["best_test_accuracy"])
            best_model_score = model_report[best_model_name]["best_test_accuracy"]
            best_model = models[best_model_name]
            selected_features = model_report[best_model_name]["best_features"]
            # for i in range(len(selected_features)):
            #     selected_features[i] += 20
            print("Selected features for best model ", selected_features)
            # selected_columns = [0,4,5,6,7,10,14,17,18,19] + [20,21,23,24,38,39]
            # x_test = x_test[:,selected_features]
            best_model = best_model.__class__(**model_report[best_model_name]["best_params"])
            x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
            x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0)
            best_model.fit(x_train[:,selected_features],y_train)

            if best_model_score < 0.6:
                raise CustomException("No best model found.",sys)
            save_object(filepath=self.model_trainer_config.model_path,obj=best_model)
            predicted = best_model.predict(x_test[:,selected_features])
            accuracy = accuracy_score(y_test,predicted)
            logging.info(f"best model : {best_model_name} on both training and testing data with\
                          accuracy {accuracy}")
            save_json_object(filepath=self.model_trainer_config.model_report_path,obj=model_report)
            return(accuracy,best_model_name,selected_features)
        except Exception as e:
            raise CustomException(e,sys) from e
