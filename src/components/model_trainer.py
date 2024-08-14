import os, sys
from src.logger import logging
from src.exception import CustomException
import numpy as np
import pandas as pd
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

from src.utils import evaluate_models,save_object, save_json_object

@dataclass
class ModelTrainerConfig():
    model_path = os.path.join('.','model.pkl')
    model_report_path = os.path.join('.','model_report.json')

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Model Training Started')
            x_train,y_train,x_test,y_test = train_array[:,:-1], train_array[:,-1],test_array[:,:-1],test_array[:,-1]
            models = {
                "Logistic Regression":LogisticRegression(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Adaboost Classifier":AdaBoostClassifier(),
                "Gradient Boosting Classifier":GradientBoostingClassifier(verbose=1),
                "Random Forest Classifier": RandomForestClassifier(verbose=1),
                "Support Vector Machine": SVC(verbose=True),
                "K Nearest Neighbour":KNeighborsClassifier(),
                "Naive Bayes":GaussianNB(),
                "Catboost Classifier": CatBoostClassifier(verbose=1),
                "XGBoost Classifier": XGBClassifier()
            }

            params = {
                "Logistic Regression" : {
                    'penalty':['l2']
                },
                "Decision Tree": {
                    "max_depth":[10,20,30],
                    "min_samples_split":[2,5,10]
                },
                "Adaboost Classifier": {
                    "n_estimators":[100,150,200],
                    "learning_rate":[0.1,0.01,0.001]
                },
                "Gradient Boosting Classifier":{
                    "n_estimators":[100,150,200],
                    "max_depth":[10,20,30],
                    "learning_rate":[0.1,0.01,0.001]
                },
                "Random Forest Classifier":{
                    'n_estimators':[450],
                    'max_features':['log2'],
                    'max_depth':[340],
                    'min_samples_split':[3],
                    'min_samples_leaf':[8,10,12],
                    'criterion':['gini']
                },
                "Support Vector Machine":{
                    'kernel':['linear','poly','sigmoid','rbf'],
                    'gamma':['scale','auto']
                },
                "K Nearest Neighbour":{
                    'metric':['euclidean']
                },
                "Naive Bayes":{},
                "Catboost Classifier":{
                    'learning_rate':[0.1,0.01,0.001],
                    'depth':[10,20,30],
                    'iterations':[100,150,200],
                    'l2_leaf_reg':[2,3,4]
                },
                "XGBoost Classifier":{}
            }

            model_report:dict = evaluate_models(x_train,y_train,x_test,y_test,models,params)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found.",sys)
            save_object(filepath=self.model_trainer_config.model_path,obj=best_model)
            predicted = best_model.predict(x_test)
            accuracy = accuracy_score(y_test,predicted)
            logging.info(f"best model : {best_model_name} on both training and testing data with accuracy {accuracy}")
            save_json_object(filepath=self.model_trainer_config.model_report_path,obj=model_report)
            return(accuracy,best_model_name)
        except Exception as e:
            raise CustomException(e,sys)