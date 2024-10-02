"""
Utility module for model evaluation and object storage. This module includes
functions for evaluating multiple models, saving objects using dill, and 
saving JSON files.
"""
import os
import sys
import json
import dill
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from src.exception import CustomException
from src.logger import logging
def evaluate_models(x_train,y_train,x_test,y_test,models,params):
    """
    Evaluate multiple models using GridSearchCV and return their accuracy scores.

    Args:
        x_train (np.ndarray): The training features.
        y_train (np.ndarray): The training labels.
        x_test (np.ndarray): The testing features.
        y_test (np.ndarray): The testing labels.
        models (dict): A dictionary of models to evaluate.
        params (dict): A dictionary of hyperparameters for each model.

    Returns:
        dict: A dictionary with model names as keys and their accuracy scores as values.

    Raises:
        CustomException: If an error occurs during model evaluation.
    """
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            logging.info(f"Evaluation initiated for {model}.")
            param = params[list(models.keys())[i]]
            gs = GridSearchCV(model,param,cv=3,verbose=1,n_jobs=-1)
            logging.info(f"GridSearchCV initiated for {model}.")
            x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
            x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0)
            gs.fit(x_train,y_train)
            logging.info(f"GridSearchCV fit done and set_params initiated for {model}.")
            model.set_params(**gs.best_params_)
            logging.info(f"setting parameters completed and fitting initiated for {model}.")
            model.fit(x_train,y_train)
            logging.info(f"prediction initiated for {model}.")
            y_test_pred = model.predict(x_test)
            logging.info(f"Getting the accuracy for train and test data for {model}")
            test_model_accuracy = accuracy_score(y_true=y_test,y_pred=y_test_pred)
            report[list(models.keys())[i]] = test_model_accuracy
            logging.info(f"Obtained accuracy of {test_model_accuracy} and completed with {model}")
        return report
    except Exception as e:
        raise CustomException(e,sys) from e

def save_object(filepath,obj):
    """
    Save a Python object to a file using dill.

    Args:
        filepath (str): The path to save the object.
        obj (object): The Python object to save.

    Raises:
        CustomException: If an error occurs during saving.
    """
    try:
        dirpath = os.path.dirname(filepath)
        os.makedirs(dirpath,exist_ok=True)
        with open(filepath,'wb') as fileobj:
            dill.dump(obj,fileobj)
    except Exception as e:
        raise CustomException(e,sys) from e

def save_json_object(filepath,obj):
    """
    Save a dictionary as a JSON file.

    Args:
        filepath (str): The path to save the JSON file.
        obj (dict): The dictionary to save as JSON.

    Raises:
        CustomException: If an error occurs during saving.
    """
    try:
        dirpath = os.path.dirname(filepath)
        os.makedirs(dirpath,exist_ok=True)
        with open(filepath,'w',encoding='utf-8') as f:
            json.dump(obj,f)
    except Exception as e:
        raise CustomException(e,sys) from e
