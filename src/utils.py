import os,sys,dill,json
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np

def evaluate_models(x_train,y_train,x_test,y_test,models,params):
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
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            logging.info(f"Getting the accuracy for train and test data for {model}")
            train_model_accuracy = accuracy_score(y_true=y_train,y_pred=y_train_pred)
            test_model_accuracy = accuracy_score(y_true=y_test,y_pred=y_test_pred)
            report[list(models.keys())[i]] = test_model_accuracy
            logging.info(f"Obtained accuracy of {test_model_accuracy} and completed with {model}")
        return report
    except Exception as e:
        raise CustomException(e,sys)
    
def save_object(filepath,obj):
    try:
        dirpath = os.path.dirname(filepath)
        os.makedirs(dirpath,exist_ok=True)
        with open(filepath,'wb') as fileobj:
            dill.dump(obj,fileobj)
    except Exception as e:
        raise CustomException(e,sys)
    
def save_json_object(filepath,obj):
    try:
        dirpath = os.path.dirname(filepath)
        os.makedirs(dirpath,exist_ok=True)
        with open(filepath,'w') as f:
            json.dump(obj,f)
    except Exception as e:
        raise CustomException(e,sys)