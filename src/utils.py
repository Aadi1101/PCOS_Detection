"""
Utility module for model evaluation and object storage. This module includes
functions for evaluating multiple models, saving objects using dill, and 
saving JSON files.
"""
import os
import sys
import json
import random
import dill
import numpy as np
from collections import defaultdict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier
from src.exception import CustomException
from src.logger import logging
def evaluate_models(x_train,y_train,x_test,y_test,models,param,epsilon,alpha,gamma):
    """
    Evaluate multiple models using GridSearchCV and return their accuracy scores.

    Args:
        x_train (np.ndarray): The training features.
        y_train (np.ndarray): The training labels.
        x_test (np.ndarray): The testing features.
        y_test (np.ndarray): The testing labels.
        models (dict): A dictionary of models to evaluate.
        params (dict): A dictionary of hyperparameters for each model.
        n_features (int): The number of features in the dataset, representing the dimensionality of each input sample.
        epsilon (float): The exploration rate used in reinforcement learning algorithms, controlling the balance between exploration and exploitation.
        - Values close to 1 promote exploration (random actions), while values close to 0 favor exploitation (best-known actions).
        alpha (float): The learning rate in reinforcement learning, defining how quickly the agent updates its knowledge with new information.
        - A high alpha value can lead to rapid learning but may risk overshooting optimal solutions.
        gamma (float): The discount factor in reinforcement learning, determining the importance of future rewards.
        - Values close to 1 prioritize long-term rewards, while values near 0 focus on immediate rewards.

    Returns:
        dict: A dictionary with model names as keys and their accuracy scores as values.

    Raises:
        CustomException: If an error occurs during model evaluation.
    """
    try:
        # selected_columns = [0,1,8,9,12,17,18,19]
        # selected_columns = [20,27,28,29,30,31,32,36,37]
        # Best Params for Decision Tree - Best params: {'max_depth': 10, 'min_samples_split': 2}
        # selected_columns = [0,1,8,9,12,17,18,19] + [20,27,28,29,30,31,32,36,37]
        # n_features = len(selected_columns)
        # x_train = x_train[:,selected_columns]
        # x_test = x_test[:,selected_columns]
        # Initialize the Q-table for reinforcement learning (state-action table)
        # q_table = np.zeros((2**n_features, n_features))
        report = {}
        q_table = defaultdict(int)
        best_model_accuracy = 0
        n_features = x_train.shape[1]
        print(n_features)

        # Iterate through each model
        for model_name, model in models.items():
            logging.info(f"Evaluation initiated for {model_name}.")
            para = param[model_name]
            best_model_test_accuracy = 0  # Track the best model accuracy per model
            best_features = []
            final_params = {}  # Track the best feature set per model
            epsilon = 0.5
            alpha = 0.01
            gamma = 0.99

            for episode in range(11):
                # Start with no features selected
                feature_selection = [0] * n_features
                state = get_state(feature_selection)  # Get initial state based on feature selection
                logging.info(f"Episode {episode + 1}/{11} started for {model_name}. Initial state: {state}.")

                for step in range(15):  # Set a maximum number of steps per episode
                    # Choose an action and toggle feature selection
                    action = choose_action(state, epsilon, n_features, q_table)
                    feature_selection[action] = 1 - feature_selection[action]  # Toggle feature selection
                    next_state = get_state(feature_selection)
                    logging.info(f"Step {step + 1}: Action {action} chosen, toggling feature {action}. Updated feature selection: {feature_selection}.")

                    # Collect selected features based on the current feature selection
                    selected_features = [i for i in range(n_features) if feature_selection[i] == 1]

                    if not selected_features:
                        logging.info("No features selected, skipping model training for this step.")
                        test_model_accuracy = 0  # If no features selected, set accuracy to 0
                    else:
                        # Use GridSearchCV to find the best hyperparameters for selected features
                        gs = GridSearchCV(model, para, cv=5, verbose=1, n_jobs=-1)
                        logging.info(f"GridSearchCV initiated for {model_name} with selected features: {selected_features}.")
                        x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
                        x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0)
                        gs.fit(x_train[:, selected_features], y_train)
                        best_params = gs.best_params_
                        updated_model = model.__class__(**best_params)
                        # Update model with the best parameters and train on selected features
                        if isinstance(model, CatBoostClassifier):
                            updated_model = CatBoostClassifier(**best_params)
                        else:
                            updated_model = model.set_params(**best_params)
                        logging.info(f"Training {model_name} with best params on selected features.")
                        updated_model.fit(x_train[:, selected_features], y_train)

                        # Predict and evaluate model performance on test data
                        print("utils selected features: ",selected_features)
                        y_train_pred = updated_model.predict(x_train[:,selected_features])
                        y_test_pred = updated_model.predict(x_test[:, selected_features])
                        test_model_accuracy = accuracy_score(y_test, y_test_pred)
                        train_model_accuracy = accuracy_score(y_train,y_train_pred)
                        precision = precision_score(y_test, updated_model.predict(x_test[:,selected_features]), average='binary',zero_division=0)
                        recall = recall_score(y_test, updated_model.predict(x_test[:,selected_features]), average='binary')
                        f1 = f1_score(y_test, updated_model.predict(x_test[:,selected_features]), average='binary')
                        roc_auc = roc_auc_score(y_test, updated_model.predict_proba(x_test[:,selected_features])[:, 1])


                    # Update Q-table based on the test accuracy as reward
                    update_q_value(state, action, test_model_accuracy, next_state, q_table, alpha, gamma)
                    logging.info(f"Q-value updated for state {state}, action {action} with reward {test_model_accuracy}.")
                    state = next_state  # Move to the next state

                    # Update best accuracy and feature set if this configuration performs better
                    if test_model_accuracy > best_model_test_accuracy:
                        best_model_test_accuracy = test_model_accuracy
                        best_model_train_accuracy = train_model_accuracy
                        best_features = selected_features
                        final_params = best_params
                        best_precision = precision
                        best_recall = recall
                        best_f1 = f1
                        best_roc_auc = roc_auc 
                    print(best_features)

                # Decay epsilon after each episode to reduce exploration over time
                epsilon = max(0.01, epsilon * 0.99)
                logging.info(f"Epsilon decayed to {epsilon} after episode {episode + 1} for {model_name}.")

            # Store the best accuracy and feature set for each model in the report
            report[model_name] = {
                "best_test_accuracy": best_model_test_accuracy,
                "best_train_accuracy": best_model_train_accuracy,  # Ensure correct key matching
                "accuracy_variance": best_model_train_accuracy - best_model_test_accuracy,  # For overfitting/underfitting check
                "precision": best_precision,  # Classification metric
                "recall": best_recall,        # Classification metric
                "f1_score": best_f1,          # Classification metric
                "roc_auc_score": best_roc_auc,  # Useful for binary classification
                "best_features": best_features,  # Already existing key
                "best_params":final_params
            }
            logging.info(f"Completed evaluation for {model_name} with best accuracy {best_model_accuracy}. Best params: {best_params}, Best features: {best_features}.")
            logging.info(f"{report[model_name]}")

        return report
    except Exception as e:
        raise CustomException(e, sys)

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
    
def get_state(feature_selection):
    try:
        return sum([int(f) * (2**i) for i,f in enumerate(feature_selection)])
    except Exception as e:
        raise CustomException(e,sys)

def choose_action(state,epsilon,n_features,q_table):
    try:
        if random.uniform(0,1) < epsilon:
            return random.randint(0,n_features-1)
        return np.argmax(q_table[state])
    except Exception as e:
        raise CustomException(e,sys)

def update_q_value(state,action,reward,next_state,q_table,alpha,gamma):
    try:
        best_next_action = np.argmax(q_table[next_state])
        q_table[state,action] = (1-alpha) * q_table[state,action] + alpha * (reward + gamma * q_table[next_state,best_next_action])
    except Exception as e:
        raise CustomException(e,sys)
