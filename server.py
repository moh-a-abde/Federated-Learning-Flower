from collections import OrderedDict
from omegaconf import DictConfig
import torch
import xgboost as xgb
import numpy as np
import os

from model import XGBoostModel, test_xgboost_model, train_xgboost_model

def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round: int):
        return {'lr': config.lr, 'momentum': config.momentum,
                'local_epochs': config.local_epochs}
    return fit_config_fn

def get_evaluate_fn(num_classes: int, input_dim: int, testloader, model_path="model.xgb"):
    def evaluate_fn(server_round: int, parameters, config):
        # Prepare data for evaluation
        X_test, y_test = prepare_test_data(testloader)

        if not os.path.exists(model_path):
            # Train the model if it doesn't exist
            X_train, y_train = prepare_test_data(testloader)  # Use the same data for training for simplicity
            train_xgboost_model(X_train, y_train, num_classes, input_dim)

        # Load the model
        model = XGBoostModel(num_classes, input_dim)
        model.model = xgb.Booster(model_file=model_path)
        
        accuracy, report = test_xgboost_model(model, X_test, y_test)
        loss = 1 - accuracy  # Mock loss as XGBoost does not provide a direct loss value
        return loss, {'accuracy': accuracy}
    return evaluate_fn

def prepare_test_data(loader):
    features_list, labels_list = [], []
    for features, labels in loader:
        features_list.append(features.numpy())
        labels_list.append(labels.numpy())
    return np.vstack(features_list), np.hstack(labels_list)
