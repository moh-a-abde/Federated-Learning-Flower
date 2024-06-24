from collections import OrderedDict
from typing import Dict
from flwr.common import NDArrays, Scalar
import torch
import flwr as fl
import numpy as np
import xgboost as xgb
from model import XGBoostModel, train_xgboost_model, test_xgboost_model

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, num_classes, input_dim) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.model = None

    def set_parameters(self, parameters):
        # This can be left empty as XGBoost does not need this method
        pass

    def get_parameters(self, config: Dict[str, Scalar]):
        # This can be left empty as XGBoost does not need this method
        return []

    def fit(self, parameters, config):
        # Prepare data for training
        X_train, y_train = self._prepare_data(self.trainloader)
        
        # Train the XGBoost model
        self.model = train_xgboost_model(X_train, y_train, self.num_classes, self.input_dim)

        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        X_val, y_val = self._prepare_data(self.valloader)
        accuracy, report = test_xgboost_model(self.model, X_val, y_val)
        loss = 1 - accuracy  # Mock loss as XGBoost does not provide a direct loss value
        return float(loss), len(self.valloader), {'accuracy': accuracy}

    def _prepare_data(self, loader):
        features_list, labels_list = [], []
        for features, labels in loader:
            features_list.append(features.numpy())
            labels_list.append(labels.numpy())
        return np.vstack(features_list), np.hstack(labels_list)

def generate_client_fn(trainloaders, valloaders, num_classes, input_dim):
    def client_fn(cid: str):
        return FlowerClient(trainloader=trainloaders[int(cid)],
                            valloader=valloaders[int(cid)],
                            num_classes=num_classes,
                            input_dim=input_dim)
    return client_fn
