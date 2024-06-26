from collections import OrderedDict
from typing import Dict
from flwr.common import NDArrays, Scalar
import torch
import flwr as fl
import xgboost as xgb
import numpy as np

from model import Net, train, test

import flwr as fl
import numpy as np

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, num_classes, input_dim) -> None:
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = Net(num_classes, input_dim)

    def get_parameters(self, config):
        return self.model.get_model_bytes()

    def set_parameters(self, parameters):
        self.model.set_model_bytes(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        epochs = config['local_epochs']
        self.model.train_model(self.trainloader, epochs)
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.test_model(self.valloader)
        return float(loss), len(self.valloader), {'accuracy': accuracy}

    def test_model(self, testloader):
        test_features, test_labels = self.model._loader_to_numpy(testloader)
        dtest = xgb.DMatrix(test_features)
        predictions = self.model.model.predict(dtest)
        predictions = np.argmax(predictions, axis=1)
        accuracy = (predictions == test_labels).mean()
        loss = 1 - accuracy  # Using error rate as loss
        return loss, accuracy

def generate_client_fn(trainloaders, valloaders, num_classes, input_dim):
    def client_fn(cid: str):
        return FlowerClient(trainloader=trainloaders[int(cid)],
                            valloader=valloaders[int(cid)],
                            num_classes=num_classes,
                            input_dim=input_dim)
    return client_fn
