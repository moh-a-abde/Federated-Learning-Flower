from collections import OrderedDict
from typing import Dict
from flwr.common import NDArrays, Scalar
import torch
import flwr as fl
import torch.optim as optim
import xgboost as xgb
import pickle

from model import Net, train, test

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, num_classes, input_dim) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = Net(num_classes, input_dim)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        # Deserialize the parameters (assuming they are stored in a dictionary format)
        self.model.model = xgb.Booster()
        self.model.model.load_model(parameters)

    def get_parameters(self, config: Dict[str, Scalar]):
        # Serialize the model parameters
        return self.model.model.save_raw()

    def fit(self, parameters, config):
        # Set the parameters
        self.set_parameters(parameters)
        lr = config['lr']
        momentum = config['momentum']
        epochs = config['local_epochs']
        # Dummy optimizer (not used in XGBoost)
        optim = None
        train(self.model, self.trainloader, optim, epochs, self.device)
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader), {'accuracy': accuracy}

def generate_client_fn(trainloaders, valloaders, num_classes, input_dim):
    def client_fn(cid: str):
        return FlowerClient(trainloader=trainloaders[int(cid)],
                            valloader=valloaders[int(cid)],
                            num_classes=num_classes,
                            input_dim=input_dim)
    return client_fn
