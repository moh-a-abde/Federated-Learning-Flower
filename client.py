from collections import OrderedDict
from typing import Dict, List, Any
from flwr.common import NDArrays, Scalar
import flwr as fl
import model
from torch.utils.data import DataLoader
import ray
from model import train_xgboost

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, testloader) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        # Initialize the model
        self.model = train_xgboost(trainloader)
        print(f"Model initialized: {self.model}")

    def set_parameters(self, parameters):
        if self.model is not None:
            param_dict = {k: v for k, v in zip(self.model.feature_names, parameters)}
            self.model.set_attr(**param_dict)
        else:
            raise ValueError("Model is not initialized")

    def get_parameters(self, config: Dict[str, Scalar]):
        if self.model is not None:
            return [self.model.attr(name) for name in self.model.feature_names]
        else:
            raise ValueError("Model is not initialized")

    def fit(self, parameters, config):
        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        self.model = model.train_xgboost(self.trainloader)
        # do local training
        train_xgboost(self.trainloader)
        return {
            "parameters": [],
            "num_examples": len(self.trainloader.dataset),
            "metrics": {}
        }

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        accuracy = 0.99  # Placeholder for actual accuracy
        self.set_parameters(parameters)
        return {
            "loss": 0.0,
            "num_examples": len(self.valloader.dataset),
            "metrics": {'accuracy': accuracy}
        }

def generate_client_fn(trainloaders, valloaders, testloader):
    def client_fn(cid: str):
        cid_int = int(cid)
        print(f"Creating client {cid_int}")
        if cid_int >= len(trainloaders):
            raise ValueError(f"Client ID {cid_int} is out of bounds for trainloaders of length {len(trainloaders)}")
        return FlowerClient(trainloader=trainloaders[cid_int],
                            valloader=valloaders[cid_int],
                            testloader=testloader)
    return client_fn
