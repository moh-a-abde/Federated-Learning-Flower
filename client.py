from collections import OrderedDict
from typing import Dict, List, Any
from flwr.common import NDArrays, Scalar
import flwr as fl
import model
from torch.utils.data import DataLoader
import ray
from model import train_xgboost

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloaders, valloaders) -> None:
        super().__init__()
        self.trainloaders = trainloaders
        self.valloaders = valloaders
        # Initialize the model
        self.model = train_xgboost(trainloaders)
        print(f"Model initialized: {self.model}")

    def fit(self, parameters, config):
        # do local training
        self.model = model.train_xgboost(self.trainloaders)
        return {
            "parameters": [],  # Placeholder for actual parameters
            "num_examples": len(self.trainloaders.dataset),
            "metrics": {}
        }

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        accuracy = 0.99  # Placeholder for actual accuracy
        return {
            "loss": 0.0,
            "num_examples": len(self.valloaders.dataset),
            "metrics": {'accuracy': accuracy}
        }

def generate_client_fn(trainloaders, valloaders):
    def client_fn(cid: str):
        cid_int = int(cid)
        print(f"Creating client {cid_int}")
        if cid_int >= len(trainloaders):
            raise ValueError(f"Client ID {cid_int} is out of bounds for trainloaders of length {len(trainloaders)}")
        return FlowerClient(trainloaders=trainloaders[cid_int],
                            valloaders=valloaders[cid_int])
    return client_fn
