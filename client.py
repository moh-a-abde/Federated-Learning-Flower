from typing import Dict, List, Any
from flwr.common import NDArrays, Scalar
import flwr as fl
import model
from torch.utils.data import DataLoader
import ray

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader: DataLoader, valloader: DataLoader, testloader: DataLoader):
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.model = None

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Dict[str, Any]:
        self.model = model.train_xgboost(self.trainloader)
        return {
            "parameters": [],
            "num_examples": len(self.trainloader.dataset),
            "metrics": {}
        }

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Dict[str, Any]:
        accuracy = 0.99  # Placeholder for actual accuracy
        return {
            "loss": 0.0,
            "num_examples": len(self.valloader.dataset),
            "metrics": {'accuracy': accuracy}
        }

def generate_client_fn(trainloaders: List[DataLoader], valloaders: List[DataLoader], testloader: DataLoader):
    def client_fn(cid: str):
        cid_int = int(cid)
        print(f"Creating client {cid_int}")
        if cid_int >= len(trainloaders):
            raise ValueError(f"Client ID {cid_int} is out of bounds for trainloaders of length {len(trainloaders)}")
        return FlowerClient(trainloader=trainloaders[cid_int],
                            valloader=valloaders[cid_int],
                            testloader=testloader)
    return client_fn
