from typing import Dict, List, Tuple
from flwr.common import NDArrays, Scalar
import flwr as fl
import model
from torch.utils.data import DataLoader

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader: DataLoader, valloader: DataLoader, testloader: DataLoader):
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.model = None

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.model = model.train_xgboost(self.trainloader)
        return [], len(self.trainloader.dataset), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        accuracy = 0.99  # Placeholder for actual accuracy
        return 0.0, len(self.valloader.dataset), {'accuracy': accuracy}

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
