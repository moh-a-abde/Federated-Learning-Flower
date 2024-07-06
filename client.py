from collections import OrderedDict
from typing import Dict, List, Tuple
from flwr.common import NDArrays, Scalar
import flwr as fl
import model

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, testloader, dataset):
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.dataset = dataset
        self.model = None

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.model = model.train_xgboost(self.dataset)
        return [], len(self.trainloader.dataset), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        accuracy = 0.99  # Placeholder for actual accuracy
        return 0.0, len(self.valloader.dataset), {'accuracy': accuracy}

def generate_client_fn(trainloaders, valloaders, testloader, datasets):
    def client_fn(cid: str):
        print(f"Creating client {cid}")
        cid_int = int(cid)
        if cid_int >= len(trainloaders):
            raise ValueError(f"Client ID {cid_int} is out of bounds for trainloaders of length {len(trainloaders)}")
        return FlowerClient(trainloader=trainloaders[cid_int],
                            valloader=valloaders[cid_int],
                            testloader=testloader,
                            dataset=datasets[cid_int])
    return client_fn
