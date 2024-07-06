from collections import OrderedDict
from typing import Dict, List
from flwr.common import NDArrays, Scalar
from typing import Tuple
import flwr as fl
import model

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, testloader):
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.model = None

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.model = model.train_xgboost()
        return [], len(self.trainloader.dataset), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        accuracy = 0.99  # Placeholder for actual accuracy
        return 0.0, len(self.valloader.dataset), {'accuracy': accuracy}

def generate_client_fn(trainloaders, valloaders, testloader):
    def client_fn(cid: str):
        return FlowerClient(trainloader=trainloaders[int(cid)],
                            valloader=valloaders[int(cid)],
                            testloader=testloader)
    return client_fn
