from collections import OrderedDict
from typing import Dict
from flwr.common import NDArrays, Scalar
import pandas as pd
import flwr as fl
import model

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, testloader):
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.model = None

    def set_parameters(self, parameters):
        pass

    def get_parameters(self, config: Dict[str, Scalar]):
        pass

    def fit(self, parameters, config):
        self.model = model.train_xgboost()
        return [], len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        accuracy = 0.99  # Placeholder for actual accuracy
        return 0.0, len(self.valloader), {'accuracy': accuracy}

def generate_client_fn(trainloaders, valloaders, testloader):
    def client_fn(cid: str):
        return FlowerClient(trainloader=trainloaders[int(cid)],
                            valloader=valloaders[int(cid)],
                            testloader=testloader)
    return client_fn

def main():
    # Placeholder trainloader, valloader, testloader setup
    trainloaders = [None] * 5
    valloaders = [None] * 5
    testloader = None

    client_fn = generate_client_fn(trainloaders, valloaders, testloader)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client_fn("0"))

if __name__ == "__main__":
    main()
