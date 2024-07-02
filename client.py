from collections import OrderedDict
from typing import Dict, List
from flwr.common import NDArrays, Scalar
import torch
import flwr as fl
import torch.optim as optim
import xgboost as xgb
import json

from model import Net, train_nn, test_nn, train_xgboost, test_xgboost

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, testloader, num_classes, input_dim, model_type) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.model_type = model_type
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.model_type == 'nn':
            self.model = Net(num_classes, input_dim).to(self.device)
        elif self.model_type == 'xgb':
            self.model = xgb.XGBClassifier()

    def set_parameters(self, parameters: List[NDArrays]):
        if self.model_type == 'nn':
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]) -> List[NDArrays]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters: List[NDArrays], config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        lr = config['lr']
        momentum = config['momentum']
        epochs = config['local_epochs']
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        train_nn(self.model, self.trainloader, self.testloader, optimizer, epochs, self.device)
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        loss, accuracy = test_nn(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader), {'accuracy': accuracy}

def generate_client_fn(trainloaders, valloaders, testloader, num_classes, input_dim, model_types):
    def client_fn(cid: str):
        model_type = model_types[int(cid) % len(model_types)]
        return FlowerClient(trainloader=trainloaders[int(cid)],
                            valloader=valloaders[int(cid)],
                            testloader=testloader,
                            num_classes=num_classes,
                            input_dim=input_dim,
                            model_type=model_type)
    return client_fn
