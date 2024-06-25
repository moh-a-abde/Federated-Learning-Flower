from collections import OrderedDict
from typing import Dict
from flwr.common import NDArrays, Scalar
import torch
import flwr as fl
import xgboost as xgb

from model import Net, train, test

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, num_classes, input_dim) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = Net(num_classes, input_dim)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        if parameters is not None:
            if self.model.model is None:
                self.model.model = xgb.Booster()
            self.model.model.load_model(parameters)
        else:
            # Initialize with default parameters if None is provided
            self.model.model = xgb.Booster()
            self.model.model.set_param({'max_depth': 6, 'eta': 0.3, 'objective': 'multi:softprob', 'num_class': self.model.num_classes})

    def get_parameters(self, config: Dict[str, Scalar]):
        if self.model.model is None:
            # Initialize the model with default parameters
            self.model.model = xgb.Booster()
            # You might want to set some default parameters here
            self.model.model.set_param({'max_depth': 6, 'eta': 0.3, 'objective': 'multi:softprob', 'num_class': self.model.num_classes})
        return self.model.model.save_raw()
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        lr = config['lr']
        momentum = config['momentum']
        epochs = config['local_epochs']
        optim = None  # Dummy optimizer
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
