from collections import OrderedDict
from typing import Dict
from flwr.common import NDArrays, Scalar
import torch
import flwr as fl
import xgboost as xgb
import numpy as np

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
                # Create a dummy dataset to initialize the model
                dummy_data = np.random.rand(10, self.model.input_dim)
                dummy_labels = np.random.randint(0, self.model.num_classes, 10)
                dtrain = xgb.DMatrix(dummy_data, label=dummy_labels)
                self.model.model = xgb.Booster(model_file=parameters)
                self.model.model.set_param({'max_depth': 6, 'eta': 0.3, 'objective': 'multi:softprob', 'num_class': self.model.num_classes})
            else:
                self.model.model.load_model(parameters)
        else:
            # Initialize with default parameters if None is provided
            self.get_parameters({})  # This will create and initialize the model

    def get_parameters(self, config: Dict[str, Scalar]):
        if self.model.model is None:
            # Create a dummy dataset to initialize the model
            dummy_data = np.random.rand(10, self.model.input_dim)
            dummy_labels = np.random.randint(0, self.model.num_classes, 10)
            dtrain = xgb.DMatrix(dummy_data, label=dummy_labels)
        
            params = {
                'max_depth': 6,
                'eta': 0.3,
                'objective': 'multi:softprob',
                'num_class': self.model.num_classes
            }
        
            # Train the model with one iteration
            self.model.model = xgb.train(params, dtrain, num_boost_round=1)
    
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
