from collections import OrderedDict
from typing import Dict
from flwr.common import NDArrays, Scalar
import torch
import flwr as fl
import torch.optim as optim
import xgboost as xgb

from model import Net, train_nn, test_nn, train_xgboost

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, num_classes, input_dim, model_type='nn'):
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        if model_type == 'nn':
            self.model = Net(num_classes, input_dim)
        elif model_type == 'xgb':
            self.model = None  # XGBoost model will be created during training

    def set_parameters(self, parameters):
        if self.model_type == 'nn':
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        if self.model_type == 'nn':
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        return []

    def fit(self, parameters, config):
        if self.model_type == 'nn':
            self.set_parameters(parameters)
            lr = config['lr']
            momentum = config['momentum']
            epochs = config['local_epochs']
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
            train_nn(self.model, self.trainloader, optimizer, epochs, self.device)
            return self.get_parameters({}), len(self.trainloader), {}
        elif self.model_type == 'xgb':
            X, y = self.get_dataset(self.trainloader)
            model = train_xgboost(X, y, tsz=0.2)
            self.model = model
            return [], len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        if self.model_type == 'nn':
            self.set_parameters(parameters)
            loss, accuracy = test_nn(self.model, self.valloader, self.device)
            return float(loss), len(self.valloader), {'accuracy': accuracy}
        elif self.model_type == 'xgb':
            X, y = self.get_dataset(self.valloader)
            predictions = self.model.predict(xgb.DMatrix(X))
            accuracy = accuracy_score(y, predictions)
            return 0.0, len(self.valloader), {'accuracy': accuracy}

    def get_dataset(self, loader):
        features, labels = [], []
        for feature, label in loader:
            features.append(feature.numpy())
            labels.append(label.numpy())
        X = np.vstack(features)
        y = np.hstack(labels)
        return X, y

def generate_client_fn(trainloaders, valloaders, num_classes, input_dim):
    def client_fn(cid: str):
        model_type = 'nn' if int(cid) == 0 else 'xgb'
        return FlowerClient(trainloader=trainloaders[int(cid)], valloader=valloaders[int(cid)], num_classes=num_classes, input_dim=input_dim, model_type=model_type)
    return client_fn
