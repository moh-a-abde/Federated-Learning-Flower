from collections import OrderedDict
from omegaconf import DictConfig
import torch
import xgboost as xgb
from sklearn.metrics import accuracy_score

from model import Net, test_nn

def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round: int):
        return {
            'lr': config.lr,
            'momentum': config.momentum,
            'local_epochs': config.local_epochs,
            'model_type': config.model_types[server_round % len(config.model_types)]  # Cycle through model types
        }
    return fit_config_fn

def get_evaluate_fn(num_classes: int, input_dim: int, testloader):
    def evaluate_fn(server_round: int, parameters, config):
        model_type = config.get('model_type', 'nn')
        
        if model_type == 'nn':
            model = Net(num_classes, input_dim)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)
            loss, accuracy = test_nn(model, testloader, device)
            return loss, {'accuracy': accuracy}
        
        elif model_type == 'xgb':
            booster = xgb.Booster()
            booster.load_model(parameters)  # Assuming parameters contain model path or serialized model
            X, y = get_dataset(testloader)
            predictions = booster.predict(xgb.DMatrix(X))
            accuracy = accuracy_score(y, predictions)
            return 0.0, {'accuracy': accuracy}
    
    return evaluate_fn

def get_dataset(loader):
    features, labels = [], []
    for feature, label in loader:
        features.append(feature.numpy())
        labels.append(label.numpy())
    X = np.vstack(features)
    y = np.hstack(labels)
    return X, y
