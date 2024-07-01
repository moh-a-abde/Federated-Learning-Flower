from collections import OrderedDict
from omegaconf import DictConfig
import torch
import xgboost as xgb

from model import Net, test_nn, test_xgboost

def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round: int):
        return {'lr': config.lr, 'momentum': config.momentum, 'local_epochs': config.local_epochs}
    return fit_config_fn

def get_evaluate_fn(num_classes: int, input_dim: int, testloader, model_types):
    def evaluate_fn(server_round: int, parameters, config):
        model_type = model_types[server_round % len(model_types)]
        if model_type == 'nn':
            model = Net(num_classes, input_dim)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)
            loss, accuracy = test_nn(model, testloader, device)
            return loss, {'accuracy': accuracy}
        elif model_type == 'xgb':
            model = xgb.Booster()
            model.load_model(parameters)
            accuracy = test_xgboost(model, testloader)
            return 0.0, {'accuracy': accuracy}
    return evaluate_fn
