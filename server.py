from collections import OrderedDict
import flwr as fl
from model import train_xgboost

def get_on_fit_config():
    def fit_config_fn(server_round: int):
        return {'lr': 0.01, 'momentum': 0.99, 'local_epochs': 10}
    return fit_config_fn

def get_evaluate_fn():
    def evaluate_fn(server_round: int, parameters, config):
        model = train_xgboost()
        accuracy = 0.99  # Placeholder for actual accuracy
        return 0.0, {'accuracy': accuracy}
    return evaluate_fn

def start_server():
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,
        fraction_eval=0.1,
        min_fit_clients=2,
        min_eval_clients=2,
        min_available_clients=2,
        eval_fn=get_evaluate_fn(),
        on_fit_config_fn=get_on_fit_config(),
    )
    fl.server.start_server(config={"num_rounds": 10}, strategy=strategy)

if __name__ == "__main__":
    start_server()
