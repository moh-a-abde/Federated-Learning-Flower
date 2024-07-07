import flwr as fl
from client import generate_client_fn
from dataset import prepare_dataset
import pandas as pd
import yaml

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    config = load_config('conf/base.yaml')

    num_partitions = config['num_clients']
    batch_size = config['batch_size']
    num_classes = config['num_classes']
    val_ratio = config.get('val_ratio', 0.1)
    csv_files = [config['data'][f'file_path_{i}'] for i in range(1, 6)]
    num_rounds = config['num_rounds']

    if len(csv_files) != num_partitions:
        raise ValueError(f"Number of CSV files ({len(csv_files)}) does not match number of clients ({num_partitions})")

    trainloaders, valloaders, datasets = prepare_dataset(
        num_partitions, batch_size, num_classes, val_ratio, csv_files
    )

    client_fn = generate_client_fn(trainloaders, valloaders, datasets[0])

    # Define your strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        min_fit_clients=cfg.num_clients_per_round_fit,
        fraction_evaluate=0.5,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, input_dim, valloaders)
)


    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={'num_cpus': 4},
    )

if __name__ == "__main__":
    main()
