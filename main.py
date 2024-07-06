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
    # Load configuration
    config = load_config('conf/base.yaml')

    # Extract parameters from configuration
    num_partitions = config['num_clients']
    batch_size = config['batch_size']
    num_classes = config['num_classes']
    val_ratio = config.get('val_ratio', 0.1)  # Default to 0.1 if not specified
    csv_file = config['data']['file_path']
    num_rounds = config['num_rounds']

    # Prepare dataset
    trainloaders, valloaders, testloader, input_dim = prepare_dataset(
        num_partitions, batch_size, num_classes, val_ratio, csv_file
    )

    # Load the full dataset to pass to the XGBoost function
    dataset = pd.read_csv(csv_file)

    client_fn = generate_client_fn(trainloaders, valloaders, testloader, dataset)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_partitions,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
    )

if __name__ == "__main__":
    main()
