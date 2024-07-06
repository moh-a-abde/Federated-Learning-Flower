import flwr as fl
from client import generate_client_fn
from dataset import prepare_dataset
import pandas as pd

def main():
    # Prepare dataset
    num_partitions = 10
    batch_size = 64
    num_classes = 5  
    val_ratio = 0.1
    csv_file = 'data/raw/zeek_live_export_7012024a_final.csv'
    
    trainloaders, valloaders, testloader, input_dim = prepare_dataset(num_partitions, batch_size, num_classes, val_ratio, csv_file)
    
    # Load the full dataset to pass to the XGBoost function
    dataset = pd.read_csv(csv_file)

    client_fn = generate_client_fn(trainloaders, valloaders, testloader, dataset)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_partitions,
        config=fl.server.ServerConfig(num_rounds=10),
    )

if __name__ == "__main__":
    main()
