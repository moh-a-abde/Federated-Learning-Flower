import flwr as fl
from client import generate_client_fn
from dataset import prepare_dataset

def main():
    # Prepare dataset
    num_partitions = 10
    batch_size = 32
    num_classes = 2  # Adjust based on your dataset
    val_ratio = 0.1
    csv_file = 'data/zeek_live_data_merged.csv'
    
    trainloaders, valloaders, testloader, input_dim = prepare_dataset(num_partitions, batch_size, num_classes, val_ratio, csv_file)

    client_fn = generate_client_fn(trainloaders, valloaders, testloader)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_partitions,
        config=fl.server.ServerConfig(num_rounds=10),
    )

if __name__ == "__main__":
    main()
