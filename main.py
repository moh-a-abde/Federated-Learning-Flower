import flwr as fl
from client import generate_client_fn

def main():
    # Placeholder trainloader, valloader, testloader setup
    trainloaders = [None] * 10
    valloaders = [None] * 10
    testloader = None

    client_fn = generate_client_fn(trainloaders, valloaders, testloader)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=10,
        config=fl.server.ServerConfig(num_rounds=10),
    )

if __name__ == "__main__":
    main()
