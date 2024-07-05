import flwr as fl

def start_server():
    # Start Flower server
    fl.server.start_server(config={"num_rounds": 3})

if __name__ == "__main__":
    start_server()
