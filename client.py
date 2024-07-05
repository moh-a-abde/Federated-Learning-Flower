import flwr as fl
import pandas as pd
import model

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model):
        self.model = model

    def get_parameters(self):
        # Return model parameters as a list of NumPy ndarrays
        return self.model.get_params()

    def fit(self, parameters, config):
        # Set model parameters
        self.model.set_params(parameters)

        # Load the dataset
        file_path = 'data/zeek_live_data_merged.csv'
        data = pd.read_csv(file_path)

        # Train the model
        self.model = model.train_xgboost()

        # Return updated model parameters and number of training examples
        return self.model.get_params(), len(data), {}

    def evaluate(self, parameters, config):
        # Set model parameters
        self.model.set_params(parameters)

        # Load the dataset
        file_path = 'data/zeek_live_data_merged.csv'
        data = pd.read_csv(file_path)

        # Evaluate the model
        accuracy = model.evaluate_xgboost(self.model, data)

        # Return loss, accuracy, and number of evaluation examples
        return 0.0, accuracy, len(data)

def main():
    # Load the model
    model_instance = model.train_xgboost()

    # Start Flower client
    fl.client.start_client(server_address="localhost:8080", client=FlowerClient(model_instance).to_client())

if __name__ == "__main__":
    main()
