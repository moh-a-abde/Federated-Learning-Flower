# client.py
import flwr as fl
from model import get_model, train_model, evaluate_model
from dataset import load_data

class FederatedClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = get_model()
        self.train_data, self.train_labels, self.test_data, self.test_labels = load_data()

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        train_model(self.model, self.train_data, self.train_labels)
        return self.model.get_weights(), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = evaluate_model(self.model, self.test_data, self.test_labels)
        return loss, len(self.test_data), {"accuracy": accuracy}

if __name__ == "__main__":
    client = FederatedClient()
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
