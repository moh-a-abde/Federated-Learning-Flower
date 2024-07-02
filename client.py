# client.py
import flwr as fl
import numpy as np
import logging
from model import get_model, train_model, evaluate_model
from dataset import load_data

logging.basicConfig(level=logging.INFO)

class FederatedClient(fl.client.NumPyClient):
    def __init__(self):
        logging.info("Initializing Federated Client")
        self.model = get_model()
        self.train_data, self.train_labels, self.test_data, self.test_labels = load_data()

    def get_parameters(self):
        logging.info("Sending model parameters to server")
        return self.model.get_weights()

    def fit(self, parameters, config):
        logging.info("Received fit request from server")
        self.model.set_weights(parameters)
        train_model(self.model, self.train_data, self.train_labels)
        return self.model.get_weights(), len(self.train_data), {}

    def evaluate(self, parameters, config):
        logging.info("Received evaluate request from server")
        self.model.set_weights(parameters)
        loss, accuracy = evaluate_model(self.model, self.test_data, self.test_labels)
        return loss, len(self.test_data), {"accuracy": accuracy}

if __name__ == "__main__":
    logging.info("Starting Federated Client")
    client = FederatedClient()
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
