# server.py
import flwr as fl
from typing import List, Tuple

def get_evaluate_fn():
    def evaluate(weights: List[np.ndarray]) -> Tuple[float, float]:
        # Load your test data and evaluate the model here
        from model import get_model, evaluate_model
        from dataset import load_data
        _, _, test_data, test_labels = load_data()
        model = get_model()
        model.set_weights(weights)
        loss, accuracy = evaluate_model(model, test_data, test_labels)
        return loss, {"accuracy": accuracy}
    return evaluate

if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(evaluate_fn=get_evaluate_fn())
    fl.server.start_server(server_address="127.0.0.1:8080", strategy=strategy)
