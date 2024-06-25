import xgboost as xgb
from sklearn.metrics import accuracy_score
import numpy as np

class Net:
    def __init__(self, num_classes: int, input_dim: int) -> None:
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.model = None  # We'll initialize this later when we have data
    
    def train_model(self, trainloader, optimizer, epochs, device: str):
        train_features, train_labels = self._loader_to_numpy(trainloader)
        dtrain = xgb.DMatrix(train_features, label=train_labels)
    
        params = self.model.get_params()
        num_rounds = epochs
    
        # Add learning rate scheduler
        evals_result = {}
        self.model = xgb.train(params, dtrain, num_rounds, evals=[(dtrain, 'train')], evals_result=evals_result, xgb_model=self.model)
    
        # Early stopping
        val_losses = evals_result['train']['mlogloss']
        if early_stopping(val_losses):
            print("Early stopping triggered")

    def test_model(self, testloader, device: str):
        test_features, test_labels = self._loader_to_numpy(testloader)
        dtest = xgb.DMatrix(test_features)
        predictions = self.model.predict(dtest)
        predictions = np.argmax(predictions, axis=1)  # Get the index of the max logit
        
        accuracy = accuracy_score(test_labels, predictions)
        avg_loss = np.mean(predictions != test_labels)
        return avg_loss, accuracy

    def _loader_to_numpy(self, loader):
        features, labels = [], []
        for data, label in loader:
            features.append(data.numpy())
            labels.append(label.numpy())
        return np.vstack(features), np.hstack(labels)

def early_stopping(val_losses, patience=10):
    if len(val_losses) > patience:
        if all(val_losses[-1] > val_losses[-(i+2)] for i in range(patience)):
            return True
    return False

# Standalone functions to be imported
def train(net, trainloader, optimizer, epochs, device: str):
    net.train_model(trainloader, optimizer, epochs, device)

def test(net, testloader, device: str):
    return net.test_model(testloader, device)
