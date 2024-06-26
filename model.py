import xgboost as xgb
from sklearn.metrics import accuracy_score
import numpy as np

import xgboost as xgb
import numpy as np

class Net:
    def __init__(self, num_classes: int, input_dim: int) -> None:
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.model = None
        self.params = {
            'max_depth': 6,
            'eta': 0.3,
            'objective': 'multi:softprob',
            'num_class': num_classes
        }

    def get_model_bytes(self):
        if self.model is None:
            return None
        return self.model.save_raw()

    def set_model_bytes(self, model_bytes):
        if model_bytes is not None:
            self.model = xgb.Booster()
            self.model.load_model(bytearray(model_bytes))
        else:
            self.model = None

    def train_model(self, trainloader, epochs):
        train_features, train_labels = self._loader_to_numpy(trainloader)
        dtrain = xgb.DMatrix(train_features, label=train_labels)
        
        if self.model is None:
            self.model = xgb.train(self.params, dtrain, num_boost_round=epochs)
        else:
            self.model = xgb.train(self.params, dtrain, num_boost_round=epochs, xgb_model=self.model)

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
