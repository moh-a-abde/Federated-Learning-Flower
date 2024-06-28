import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import train_test_split
import time
import copy

class Net:
    def __init__(self, num_classes: int, input_dim: int) -> None:
        self.num_classes = num_classes
        self.model = None

    def fit(self, X, y, num_boost_round: int, params: dict) -> None:
        dtrain = xgb.DMatrix(X, label=y)
        self.model = xgb.train(params, dtrain, num_boost_round=num_boost_round)

    def predict(self, X) -> np.ndarray:
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def predict_proba(self, X) -> np.ndarray:
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
        
    def save_model(self, path: str) -> None:
        self.model.save_model(path)
        
    def load_model(self, path: str) -> None:
        self.model = xgb.Booster()
        self.model.load_model(path)

def train(net, trainloader, optimizer, epochs, device: str):
    X_train, y_train = [], []
    for features, labels in trainloader:
        X_train.append(features.numpy())
        y_train.append(labels.numpy())
    
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    
    train = xgb.DMatrix(X_train, label=y_train)
    
    param = {
        'max_depth': 6,
        'eta': 0.35,
        'objective': 'multi:softmax',
        'num_class': net.num_classes,
        'eval_metric': 'merror',
        'tree_method': 'hist'
    }
    
    cv_params = {
        'params': param,
        'dtrain': train,
        'num_boost_round': 20,
        'nfold': 10,
        'metrics': {'merror'},
        'early_stopping_rounds': 10
    }

    cv_results = xgb.cv(**cv_params)
    best_num_boost_round = cv_results.shape[0]

    net.fit(X_train, y_train, num_boost_round=best_num_boost_round, params=param)
    
    val_losses = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs} completed")
        val_loss = log_loss(y_train, net.predict_proba(X_train))
        val_losses.append(val_loss)
        
        if early_stopping(val_losses):
            print("Early stopping triggered")
            break

    # Save the model after training
    net.save_model('model_state.json')
    
def test(net, testloader, device: str):
    X_test, y_test = [], []
    for features, labels in testloader:
        X_test.append(features.numpy())
        y_test.append(labels.numpy())
    
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    
    predictions = net.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    avg_loss = log_loss(y_test, net.predict_proba(X_test))

    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(report)
    
    return avg_loss, accuracy
    
def early_stopping(val_losses, patience=10):
    if len(val_losses) > patience:
        if all(val_losses[-1] > val_losses[-(i+2)] for i in range(patience)):
            return True
    return False
